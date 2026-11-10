import unittest
from training.personaplex.matched_storage_quota import parse_quota,FILESYSTEM


class ProviderQuota(unittest.TestCase):
    actual='Disk quotas for user dsyj94.u6ada (uid 1483809154):\n'+FILESYSTEM+' 4455744 0 105468748 20686 10000000 15000000\n'

    def test_exact_observed_provider_record_conservatively_bounds_whole_home(self):
        value=parse_quota(self.actual,1483809154)
        self.assertEqual(value['home_bytes'],4455744*1024)
        self.assertEqual(value['run_bytes'],value['home_bytes'])
        self.assertLessEqual(value['home_bytes'],32*1024**3)

    def test_ambiguous_wrong_owner_human_units_or_over32_home_fail_closed(self):
        bad=(self.actual.replace('1483809154','12'),self.actual+self.actual,
             self.actual.replace('4455744','4.3G'),self.actual.replace('4455744','33554433'),
             self.actual.replace('4455744','4455744*'),self.actual.replace(FILESYSTEM,'other:/home'))
        for text in bad:
            with self.assertRaises((ValueError,RuntimeError)):parse_quota(text,1483809154)


if __name__=='__main__':unittest.main()
