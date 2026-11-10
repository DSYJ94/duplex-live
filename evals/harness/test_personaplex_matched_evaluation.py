import unittest
from training.personaplex.matched_evaluation import complete_collection,complete_frames


class CompleteNativeCohort(unittest.TestCase):
    def test_missing_duplicated_or_changed_input_cannot_pass_full_cohort(self):
        rows=[dict(id=str(i),input_sha256='sha-'+str(i)) for i in range(32)]
        cases=[dict(r,complete=True) for r in rows]
        self.assertEqual(len(complete_collection(dict(cases=cases,required_cases=32),rows)),32)
        bad=[cases[:-1],cases[:-1]+cases[:1],cases[:-1]+[dict(cases[-1],input_sha256='changed')],
             cases[:-1]+[dict(cases[-1],complete=False)]]
        for values in bad:
            with self.assertRaisesRegex(ValueError,'All32'):complete_collection(dict(cases=values,required_cases=32),rows)

    def test_contiguous_prefix_is_not_complete_native_audio(self):
        complete_frames([dict(sequence=i) for i in range(3)],5760)
        for trace,samples in (([dict(sequence=0)],5760),([],0),([dict(sequence=0),dict(sequence=2)],3840)):
            with self.assertRaisesRegex(ValueError,'Every contiguous'):complete_frames(trace,samples)


if __name__=='__main__':unittest.main()
