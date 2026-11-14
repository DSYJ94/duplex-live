import unittest
import numpy as np
from data.personaplex.expanded_source_export import admission,resample_native
from data.personaplex.source_export import pack


class ExpandedStereoExport(unittest.TestCase):
    def test_unapproved_or_partial_cohort_cannot_export(self):
        with self.assertRaises(ValueError):admission({'owner_approved':False},{},{},{})

    def test_full_clock_conversion_preserves_both_native_channels(self):
        user=resample_native(np.full(16000,.1,dtype=np.float32),16000)
        assistant=resample_native(np.full(8000,-.2,dtype=np.float32),16000)
        self.assertEqual(len(user),24000);self.assertEqual(len(assistant),12000)
        audio,labels,start=pack(user,assistant,[dict(text='yes',start=.04,end=.2)])
        np.testing.assert_array_equal(audio[:len(user),1],user)
        np.testing.assert_array_equal(audio[start:start+len(assistant),0],assistant)
        self.assertEqual(start%1920,0);self.assertEqual(len(audio)%1920,0)
        self.assertAlmostEqual(labels[0][1][0],start/24000+.04)

    def test_wrong_clock_and_clipping_are_rejected_without_attenuation(self):
        for rate,value in [(24000,.1),(16000,1.)]:
            with self.subTest(rate=rate,value=value),self.assertRaises(ValueError):resample_native(np.full(1600,value,dtype=np.float32),rate)


if __name__=='__main__':unittest.main()
