import unittest
from evals.native_fresh_asr_control_v2 import agreement
from evals.native_fresh_inputs_v2 import specifications


class SecondCollectionASRControlTests(unittest.TestCase):
    def test_room_rune_is_a_failed_word_not_a_spelling_alias(self):
        self.assertEqual(len(specifications()),20)
        a=agreement('Check the room sign for Project Nine.','Check the rune sign for Project 9.')
        self.assertEqual(a['raw_wer'],1/7)
        self.assertEqual(a['spelling_only_wer'],1/7)
        self.assertEqual(a['spelling_equivalents'],{'color':'colour'})


if __name__=='__main__':unittest.main()
