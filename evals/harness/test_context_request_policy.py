import copy
import unittest
from evals.context_request_policy import native_prelead, native_prelead_epad


class RequestPolicyTest(unittest.TestCase):
    def scene(self):
        codes = [3] * 40
        codes[20], codes[25], codes[26] = 4, 10, 11
        side = {'timeline': {'meta': {'kind': 'private', 'split': 'train', 'project': 'Birch'},
            'events': [
                {'ch': 'user', 'kind': 'utterance', 'actual_start_ms': 0,
                 'actual_end_ms': 1000, 'text': 'Birch'},
                {'ch': 'assistant', 'kind': 'marker', 'start_ms': 1600, 'text': ''}]},
            'user_words': [{'text': 'Birch', 'start_s': .2, 'end_s': .8}],
            'alignments': [['Check', [2., 2.08], 'SPEAKER_MAIN'],
                           ['now', [2.08, 2.16], 'SPEAKER_MAIN']],
            'reference_arrivals': [{'ready_at_s': 2.8, 'text': 'Actual value.'}]}
        return codes, side

    def test_derived_controls_retain_words_and_source_clocks(self):
        codes, side = self.scene()
        before = copy.deepcopy(side)
        result, records = native_prelead(codes, side, {'Check': [10], 'now': [11]}.__getitem__)
        self.assertEqual([f for f, t in enumerate(result) if t == 4], [24])
        self.assertEqual(result[25:27], codes[25:27])
        self.assertEqual(codes[20], 4)
        self.assertEqual(side, before)
        self.assertEqual(records[0]['source_dispatch_frame'], 20)
        self.assertFalse(records[0]['actual_worker_dispatch_remeasured'])

    def test_epad_ret_variant_retains_other_word_cues(self):
        codes, side = self.scene()
        codes[24], codes[30] = 0, 0
        result, records = native_prelead_epad(codes, side, {'Check': [10], 'now': [11]}.__getitem__)
        self.assertEqual(result[23:27], [0, 4, 10, 11])
        self.assertEqual(result[30], 0)
        self.assertEqual(records[0]['derived_end_padding_frame'], 23)
        codes[23] = 12
        side['alignments'].insert(0, ['Earlier', [1.84, 1.9], 'SPEAKER_MAIN'])
        with self.assertRaisesRegex(RuntimeError, 'EPAD/RET'):
            native_prelead_epad(codes, side, {'Earlier': [12], 'Check': [10], 'now': [11]}.__getitem__)

    def test_project_must_be_in_arrived_user_words(self):
        codes, side = self.scene()
        side['user_words'][0]['text'] = 'Some project'
        with self.assertRaisesRegex(RuntimeError, 'identification'):
            native_prelead(codes, side, {'Check': [10], 'now': [11]}.__getitem__)

    def test_already_delivered_result_cannot_be_moved_before_request(self):
        codes, side = self.scene()
        side['reference_arrivals'][0]['ready_at_s'] = 1.8
        with self.assertRaisesRegex(RuntimeError, 'already supplied'):
            native_prelead(codes, side, {'Check': [10], 'now': [11]}.__getitem__)

    def test_heldout_labels_cannot_be_derived(self):
        codes, side = self.scene()
        side['timeline']['meta']['split'] = 'validation'
        with self.assertRaisesRegex(RuntimeError, 'TRAIN-only'):
            native_prelead(codes, side, {'Check': [10], 'now': [11]}.__getitem__)

    def test_cancellation_waits_for_complete_user_instruction(self):
        codes, side = self.scene()
        side['timeline']['meta']['kind'] = 'cancel'
        codes += [3] * 30
        codes[45], codes[50] = 4, 15
        side['timeline']['events'] += [
            {'ch': 'user', 'kind': 'utterance', 'actual_start_ms': 3000,
             'actual_end_ms': 4000, 'text': 'Cancel that lookup.'},
            {'ch': 'assistant', 'kind': 'marker', 'start_ms': 3600, 'text': ''}]
        side['alignments'].append(['Cancelled', [4., 4.1], 'SPEAKER_MAIN'])
        side['user_words'].append({'text': 'Cancel', 'start_s': 3.1, 'end_s': 3.5})
        encode = {'Check': [10], 'now': [11], 'Cancelled': [15]}.__getitem__
        with self.assertRaisesRegex(RuntimeError, 'complete user instruction'):
            native_prelead(codes, side, encode)
        side['timeline']['events'][-2]['actual_end_ms'] = 3800
        result, records = native_prelead(codes, side, encode)
        self.assertEqual([f for f, t in enumerate(result) if t == 4], [24, 49])
        self.assertEqual(records[1]['operation'], 'cancel')
        side['user_words'].pop()
        with self.assertRaisesRegex(RuntimeError, 'complete user instruction'):
            native_prelead(codes, side, encode)

    def test_prelead_collision_never_drops_a_spoken_token(self):
        codes, side = self.scene()
        # A multi-subword word starts before the source marker and occupies
        # the frame before the following word. Never overwrite its last piece.
        codes[19:27] = [12, 4, 13, 14, 15, 16, 10, 11]
        side['alignments'].insert(0, ['Earlier', [1.52, 2.], 'SPEAKER_MAIN'])
        with self.assertRaisesRegex(RuntimeError, 'overwrite'):
            native_prelead(codes, side,
                {'Earlier': [12, 13, 14, 15, 16], 'Check': [10], 'now': [11]}.__getitem__)



if __name__ == '__main__': unittest.main()
