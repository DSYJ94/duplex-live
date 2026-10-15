import unittest
from evals.lychee_codec_data import word_groups


class SourceBoundsTests(unittest.TestCase):
    def test_groups_retain_independently_measured_words_and_bounds(self):
        words=[['The',[.2,.4],'SPEAKER_MAIN'],['marker',[.41,.7],'SPEAKER_MAIN'],
            ['yellow',[1.4,1.7],'SPEAKER_MAIN'],['circle',[1.72,2.1],'SPEAKER_MAIN']]
        groups=word_groups(words,2.4)
        self.assertEqual(groups,[[('The',[.2,.4]),('marker',[.41,.7])],
            [('yellow',[1.4,1.7]),('circle',[1.72,2.1])]])

    def test_invalid_or_overlapping_source_words_fail(self):
        for words in ([['missing end',[.1,4.],'SPEAKER_MAIN']],
            [['one',[.1,.5],'SPEAKER_MAIN'],['overlap',[.3,.7],'SPEAKER_MAIN']]):
            with self.assertRaises(ValueError):word_groups(words,1.)


if __name__=='__main__':unittest.main()
