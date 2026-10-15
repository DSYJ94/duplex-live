import unittest
from evals.personaplex_pairs import contrast_candidates


def world(phrase, question='What is the badge phrase?'):
    return {'timeline':{'meta':{'split':'train','kind':'private','project':'Birch North',
        'field':'badge phrase','phrase':phrase},'events':[
            {'ch':'user','kind':'utterance','start_ms':0,'text':question}]},
        'reference_arrivals':[{'ready_at_s':8.4}]}


class PairTests(unittest.TestCase):
    def test_only_same_original_question_with_different_first_word(self):
        a,b,c = (f'teacher-cf-recovery-train-000-{i}' for i in range(3))
        other='teacher-cf-recovery-train-001-0'
        rows={a:world('yellow circle'),b:world('purple star'),c:world('yellow star'),other:world('red triangle')}
        self.assertEqual(contrast_candidates(rows),[(a,b),(b,a),(b,c),(c,b)])
        rows[b]=world('purple star','What is the other field?')
        self.assertEqual(contrast_candidates(rows),[])

    def test_held_out_or_mismatched_record_cannot_be_paired(self):
        a,b='teacher-cf-recovery-train-000-0','teacher-cf-recovery-train-000-1'
        rows={a:world('yellow circle'),b:world('purple star')}
        rows[b]['timeline']['meta']['split']='validation'
        self.assertEqual(contrast_candidates(rows),[])
        rows[b]=world('purple star');rows[b]['timeline']['meta']['project']='Birch South'
        self.assertEqual(contrast_candidates(rows),[])


if __name__ == '__main__':
    unittest.main()
