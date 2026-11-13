import copy
import unittest
from data.venus.native_curriculum_intake import build


def source():
    rows=[]
    for fold, count in [('TRAIN',1008),('DEV',64)]:
        for i in range(count):
            rows.append(dict(id=fold+str(i),fold=fold,question=fold+' question '+str(i),response='The answer is2.',
                source_kind='authored_verified_arithmetic',template=i%16))
    return dict(schema='dl-reasoning-text-curriculum-v1',rows=rows,official_test_used=False,native_audio_created=False)


class Intake(unittest.TestCase):
    def test_preserves_words_folds_and_exposes_coverage_gaps(self):
        old=source(); value=build(old,'a'*64)
        self.assertEqual(value['counts'],dict(TRAIN=1008,DEV=64))
        self.assertEqual(value['rows'][0]['source_record'],old['rows'][0])
        self.assertEqual(len(value['shared_template_groups']),16)
        self.assertFalse(value['training_ready'])
        self.assertTrue(all(r['routing_target'] is None for r in value['rows']))
    def test_crossfold_duplicate_and_subset_rejected(self):
        old=source();old['rows'][-1]['question']=old['rows'][0]['question']
        with self.assertRaises(ValueError):build(old,'a'*64)
        old=source();old['rows'].pop()
        with self.assertRaises(ValueError):build(old,'a'*64)
    def test_unapproved_source_rejected(self):
        old=source();old['official_test_used']=True
        with self.assertRaises(ValueError):build(old,'a'*64)


if __name__=='__main__':unittest.main()
