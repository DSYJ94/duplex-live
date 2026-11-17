import copy
import unittest
from training.personaplex.broad_corpus_partitions import propose


def row(identity,question,*,source='public/a',group=None,answer='Source answer'):
    return dict(id=identity,source_dataset=source,source_group_id=group or identity,content_sha256='a'*64,
                admitted_for_training=False,split=None,official_source_split='train',
                turns=[dict(role='user',text=question),dict(role='assistant',text=answer)])


class BroadPartitionsTest(unittest.TestCase):
    def test_parent_exact_and_numeric_variants_cannot_cross_proposed_fold(self):
        rows=[row('a','Compute the sum of 12 and 9.'),row('b','Compute the sum of 14 and 3.',source='public/b'),
              row('c','A different question',group='shared'),row('d','Another topic',group='shared'),
              row('e','Explain a cell'),row('f','Explain a cell',source='public/b')]
        result=propose(rows);assign={entry['id']:entry for entry in result['assignments']}
        for left,right in [('a','b'),('c','d'),('e','f')]:
            self.assertEqual(assign[left]['family_id'],assign[right]['family_id'])
            self.assertEqual(assign[left]['proposed_fold'],assign[right]['proposed_fold'])
        self.assertEqual(result['report']['families'],3)
        self.assertEqual(result['report']['cross_source_families'],2)
        self.assertFalse(result['report']['semantic_paraphrase_clustering_complete'])

    def test_answers_do_not_supply_grouping_or_fold(self):
        rows=[row('a','First question'),row('b','Second question')]
        before=propose(rows)
        changed=copy.deepcopy(rows)
        changed[0]['turns'][1]['text']='The same answer as the second question'
        changed[1]['turns'][1]['text']=changed[0]['turns'][1]['text']
        self.assertEqual(before,propose(list(reversed(changed))))

    def test_official_holdouts_or_preassigned_data_are_rejected(self):
        for field,value in [('official_source_split','validation'),('split','DEV'),('admitted_for_training',True)]:
            record=row('a','Question');record[field]=value
            with self.assertRaises(ValueError):propose([record])


if __name__=='__main__':unittest.main()
