import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from training.personaplex.corpus_semantic_review import parse,review,collect


def row(identity='record-1'):
    return dict(id=identity,turns=[dict(role='user',text='Supplied context: The red box holds a key. What is inside?'),
                                  dict(role='assistant',text='A key.')])


def verdict():
    return dict(domain='everyday',task='extraction',verdict='supported',answerability='answerable',
                support='supplied_context',evidence_quotes=['The red box holds a key.'],notes='The supplied context supports this answer.')


def actor(value=None):
    return Mock(identity={'repository':'test-double-not-native'},answer=Mock(return_value=dict(
        text=json.dumps(verdict() if value is None else value),ended_with_EOS=True,generation_budget_exhausted=False)))


class CorpusSemanticReviewTest(unittest.TestCase):
    def test_host_rejects_running_parent_before_reading_or_loading_models(self):
        from unittest.mock import patch
        from training.personaplex.corpus_review_host import require_parent
        with patch('training.personaplex.corpus_review_host.parent_provider',return_value={'JobState':'RUNNING','ExitCode':'0:0'}):
            with self.assertRaises(ValueError):require_parent()

    def test_user_context_evidence_is_not_teacher_self_confirmation(self):
        self.assertEqual(parse(json.dumps(verdict()),row())['support'],'supplied_context')
        for quotes in ([],['A key.'],['This never appeared.']):
            value=dict(verdict(),evidence_quotes=quotes)
            with self.assertRaises(ValueError):parse(json.dumps(value),row())
        with self.assertRaises(ValueError):parse('{"domain":"everyday","domain":"science"}',row())

    def test_complete_review_never_auto_admits_training(self):
        record=review(row(),actor())
        self.assertEqual(record['review_status'],'structured_review_complete')
        self.assertFalse(record['semantic_qualified']);self.assertFalse(record['training_admitted'])
        incomplete=actor();incomplete.answer.return_value['ended_with_EOS']=False
        self.assertEqual(review(row(),incomplete)['review_status'],'incomplete_generation')
        invalid=actor(dict(verdict(),domain='invented'))
        self.assertEqual(review(row(),invalid)['review_status'],'invalid_review')

    def test_all_disjoint_shards_preserve_every_record_and_bad_outputs(self):
        rows=[row('record-%02d'%i) for i in range(9)]
        completed=[]
        with tempfile.TemporaryDirectory() as temp:
            for rank in range(4):
                folder=Path(temp)/str(rank)
                result=collect(rows,folder,actor(),lambda needed:None,rank=rank,world_size=4,source='test')
                completed.extend(entry['id'] for entry in result['results'])
                self.assertFalse(result['training_admitted'])
                self.assertTrue((folder/'terminal.json').is_file())
                for entry in result['results']:
                    self.assertTrue((folder/entry['file']).is_file())
        self.assertEqual(sorted(completed),sorted(row['id'] for row in rows))
        self.assertEqual(len(completed),len(set(completed)))


if __name__=='__main__':unittest.main()
