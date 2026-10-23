import copy
import json
import unittest

from data.synthetic.native_reasoning_delegation import build,solve,validate


class ReasoningDelegationSource(unittest.TestCase):
    def test_every_oracle_matches_an_independent_bitmask_search(self):
        plan=build();checked=0
        for row in plan['cases']:
            for result in row['annotations_not_model_inputs']['actual_worker_returns_required']:
                problem=result['problem'];valid=[]
                for mask in range(64):
                    chosen=[x for i,x in enumerate(problem['items']) if (mask>>i)&1]
                    if len(chosen)!=3:continue
                    names=sorted(x['name'] for x in chosen);cost=sum(x['cost'] for x in chosen)
                    if (cost>problem['budget'] or problem['required'] not in names
                            or all(x in names for x in problem['incompatible'])):continue
                    valid.append((sum(x['points'] for x in chosen),cost,names))
                best=sorted(valid,key=lambda x:(-x[0],x[1],x[2]))[0]
                self.assertEqual(result['oracle'],dict(points=best[0],cost=best[1],names=best[2]));checked+=1
        self.assertEqual(checked,32)

    def test_replacement_really_changes_answer_and_old_result_is_not_deliverable(self):
        for row in build()['cases']:
            labels=row['annotations_not_model_inputs'];results=labels['actual_worker_returns_required']
            if row['family']=='replace-reasoning':
                self.assertNotEqual(results[0]['oracle']['names'],results[1]['oracle']['names'])
                self.assertFalse(results[0]['deliverable']);self.assertTrue(results[1]['deliverable'])
            if row['family']=='cancel-reasoning':self.assertFalse(results[0]['deliverable'])
            payload=json.dumps(row['inference_inputs'])
            self.assertNotIn('oracle',row['inference_inputs']);self.assertNotIn('expected_final',payload)
            self.assertNotIn(labels['expected_final_spoken_words'],payload)

    def test_subset_or_changed_oracle_or_forced_delegation_cannot_be_source_approval(self):
        original=build()
        for mode in ('subset','oracle','forced','route','input'):
            p=copy.deepcopy(original)
            if mode=='subset':p['cases'].pop()
            if mode=='oracle':p['cases'][1]['annotations_not_model_inputs']['actual_worker_returns_required'][0]['oracle']['points']+=1
            if mode=='forced':p['cases'][1]['annotations_not_model_inputs']['native_request_forcing_allowed']=True
            if mode=='route':p['cases'][1]['annotations_not_model_inputs']['worker_route']='delegate'
            if mode=='input':p['cases'][1]['inference_inputs']['oracle']='not allowed'
            with self.subTest(mode=mode),self.assertRaises(ValueError):validate(p)
        self.assertFalse(original['training_approved']);self.assertEqual(original['api_calls'],0)


if __name__=='__main__':unittest.main()
