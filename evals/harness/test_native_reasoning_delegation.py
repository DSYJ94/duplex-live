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

    def test_spoken_question_answer_and_lifecycle_annotations_remain_one_canonical_source(self):
        original=build()
        mutations={
            'question':lambda p:p['cases'][1]['inference_inputs']['authored_user_utterances'].__setitem__(0,'Choose the most expensive bundle instead.'),
            'answer':lambda p:p['cases'][1]['annotations_not_model_inputs'].update(expected_final_spoken_words='Choose the wrong items.'),
            'pending_answer':lambda p:p['cases'][1]['annotations_not_model_inputs'].update(expected_pending_basic_words='Two badges cost nine tokens.'),
            'lifecycle_delay':lambda p:p['cases'][2]['annotations_not_model_inputs'].update(minimum_result_delivery_delay_s=0.),
            'lifecycle_dispatch':lambda p:p['cases'][2]['annotations_not_model_inputs'].update(lifecycle_event_after_actual_dispatch=False),
            'revision':lambda p:p['cases'][3]['annotations_not_model_inputs']['actual_worker_returns_required'][1].update(revision=1),
            'delivery':lambda p:p['cases'][2]['annotations_not_model_inputs']['actual_worker_returns_required'][0].update(deliverable=True),
            'matched_instance':lambda p:p['cases'][1].update(matched_instance=7),
            'voice':lambda p:p['cases'][1].update(user_voice='unapproved-voice'),
        }
        for name,mutate in mutations.items():
            p=copy.deepcopy(original);mutate(p)
            with self.subTest(mutation=name),self.assertRaises(ValueError):validate(p)

    def test_source_cannot_weaken_retention_or_claim_owner_or_heldout_inputs(self):
        original=build()
        for field,value in [('owner_conversations_used',True),
                            ('existing_folds_or_heldouts_read_or_changed',True),
                            ('existing_control40_or_retention_gates_changed',True)]:
            p=copy.deepcopy(original);p[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate(p)
        for field in original['required_retention']:
            p=copy.deepcopy(original);p['required_retention'][field]=0
            with self.subTest(retention=field),self.assertRaises(ValueError):validate(p)


if __name__=='__main__':unittest.main()
