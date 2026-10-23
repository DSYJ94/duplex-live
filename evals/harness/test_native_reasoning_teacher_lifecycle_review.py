import copy
import unittest
from data.synthetic.native_reasoning_delegation import build,answer
from evals.native_reasoning_teacher_lifecycle_review import case_review,bundle_errors


class LifecycleScoring(unittest.TestCase):
    def test_cancel_fails_when_original_was_delivered_even_if_basic_reply_was_correct(self):
        row=build()['cases'][2]
        original=dict(phase='wire_sent',cancelled=False,receipt=dict(operation='respond_to_user'),wire_write_completed_at_s=20.)
        r=case_review(row,dict(tasks={'original':original}));self.assertFalse(r['structural_pass'])
        self.assertTrue(any('Missing actual' in x for x in r['reasons']));self.assertTrue(any('Original result was delivered' in x for x in r['reasons']))

    def test_cancel_requires_actual_retirement_and_does_not_claim_remote_abort(self):
        row=build()['cases'][2];tasks=dict(original=dict(phase='stale_completion_preserved',cancelled=True),
            cancel=dict(phase='wire_sent',cancelled=False,wire_write_completed_at_s=10.,receipt=dict(
                operation='cancel_reasoning',retired_requests=[dict(task_id='original',revision=1)],remote_execution_aborted=False)))
        self.assertTrue(case_review(row,dict(tasks=tasks))['structural_pass'])
        broken=copy.deepcopy(tasks);broken['cancel']['receipt']['remote_execution_aborted']=True
        self.assertFalse(case_review(row,dict(tasks=broken))['structural_pass'])

    def test_replacement_must_advance_revision_and_use_changed_oracle(self):
        row=build()['cases'][3];target=row['annotations_not_model_inputs']['actual_worker_returns_required'][-1]['oracle']
        tasks=dict(original=dict(phase='stale_completion_preserved',cancelled=True),replacement=dict(
            phase='wire_sent',cancelled=False,wire_write_completed_at_s=10.,receipt=dict(operation='replace_reasoning',
                retired_requests=[dict(task_id='original',revision=1)],remote_execution_aborted=False,revision=2,content=answer(target))))
        self.assertTrue(case_review(row,dict(tasks=tasks))['structural_pass'])
        tasks['replacement']['receipt']['revision']=1;self.assertFalse(case_review(row,dict(tasks=tasks))['structural_pass'])
        self.assertTrue(bundle_errors('Choose only a ribbon for zero tokens and zero points.',target))


if __name__=='__main__':unittest.main()
