"""Cohort and gate-order fixtures; never read the real sealed plan or audio."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals.native_control40_fresh_review import cohort,SCHEMA,REASONING_SCHEMA,REASONING_ROLES
from evals.native_control40_fresh_session import cases,preflight,ROOT
from evals.moshi_model_download import sha256
from portable.context.native_fresh_development import plan_roles,preflight_document


class ReasoningFreshCohortTests(unittest.TestCase):
    def test_explicit_candidate_and_parent_binding(self):
        self.assertEqual(cohort(REASONING_SCHEMA),(REASONING_ROLES,
            'dl-native-reasoning32-retained154-stream49-control40-proof-v1'))
        self.assertEqual(cohort(SCHEMA)[0],('baseline520','candidate-control40'))
        self.assertEqual(plan_roles('dl-native-reasoning32-fresh-paired40-preflight-v1'),REASONING_ROLES)
        for bad in ('', 'reasoning32', 'dl-native-v2-development-review-v1'):
            with self.assertRaises(ValueError):cohort(bad)
            with self.assertRaises(ValueError):plan_roles(bad)

    def test_complete40_with_synthetic_placeholder_plan_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=Path(tmp)/'placeholder-plan.json'
            plan.write_text(json.dumps(dict(do_not_train=True,training_approved=False,
                cases=[dict(id='fixture-%02d'%i,kind='ordinary',backend='observe-only') for i in range(20)])))
            with patch('evals.native_control40_fresh_session.PLAN',plan), \
                    patch('evals.native_control40_fresh_session.PLAN_SHA256',sha256(plan)):
                rows=cases(Path(tmp)/'candidate',Path(tmp)/'context-training-proof-7510331',REASONING_ROLES[1])
            self.assertEqual([r['index'] for r in rows],list(range(40)))
            self.assertEqual([r['scenario'] for r in rows[:20]],[r['scenario'] for r in rows[20:]])
            self.assertEqual({r['role'] for r in rows},set(REASONING_ROLES))
            self.assertTrue(all(r['required_gpus']==1 for r in rows))
            self.assertTrue(all(not {'answer','expected','worker_only_world'} & r.keys() for r in rows))

    def test_unsupported_role_rejected_before_any_plan_read(self):
        with patch('pathlib.Path.read_text',side_effect=AssertionError('sealed plan must remain unread')):
            with self.assertRaises(ValueError):cases('/candidate','/baseline','candidate-unreviewed')

    def test_missing_review_never_reads_fresh_quality_or_plan(self):
        with patch('evals.native_fresh_review.validate',side_effect=ValueError('review required')), \
                patch('evals.native_control40_fresh_session.cases') as planned, \
                patch('evals.native_fresh_source_reapproval.approved_inputs') as inputs, \
                patch('evals.context_rollout_conditioning.verify_checkpoint') as restored:
            with self.assertRaisesRegex(ValueError,'review required'):
                preflight('/candidate',ROOT/'docs/unapproved-fixture.json','/sealed','/quality','/out')
            planned.assert_not_called();inputs.assert_not_called();restored.assert_not_called()

    def test_reviewed_reasoning_preflight_preserves_complete_same_resource_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();proof=root/'candidate';baseline=root/'context-training-proof-7510331'
            review=root/'review.json';quality=root/'quality.json'
            review.write_text('{}');quality.write_text('{}')
            base_identity=dict(proof=str(baseline),checkpoint_sha256='base',native_protocol='fixed')
            identity=dict(checkpoint_sha256='candidate',native_protocol='fixed')
            base=dict(voice_sha256='voice',data_sha256='data')
            parent=dict(base,warm_start=base_identity)
            rows=[dict(index=i,scenario='fixture-%02d'%(i%20),role=REASONING_ROLES[i//20]) for i in range(40)]
            def restored(path):
                return (identity,parent) if path==proof else (base_identity,base)
            with patch('evals.native_control40_fresh_session.ROOT',root), \
                    patch('evals.native_fresh_review.validate',return_value=dict(schema=REASONING_SCHEMA,
                        native_asr_placement='shared-gpu-evaluation')), \
                    patch('evals.context_rollout_conditioning.verify_checkpoint',side_effect=restored), \
                    patch('evals.native_control40_fresh_session.cases',return_value=rows) as planned, \
                    patch('evals.native_fresh_source_reapproval.approved_inputs',return_value=dict(
                        rows=[dict(scenario_id=r['scenario']) for r in rows[:20]])):
                result=preflight(proof,review,root/'unread-inputs',quality,root/'preflight.json')
            self.assertEqual(planned.call_args.args[2],'candidate-reasoning32')
            self.assertEqual(result['schema'],'dl-native-reasoning32-fresh-paired40-preflight-v1')
            self.assertTrue(result['complete40']);self.assertTrue(result['no_passing_subset'])
            self.assertEqual(result['cases'],rows);self.assertEqual(result['maximum_gpus'],1)
            self.assertEqual(result['gpu_mode'],'serial-one-gpu')
            self.assertFalse(result['training_approved']);self.assertFalse(result['checkpoint_adopted'])
            self.assertFalse(result['actual_inference_executed'])

    def test_collector_binds_reasoning_role_to_preflight_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);review=out/'review.json';quality=out/'quality.json'
            review.write_text('{}');quality.write_text('{}')
            rows=[dict(index=i,role='baseline520' if i<20 else 'candidate-reasoning32') for i in range(40)]
            plan=dict(schema='dl-native-reasoning32-fresh-paired40-preflight-v1',complete40=True,
                cases=rows,candidate_proof=str(out/'candidate'),baseline_proof=str(out/'context-training-proof-7510331'),
                gpu_mode='serial-one-gpu',maximum_gpus=1,behavior_review_sha256=sha256(review),
                source_quality_sha256=sha256(quality),source_profile='owner-approved-turbo-spelling-v1',
                evaluation_only=True,training_approved=False,checkpoint_adopted=False)
            path=out/'preflight.json';path.write_text(json.dumps(plan))
            args=SimpleNamespace(out=out,proof=out/'candidate',review=review,quality=quality,
                preflight_sha256=sha256(path))
            with patch('evals.native_control40_fresh_session.cases',return_value=rows) as planned:
                self.assertEqual(preflight_document(args),plan)
                self.assertEqual(planned.call_args.args[2],'candidate-reasoning32')
                plan['cases']=rows[:-1];path.write_text(json.dumps(plan));args.preflight_sha256=sha256(path)
                with self.assertRaises(ValueError):preflight_document(args)


if __name__=='__main__':unittest.main()
