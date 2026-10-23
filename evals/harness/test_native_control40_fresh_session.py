"""Portable full40 scheduling fixtures; no actual inference or approvals."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.native_control40_fresh_session import cases,preflight,ROOT
from evals.native_control40_fresh_review import SCHEMA


class PortableFull40SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.proof=self.root/'pilot/training';self.baseline=self.root/'context-training-proof-7510331'
        self.review=ROOT/'docs/fixture-only-portable-review.json'
        self.quality=self.root/'quality.json';self.quality.write_text('{}')
        self.identity=dict(checkpoint_sha256='candidate',native_protocol='unchanged')
        self.base_identity=dict(checkpoint_sha256='baseline',native_protocol='unchanged',proof=str(self.baseline))
        self.base=dict(voice_sha256='voice',data_sha256='data')
        self.parent=dict(self.base,warm_start=self.base_identity)
        rows=cases(self.proof,self.baseline)
        self.manifest=dict(rows=[dict(scenario_id=r['scenario']) for r in rows[:20]])

    def call(self,manifest=None,parent=None):
        rows=cases(self.proof,self.baseline)
        def verified(proof):
            return ((self.identity,parent or self.parent) if proof==self.proof
                    else (self.base_identity,self.base))
        with patch('evals.native_fresh_review.validate',return_value=dict(schema=SCHEMA,
                native_asr_placement='shared-gpu-evaluation')), \
                patch('evals.context_rollout_conditioning.verify_checkpoint',side_effect=verified), \
                patch('evals.native_fresh_source_reapproval.approved_inputs',return_value=manifest or self.manifest) as inputs, \
                patch('evals.native_control40_fresh_session.sha256',return_value='fixture-digest'), \
                patch('evals.native_control40_fresh_session.cases',return_value=rows):
            result=preflight(self.proof,self.review,self.root/'data',self.quality,self.root/'preflight.json')
            self.assertEqual(inputs.call_count,1)
            return result

    def test_all40_use_actual_proofs_and_explicit_serial_resource(self):
        rows=cases(self.proof,self.baseline)
        self.assertEqual([r['index'] for r in rows],list(range(40)))
        self.assertEqual([r['scenario'] for r in rows[:20]],[r['scenario'] for r in rows[20:]])
        self.assertEqual({r['proof'] for r in rows},{str(self.proof.resolve()),str(self.baseline.resolve())})
        for row in rows:
            self.assertEqual(row['required_gpus'],1)
            self.assertEqual(row['assigned_cuda_devices'],['0'])
            self.assertNotIn('checkpoint_job',row)
            self.assertNotIn('answer',row)
            self.assertNotIn('worker_only_world',row)
            self.assertNotIn('expected',row)

    def test_wrong_parent_or_same_model_rejected(self):
        for candidate,baseline in ((self.baseline,self.baseline),(self.proof,self.root/'other-baseline')):
            with self.assertRaises(ValueError):cases(candidate,baseline)

    def test_changed_sealed_plan_rejected(self):
        with patch('evals.native_control40_fresh_session.sha256',return_value='different'):
            with self.assertRaises(ValueError):cases(self.proof,self.baseline)

    def test_unreviewed_candidate_cannot_read_quality_or_fresh_audio(self):
        with patch('evals.native_fresh_review.validate',side_effect=ValueError('no complete review')), \
                patch('evals.native_fresh_source_reapproval.approved_inputs') as inputs, \
                patch('evals.context_rollout_conditioning.verify_checkpoint') as checkpoint:
            with self.assertRaisesRegex(ValueError,'no complete review'):
                preflight(self.proof,self.review,self.root/'data',self.quality,self.root/'preflight.json')
            inputs.assert_not_called();checkpoint.assert_not_called()
        self.assertFalse((self.root/'preflight.json').exists())

    def test_complete_preflight_is_measurement_only_and_does_not_overwrite(self):
        result=self.call()
        self.assertEqual(result['batches'],[[i] for i in range(40)])
        self.assertFalse(result['actual_inference_executed'])
        self.assertFalse(result['behavior_or_adoption_approved'])
        self.assertFalse(result['training_approved']);self.assertTrue(result['no_passing_subset'])
        with self.assertRaises(FileExistsError):self.call()

    def test_partial_or_duplicate_inputs_rejected(self):
        for rows in (self.manifest['rows'][:-1],self.manifest['rows'][:-1]+[self.manifest['rows'][0]]):
            with self.subTest(count=len(rows)),self.assertRaises(ValueError):self.call(manifest=dict(rows=rows))

    def test_different_voice_or_parent_identity_rejected_before_source_reads(self):
        for field,value in [('voice_sha256','different'),('data_sha256','different'),
                ('warm_start',dict(self.base_identity,checkpoint_sha256='different'))]:
            parent=copy.deepcopy(self.parent);parent[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.call(parent=parent)


if __name__=='__main__':unittest.main()
