"""A new corpus cannot omit retained objectives or bypass full restoration."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals.native_reasoning_stage import OPTIONS, SCHEMA, validate, check_readiness, check_bindings
from evals.harness.test_native_reasoning_training import ReasoningTraining
from evals.harness.test_native_control_stage import ControlStage
from evals.harness.test_context_native_stream_training import NativeStreamTraining


class ReasoningStage(unittest.TestCase):
    def arguments(self):
        values = {n: Path('/' + n) for n in OPTIONS}
        values.update({('control40_'+n):Path('/'+n) for n in
            ('cache','engine','numerical','gradient','parent_review','development_evidence')})
        values.update(native_audio49_audit=Path('/audit'),native_audio49_complete_batch=True,
            native_audio49_stream_auxiliary=True,native_audio49_parent_adam=True,
            actual_fit_updates=8,resume_training=None,native_audio49_step_scale=.1,
            warm_start_proof=Path('/context-training-proof-7510331'),
            native_audio49_preflight_only=True,native_audio49_stream_gradient_readiness=True)
        return values

    def test_explicit_complete_configuration_retains_every_old_stage(self):
        self.assertFalse(validate(SimpleNamespace()))
        values = self.arguments()
        self.assertTrue(validate(SimpleNamespace(**values)))
        for field, value in [('reasoning32_engine',None),('control40_cache',None),
            ('actual_fit_updates',32),('native_audio49_complete_batch',False),
            ('native_audio49_stream_auxiliary',False),('native_audio49_parent_adam',False),
            ('warm_start_proof',Path('/rejected-control40-v11'))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(SimpleNamespace(**{**values,field:value}))

    def fixture(self):
        policy, steps = ReasoningTraining().fixture()
        control_policy, control = ControlStage().fixture()
        stream = NativeStreamTraining(); native_policy = stream.declared()
        native_steps = stream.steps(native_policy)
        for i, step in enumerate(steps):
            step.update(control['steps'][i])
            step.update(native_steps[i])
        for i, step in enumerate(control['steps']):
            step.update(native_steps[i])
        control.update(original154_gradient_source_sha256='base',native49_readiness_sha256='native')
        ready = dict(schema='dl-native-reasoning32-combined-readiness-v1',passed=True,
            policy=policy,control40_policy=control_policy,native49_policy=native_policy,
            all_parameters_unchanged=True,optimizer_updates=0,checkpoint_written=False,
            training_approved=False,original154_components=154,source_commit='committed',
            base154_gradient_sha256=control['base154_gradient_sha256'],steps=steps,
            fixed_parent154_gradient_reused_exactly=True,peak_allocated_bytes=1,
            control40_combined_readiness_sha256='control',
            original154_gradient_source_sha256='base',native49_readiness_sha256='native')
        return policy, ready, control

    def test_any_missing_old_or_new_teacher_prevents_combined_approval(self):
        policy, ready, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ready.json'
            path.write_text(json.dumps(ready))
            self.assertTrue(check_readiness(path,policy,'committed')['passed'])
            for mode in ('original154','native49','control40','reasoning32','source','gradient','optimized'):
                bad=deepcopy(ready)
                if mode=='original154':bad['original154_components']=153
                if mode=='native49':bad['steps'][0]['native_stream_teacher_microbatch'].pop()
                if mode=='control40':bad['steps'][0]['control40_microbatch'].pop()
                if mode=='reasoning32':bad['steps'][0]['reasoning32_microbatch'].pop()
                if mode=='source':bad['source_commit']='other'
                if mode=='gradient':bad['steps'][0]['combined_gradient_norm']=float('nan')
                if mode=='optimized':bad['optimizer_updates']=1
                path.write_text(json.dumps(bad))
                with self.subTest(mode=mode),self.assertRaises(ValueError):check_readiness(path,policy,'committed')

    def test_nested_readiness_must_use_same_retained_parameter_point_and_partitions(self):
        _, ready, control = self.fixture()
        check_bindings(ready,ready['control40_policy'],ready['native49_policy'],control,'control')
        for mode in ('gradient','partition','source','digest'):
            bad=deepcopy(ready)
            if mode=='gradient':bad['base154_gradient_sha256']['0']='different'
            if mode=='partition':bad['steps'][0]['control40_microbatch'][0]['text_ce']=100.
            if mode=='source':bad['source_commit']='other'
            if mode=='digest':bad['control40_combined_readiness_sha256']='different'
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                check_bindings(bad,ready['control40_policy'],ready['native49_policy'],control,'control')

    def test_legacy_checkpoint_entry_still_requires_legacy_schemas(self):
        from evals import native_control_stage as control
        with patch('evals.context_native_audio49_training.verify_retained_checkpoint',return_value=({},{})) as retained, \
             patch.object(control,'verify_control_layer',return_value=({},{})):
            control.verify_checkpoint(Path('/proof'))
        retained.assert_called_once_with(Path('/proof'),control.SCHEMA,control.RELOAD_SCHEMA)

    def test_new_loader_route_requires_new_strict_verifier(self):
        from evals.context_rollout_conditioning import verify_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            proof=Path(directory)
            (proof/'report.json').write_text(json.dumps({'schema':SCHEMA}))
            (proof/'reload-report.json').write_text('{}')
            with patch('evals.native_reasoning_stage.verify_checkpoint',side_effect=ValueError('No complete32 restore')) as strict:
                with self.assertRaisesRegex(ValueError,'complete32'):verify_checkpoint(proof)
                strict.assert_called_once_with(proof)

    def test_native_development_preserves_baseline_and_requires_full_restore(self):
        from portable.context.native_development import proofs,restored_reasoning32
        selected=proofs(reasoning_candidate=Path('/new32'))
        self.assertEqual(selected['baseline520'],proofs()['baseline520'])
        self.assertEqual(list(selected),['baseline520','candidate-reasoning32'])
        with self.assertRaises(ValueError):proofs(Path('/control'),Path('/reasoning'))
        report=dict(schema=SCHEMA,updates_this_run=8,adam_parameter_steps_after=[528],training_pid=100)
        reload=dict(pid=101,parent_adam_parameter_steps_restored=[528],optimizer_and_rng_independently_compared_exact=True)
        with tempfile.TemporaryDirectory() as directory:
            proof=Path(directory);(proof/'report.json').write_text(json.dumps(report))
            (proof/'reload-report.json').write_text(json.dumps(reload))
            with patch('evals.native_reasoning_stage.verify_checkpoint',return_value=({},report)) as strict:
                self.assertTrue(restored_reasoning32(proof)['preexisting_fresh_full_cuda_restore_verified'])
                strict.assert_called_once_with(proof)
                reload['pid']=100;(proof/'reload-report.json').write_text(json.dumps(reload))
                with self.assertRaises(ValueError):restored_reasoning32(proof)
            with patch('evals.native_reasoning_stage.verify_checkpoint',side_effect=ValueError('No complete32 restore')):
                with self.assertRaisesRegex(ValueError,'complete32'):restored_reasoning32(proof)


if __name__=='__main__':unittest.main()
