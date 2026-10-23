import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals.native_control_stage import validate,check_readiness
from evals.harness.test_native_control_training import ControlTraining
from evals.native_control_training import schedule
from evals.native_control_gradient import reference_dependencies,function_tree


class ControlStage(unittest.TestCase):
    def test_validated_original_gradient_export_survives_legacy_clear(self):
        import torch
        from evals.context_native_complete_batch import preflight
        from evals.harness.test_context_native_complete_batch import CompleteBalancedBatch
        model=torch.nn.Linear(2,1,bias=False);learned=dict(model.named_parameters());groups=CompleteBalancedBatch().groups()
        def backward(*_):
            model.weight.square().sum().backward()
            return groups
        saved={}
        with tempfile.TemporaryDirectory() as directory, \
             patch('evals.context_native_complete_batch.backward',side_effect=backward), \
             patch('torch.cuda.max_memory_allocated',return_value=0), \
             patch.dict('os.environ',{'DL_TASK_COMMIT':'cpu-capture-test'}):
            expected=2*model.weight.detach().clone()
            preflight(model,learned,SimpleNamespace(out=Path(directory)),
                      {'complete_batch_membership':groups},capture_gradients=saved)
            self.assertIsNone(model.weight.grad)
            self.assertTrue(torch.equal(saved['weight'],expected))
            self.assertFalse(saved['weight'].requires_grad)

    def test_all_options_and_retained_recipe_are_mandatory(self):
        self.assertFalse(validate(SimpleNamespace()))
        names=('cache','engine','numerical','gradient','parent_review','development_evidence')
        base={('control40_'+n):Path('/'+n) for n in names}
        base.update(native_audio49_audit=Path('/audit'),native_audio49_complete_batch=True,
            native_audio49_stream_auxiliary=True,native_audio49_parent_adam=True,
            actual_fit_updates=8,resume_training=None,native_audio49_step_scale=.1,
            warm_start_proof=Path('/context-training-proof-7510331'),
            native_audio49_preflight_only=True,native_audio49_stream_gradient_readiness=True)
        self.assertTrue(validate(SimpleNamespace(**base)))
        for name,value in [('control40_engine',None),('native_audio49_complete_batch',False),
                           ('native_audio49_stream_auxiliary',False),('actual_fit_updates',32),
                           ('warm_start_proof',Path('/context-training-proof-7590052')),
                           ('native_audio49_stream_gradient_readiness',False)]:
            changed={**base,name:value}
            with self.subTest(name=name),self.assertRaises(ValueError):validate(SimpleNamespace(**changed))

    def fixture(self):
        source=ControlTraining();policy=dict(teachers=schedule(source.records()))
        steps=source.steps(policy)
        counts={r['id']:dict(text=400,audio=[400]+[399]*7) for r in policy['teachers']}
        policy.update(target_counts=counts,text_loss_targets_per8=40*400,assistant_audio_loss_targets_per8=40*(8*400-7))
        for s in steps:s.update(combined_gradient_norm=1.,frozen_parameter_gradients=0)
        return policy,dict(schema='dl-native-control40-combined-readiness-v1',passed=True,policy=policy,
            all_parameters_unchanged=True,optimizer_updates=0,checkpoint_written=False,source_commit='committed',
            peak_allocated_bytes=1,original154_components=154,base154_gradient_sha256={str(i):'sha' for i in range(26)},
            fixed_parent154_gradient_reused_exactly=True,steps=steps)

    def test_missing_old_terms_subset_or_changed_control_target_cannot_start_optimizer(self):
        policy,original=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ready.json';path.write_text(json.dumps(original))
            self.assertTrue(check_readiness(path,policy,'committed')['passed'])
            for mode in ('old_terms','subset','targets','nan','optimized','source'):
                r=copy.deepcopy(original)
                if mode=='old_terms':r['original154_components']=153
                if mode=='subset':r['steps'][3]['control40_microbatch'].pop()
                if mode=='targets':r['steps'][3]['control40_microbatch'][0]['text_targets']=399
                if mode=='nan':r['steps'][0]['combined_gradient_norm']=float('nan')
                if mode=='optimized':r['optimizer_updates']=1
                if mode=='source':r['source_commit']='other'
                path.write_text(json.dumps(r))
                with self.subTest(mode=mode),self.assertRaises(ValueError):check_readiness(path,policy,'committed')

    def test_original_numerical_math_is_identical_despite_metadata_schema_extension(self):
        self.assertTrue(reference_dependencies('7769e7cc06f80e609db3f8ac2611977cd0d7cb66'))
        self.assertNotEqual(function_tree('def forward(x): return x+1','forward'),
                            function_tree('def forward(x): return x+2','forward'))


if __name__=='__main__':unittest.main()
