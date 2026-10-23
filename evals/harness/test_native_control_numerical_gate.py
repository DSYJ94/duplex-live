import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_control_render import sha256
from evals.native_control_gradient import numerical_gate
from evals.native_control_numerical_precise import ORIGINAL_FAILURE_SHA


class Complete40NumericalGate(unittest.TestCase):
    def fixture(self,folder,engine):
        (engine/'engine-inputs.json').write_text(json.dumps(dict(cases=[dict(id=str(i)) for i in range(40)])))
        original=dict(code_sha256=sha256(Path('evals/native_control_numerical.py')),
                      source_commit='7769e7cc06f80e609db3f8ac2611977cd0d7cb66',execution=dict(matmul_tf32=False))
        r=dict(schema='dl-native-control40-numerical-fp64-attention-v2',complete40=True,passed=True,
               all_model_parameters_unchanged=True,threshold=.001,causal_prefix_error_required=0.,
               optimizer_updates=0,new_checkpoint_written=False,training_approved=False,behavior_qualified=False,
               peak_allocated_bytes=60_000_000_000,parent=dict(parent='cpu-fixture'),
               engine_report_sha256=sha256(engine/'engine-inputs.json'),
               code_sha256=sha256(Path('evals/native_control_numerical_precise.py')),
               reference_code_sha256=sha256(Path('evals/native_control_cached_reference.py')),
               source_commit=original['source_commit'],execution=original['execution'],
               independent_reference_profile='fp32-state-fp64-linear-attention-fp32-output-v2',
               original_failure_sha256=ORIGINAL_FAILURE_SHA,original_numerical_code_sha256=original['code_sha256'],
               attention_reference_code_sha256=sha256(Path('evals/native_control_attention_reference.py')),
               input_inventory_rechecked_and_execution_source_clean=True,
               reference=dict(arithmetic='fp32-weights-activations-fp64-linear-fp32-output',allocation_only=True,
                              training_approved=False,calls=100,cast_cache_hits=10,
                              maximum_cached_bytes=32*1024**3,peak_cached_bytes=10*1024**3),
               attention_reference=dict(calls=100,accumulation_dtype='float64',input_output_dtype='float32',
                                        dropout=0.,production_or_training_execution_changed=False),
               immutable_input_hash_reuse=dict(all_cached_identities_rechecked=True,
                    validation_results_or_model_outputs_cached=False,unique_hashed_files=10),
               cases=[dict(id=str(i),passed=True,causal_prefix_maximum_error=0.,
                           native_training_text_alignment_max_error=0.,native_depth_parity=dict(passed=True),
                           comparisons=dict(hidden=dict(maximum_absolute_error=0.),text_logits=dict(maximum_absolute_error=0.)))
                      for i in range(40)])
        return r,original

    def test_new_profile_requires_all40_unchanged_gates_and_independent_reference_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);engine=folder/'engine';engine.mkdir();r,original=self.fixture(folder,engine)
            # GPU calculations are not mocked into an approval: these are pure
            # report-validator fixtures, never used by an actual workload.
            with patch('evals.native_control_numerical_precise.original_failure',return_value=original):
                (folder/'numerical.json').write_text(json.dumps(r))
                self.assertTrue(numerical_gate(folder,engine,r['parent'])['passed'])
                for mode in ('subset','threshold','nan','causal','parameters','runtime','attention','cached_outputs',
                             'unverified_inputs','reference_code','original_failure','numerical_code','gradient_claim'):
                    changed=copy.deepcopy(r)
                    if mode=='subset':changed['cases'].pop()
                    elif mode=='threshold':changed['threshold']=.01
                    elif mode=='nan':changed['cases'][0]['comparisons']['text_logits']['maximum_absolute_error']=float('nan')
                    elif mode=='causal':changed['cases'][0]['causal_prefix_maximum_error']=.001
                    elif mode=='parameters':changed['all_model_parameters_unchanged']=False
                    elif mode=='runtime':changed['execution']['matmul_tf32']=True
                    elif mode=='attention':changed['attention_reference']['production_or_training_execution_changed']=True
                    elif mode=='cached_outputs':changed['immutable_input_hash_reuse']['validation_results_or_model_outputs_cached']=True
                    elif mode=='unverified_inputs':changed['input_inventory_rechecked_and_execution_source_clean']=False
                    elif mode=='reference_code':changed['reference_code_sha256']='0'*64
                    elif mode=='original_failure':changed['original_failure_sha256']='0'*64
                    elif mode=='numerical_code':changed['code_sha256']='0'*64
                    elif mode=='gradient_claim':changed['training_approved']=True
                    (folder/'numerical.json').write_text(json.dumps(changed))
                    with self.subTest(mode=mode),self.assertRaises(ValueError):numerical_gate(folder,engine,r['parent'])


if __name__=='__main__':unittest.main()
