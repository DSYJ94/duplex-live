"""Synthetic complete recovery artifacts; no model, Torch or GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.native_saved_stage_proof import (
    validate_records,SCHEMA,RESULT_SCHEMA,JOB_SCHEMA,PROFILE,METADATA,
    SOURCE,INVENTORY,OBSERVATION,PILOT,PINNED,canonical,record_terminal)


class SavedStageProofTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);scenarios=['scenario-%02d'%i for i in range(28)]
        (root/'data.json').write_text(json.dumps(dict(components=[dict(scenario=s) for s in scenarios])))
        native=lambda:dict(native_text_calls_sha256='b'*64,assistant_audio_logits_sha256='c'*64,
            assistant_audio_targets=[2]*8,text_targets=3)
        audio=dict(teacher_ids=['a'+str(i) for i in range(49)],
            native_stream_auxiliary=dict(teachers=[dict(id='n'+str(i)) for i in range(49)]))
        policies={name:dict(teachers=[dict(id=prefix+str(i)) for i in range(count)],
            target_counts={prefix+str(i):dict(text=3,audio=[2]*8) for i in range(count)})
            for name,prefix,count in (('control40','c',40),('reasoning32','r',32))}
        meta=dict(recipe=dict(assistant_audio_auxiliary=audio,**policies))
        self.inputs=dict(training=root,steps=[dict(step=i) for i in range(1,9)],
            prefix=dict(metadata=meta,parameter_names=[m+'.lora_'+p+'.weight'
                for m in ['module'+str(i) for i in range(13)] for p in ('A','B')]),
            partials=dict(native49={'n'+str(i):'a'*64 for i in range(49)},
                control40={'c'+str(i):native() for i in range(31)},reasoning32={}))
        self.context=dict(derived_ast_sha256='fixture-construction',optimizer_branch_included=False)
        self.producer_files={'producer':'fixture-hash'};self.input_code={'original':'fixture-hash'}
        metrics=[dict(scenario=s,nll=1.,top1=.5,first_supervised_token_probability=.25) for s in scenarios]
        fingerprints=dict(original28={str(i)+':'+s:'a'*64 for i,s in enumerate(scenarios)},
            audio49={i:'a'*64 for i in audio['teacher_ids']},
            native49={'n'+str(i):'a'*64 for i in range(49)},
            control40={'c'+str(i):native() for i in range(40)},
            reasoning32={'r'+str(i):native() for i in range(32)})
        first=dict(schema=RESULT_SCHEMA,phase='finalization',producer_commit='d'*40,
            model_source_commit=SOURCE,pid=100,process_identity=dict(pid=100,birth=1,host='fixture',boot_id='fixture-boot'),
            at_utc='2026-10-24T18:00:00+00:00',context=self.context,
            reference_finalization_sha256=None,full_independent_restore_verified=False,
            original_inventory_sha256=INVENTORY,original_terminal_observation_sha256=OBSERVATION,
            training_state_sha256=PINNED[PILOT+'/training/training-state.pt'],
            adapter_sha256=PINNED[PILOT+'/training/adapters.safetensors'],
            original_fit_actual_exit=-15,original_fit_passed=False,original_stage_updates=8,
            optimizer_updates_during_recovery=0,adam_parameter_steps=[528],optimizer_and_rng_exact=True,
            original_partial_fingerprints_exact=True,quality_qualified=False,native_behavior_evaluated=False,
            model_adopted=False,passed=True,evidence=dict(state_metadata=copy.deepcopy(meta),metadata_sha256=METADATA,
                original_steps=copy.deepcopy(self.inputs['steps']),all_fingerprints=fingerprints,
                initial_stage_metrics_recomputed=metrics,final_metrics_recomputed=copy.deepcopy(metrics),
                adapter_modules=['module'+str(i) for i in range(13)]))
        second=copy.deepcopy(first);second.update(phase='independent_restore',pid=101,
            process_identity=dict(pid=101,birth=2,host='fixture',boot_id='fixture-boot'),full_independent_restore_verified=True)
        self.records={'finalization-record.json':first,'independent-record.json':second}
        stamp=1792864800.
        for prefix,result in (('finalization',first),('independent',second)):
            self.records[prefix+'-runtime.json']=dict(schema='dl-native-saved-stage-runtime-inputs-v1',passed=True,
                source_commit=SOURCE,producer_commit='d'*40,context=self.context,
                producer_implementation_sha256=self.producer_files,code_sha256=self.input_code,profile=PROFILE,
                optimizer_updates_during_recovery=0,original_fit_relabelled=False,model_adopted=False,
                cache=dict(mode=PROFILE,unique_hashed_files=100,all_cached_identities_rechecked=True,
                    validation_results_or_model_outputs_cached=False),
                initial_guard=dict(execution_authorized=True,live_provider_deadline_verified=True,
                    live_external_cost_guard_verified=True,single_gpu_across_providers_verified=True,
                    provider='gcp',resource_id='fixture-vm',authorization_sha256='e'*64,
                    checked_unix=stamp-100,deadline_unix=stamp+1000,
                    remaining_budget_usd=100.,planned_maximum_cost_usd=90.))
            self.records[prefix+'-job.json']=dict(schema=JOB_SCHEMA,phase='complete',kind=result['phase'],
                actual_exit=0,observed_terminal=True,child_identity=result['process_identity'],
                producer_commit='d'*40,model_source_commit=SOURCE,started_unix=stamp-50,ended_unix=stamp+50)
        self.seal()

    def seal(self):
        self.records['independent-record.json']['reference_finalization_sha256']=canonical(self.records['finalization-record.json'])
        for prefix in ('finalization','independent'):
            result=self.records[prefix+'-record.json'];runtime=self.records[prefix+'-runtime.json'];job=self.records[prefix+'-job.json']
            runtime['result_sha256']=canonical(result);job['result_canonical_sha256']=canonical(result)
            job['runtime_canonical_sha256']=canonical(runtime)

    def check(self):return validate_records(self.records,self.inputs,self.producer_files,self.input_code,self.context)

    def both(self,key,value):
        for prefix in ('finalization','independent'):self.records[prefix+'-record.json'][key]=value
        self.seal()

    def test_complete_records_preserve_failed_original_and_zero_updates(self):
        first,second=self.check()
        self.assertFalse(first['original_fit_passed']);self.assertTrue(second['full_independent_restore_verified'])
        self.assertEqual(second['optimizer_updates_during_recovery'],0)

    def test_passing_partial_even_if_both_processes_agree_is_rejected(self):
        for name in ('original28','audio49','native49','control40','reasoning32'):
            saved=copy.deepcopy(self.records)
            for prefix in ('finalization','independent'):
                rows=self.records[prefix+'-record.json']['evidence']['all_fingerprints'][name]
                rows.pop(next(reversed(rows)))
            self.seal()
            with self.subTest(name=name),self.assertRaises(ValueError):self.check()
            self.records=saved

    def test_failed_original_cannot_be_relabelled_or_training_repeated(self):
        for key,value in [('original_fit_passed',True),('original_fit_actual_exit',0),
                ('optimizer_updates_during_recovery',8),('original_stage_updates',7),('adam_parameter_steps',[520]),
                ('model_adopted',True),('quality_qualified',True),('optimizer_and_rng_exact',False)]:
            saved=copy.deepcopy(self.records);self.both(key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.records=saved

    def test_source_or_named_adapter_scope_cannot_change(self):
        for key,value in [('model_source_commit','f'*40),('training_state_sha256','f'*64),
                ('adapter_sha256','f'*64),('original_inventory_sha256','f'*64)]:
            saved=copy.deepcopy(self.records);self.both(key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.records=saved
        for prefix in ('finalization','independent'):
            self.records[prefix+'-record.json']['evidence']['adapter_modules'][-1]='other'
        self.seal()
        with self.assertRaises(ValueError):self.check()

    def test_same_process_reused_pid_or_cross_host_is_rejected(self):
        second=self.records['independent-record.json']
        for key,value in [('pid',100),('host','other'),('boot_id','other')]:
            saved=copy.deepcopy(self.records)
            second=self.records['independent-record.json'];second['process_identity'][key]=value
            if key=='pid':second['pid']=value
            self.seal()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.records=saved

    def test_missing_actual_exit_zero_or_liveness_proof_rejected(self):
        for key,value in [('actual_exit',-15),('actual_exit',False),('observed_terminal',False),('phase','running')]:
            saved=copy.deepcopy(self.records);self.records['independent-job.json'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.records=saved

    def test_failed_or_changed_runtime_and_hash_cache_rejected(self):
        for key,value in [('passed',False),('source_commit','f'*40),('producer_implementation_sha256',{}),
                ('code_sha256',{}),('original_fit_relabelled',True)]:
            saved=copy.deepcopy(self.records);self.records['independent-runtime.json'][key]=value;self.seal()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.records=saved
        self.records['independent-runtime.json']['cache']['validation_results_or_model_outputs_cached']=True;self.seal()
        with self.assertRaises(ValueError):self.check()

    def test_both_matching_altered_losses_or_source_counts_are_not_enough(self):
        for mutate in (lambda e:e['final_metrics_recomputed'].pop(),
                lambda e:e['all_fingerprints']['reasoning32']['r31'].update(text_targets=4),
                lambda e:e['all_fingerprints']['native49'].update(n0='d'*64),
                lambda e:e['state_metadata']['recipe'].update(retention_kl_weight=0.),
                lambda e:e['original_steps'].pop()):
            saved=copy.deepcopy(self.records)
            for prefix in ('finalization','independent'):mutate(self.records[prefix+'-record.json']['evidence'])
            self.seal()
            with self.assertRaises(ValueError):self.check()
            self.records=saved

    def test_unbounded_unapproved_or_different_resource_rejected(self):
        for key,value in [('execution_authorized',False),('live_external_cost_guard_verified',False),
                ('resource_id','different-vm'),('planned_maximum_cost_usd',101.),('planned_maximum_cost_usd',True),
                ('authorization_sha256','g'*64),('deadline_unix',1792909000.)]:
            saved=copy.deepcopy(self.records);self.records['independent-runtime.json']['initial_guard'][key]=value;self.seal()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.records=saved

    def test_legacy_checkpoint_router_requires_strict_new_proof(self):
        from evals.context_rollout_conditioning import verify_checkpoint
        proof=Path(self.temp.name)
        (proof/'report.json').write_text(json.dumps(dict(schema=SCHEMA)))
        (proof/'reload-report.json').write_text('{}')
        with patch('evals.native_saved_stage_proof.verify_checkpoint',side_effect=ValueError('restore incomplete')) as strict:
            with self.assertRaisesRegex(ValueError,'restore incomplete'):verify_checkpoint(proof)
            strict.assert_called_once_with(proof)

    def test_terminal_recorder_observes_real_child_exit_and_identity(self):
        from types import SimpleNamespace
        folder=Path(self.temp.name);result=self.records['finalization-record.json']
        (folder/'saved-stage-finalization.json').write_text(json.dumps(result))
        (folder/'recovery-runtime-inputs.json').write_text(json.dumps(self.records['finalization-runtime.json']))
        child=SimpleNamespace(pid=100,returncode=0,poll=lambda:0)
        with patch('portable.context.native_preflight_takeover.process',return_value=None):
            receipt=record_terminal(child,folder,'d'*40,1792864750.,folder/'job.json')
            self.assertEqual(receipt['actual_exit'],0);self.assertTrue(receipt['observed_terminal'])
        for handle in (dict(birth=2,state='Z'),dict(birth=1,state='R')):
            with patch('portable.context.native_preflight_takeover.process',return_value=handle),self.assertRaises(ValueError):
                record_terminal(child,folder,'d'*40,1792864750.,folder/'refused.json')
        child.poll=lambda:-15
        with self.assertRaises(ValueError):record_terminal(child,folder,'d'*40,1792864750.,folder/'refused.json')
        self.assertFalse((folder/'refused.json').exists())

    def test_reasoning_development_dispatch_uses_strict_recovery_only(self):
        from portable.context.native_development import restored_reasoning32
        proof=Path(self.temp.name)
        (proof/'report.json').write_text(json.dumps(dict(schema=SCHEMA)))
        (proof/'reload-report.json').write_text(json.dumps(dict(pid=101,finalization_pid=100)))
        identity=dict(report_sha256='report',reload_report_sha256='reload')
        with patch('evals.native_saved_stage_proof.verify_checkpoint',return_value=(identity,{})) as strict:
            value=restored_reasoning32(proof)
            strict.assert_called_once_with(proof)
            self.assertFalse(value['original_fit_passed']);self.assertFalse(value['behavior_or_adoption_approved'])
            self.assertEqual(value['recovery_optimizer_updates'],0)
        with patch('evals.native_saved_stage_proof.verify_checkpoint',side_effect=ValueError('complete restore required')):
            with self.assertRaises(ValueError):restored_reasoning32(proof)

    def test_recovered_candidate_does_not_bypass_ordinary4_gate(self):
        from evals.native_control40_fresh_review import validate_review,REASONING_SCHEMA
        from evals.native_fresh_inputs_v2 import PLAN_SHA256
        from evals.moshi_model_download import sha256
        proof=Path(self.temp.name).resolve();plan=Path(__file__).resolve().parents[1]/'fixtures/native-v2-development-20261019.json'
        identity=dict(checkpoint_sha256='adapter',training_state_sha256='state',reload_report_sha256='reload',
            recovered_saved_stage=True,recovery_optimizer_updates=0)
        parent=dict(schema=SCHEMA,original_stage_updates=8,updates_this_run=0,original_fit_actual_exit=-15,
            original_fit_passed=False,adam_parameter_steps_after=[528],additional_native_v2=dict(complete24_per8=True))
        review=dict(schema=REASONING_SCHEMA,candidate_proof=str(proof),candidate_checkpoint_sha256='adapter',
            candidate_training_state_sha256='state',independent_reload_report_sha256='reload',
            evaluation_plan_sha256=sha256(plan),fresh_plan_sha256=PLAN_SHA256,all10_native_cases_reviewed=True,
            all4_ordinary_core_correct=False,fresh_evaluation_authorized=True,checkpoint_adopted=False,
            native_asr_placement='shared-gpu-evaluation',cases=[dict(index=i) for i in range(10)])
        with self.assertRaisesRegex(ValueError,'ordinary4'):
            validate_review(review,proof,parent,identity)
        identity['recovered_saved_stage']=False
        with self.assertRaisesRegex(ValueError,'zero-update recovery'):
            validate_review(review,proof,parent,identity)


if __name__=='__main__':unittest.main()
