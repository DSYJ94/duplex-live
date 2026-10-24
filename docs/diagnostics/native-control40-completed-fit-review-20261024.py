import json,hashlib,datetime,subprocess,sys,math
from pathlib import Path
commit='c9efaaac24d983342f7eded46617e6450bc102ce'
s=Path('runs/local-migration-20261023/native-control40-continuous-copyback-v17/20261024T065903.224466Z')
p=s/'native-control40-precise-pilot-v11-environment-retry/training'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inv=json.loads((s/'inventory.json').read_text());bound={x['path']:x for x in inv['files']}
verified={}
for name in ('report.json','immutable-input-training.json','approved-immutable-input-readiness.json','adapters.safetensors','training-state.pt','audio49-data.json','data.json'):
 f=p/name;entry=bound[str(f.relative_to(s))];digest=sha(f)
 assert f.stat().st_size==entry['bytes'] and digest==entry['sha256']
 verified[name]=dict(bytes=f.stat().st_size,sha256=digest)
r=json.loads((p/'report.json').read_text());a=r['assistant_audio_auxiliary'];c=r['control40']
assert r['schema']=='dl-native-control40-retained154-stream49-proof-v1' and r['passed'] is True
assert r['repo_commit']==commit and r['optimizer_steps']==r['updates_this_run']==8 and r['initial_optimizer_steps']==0
assert r['training_pid']==101677 and r['resumed_from_training_state'] is None
assert r['frozen_weights_unchanged'] and r['base_audio_parameters_frozen'] and r['adapter_construction_rng_preserved']
assert r['all_actual_temporal_audits_passed'] and r['all_component_audits_passed']
assert r['trainable_parameters']==1894400 and r['rank']==8 and r['last_layers']==4 and r['scaling']==2
assert r['adam_parameter_steps_after']==[528] and a['parent_adam_parameter_steps']==520 and a['final_adam_parameter_steps']==528
assert r['checkpoint_sha256']==verified['adapters.safetensors']['sha256']=='0cb85b4a3b712ba5687e3e23dba6541891cbad954bde4a155ae9cb913517bcff'
assert r['training_state_sha256']==verified['training-state.pt']['sha256']=='86dde08eb1591d0bc04a36c47be0dc7ee1b90019d7ae520f1d049cc37d8dc278'
assert all(r[k]==0 for k in ('past_loss_targets','original_actual_history_audio_loss_targets','user_audio_loss_targets','prefix_audio_loss_targets','supplied_span_audio_loss_targets'))
assert all(r[k] is False for k in ('quality_qualified','free_running_result_use_proven','audio_behavior_preservation_proven'))
for name in ('evals/context_native_complete_batch.py','evals/context_native_stream_training.py','evals/native_control_training.py','evals/native_control_stage.py','portable/context/native_training_cached.py'):
 assert Path(name).read_bytes()==subprocess.check_output(['git','show',commit+':'+name])
from evals.context_native_complete_batch import validate_exposure
from evals.context_native_stream_training import exposure as native_exposure
from evals.native_control_training import exposure as control_exposure
from evals.native_control_stage import exact_target_counts
from portable.context.native_training_cached import verify_receipt,PROFILE
validate_exposure(r['steps'],a['teacher_ids'],a['source_unique49_acoustic_targets'],a['complete_batch_membership'])
native_exposure(r['steps'],a['native_stream_auxiliary']);control_exposure(r['steps'],c);exact_target_counts(r['steps'],c)
for name,kind in (('immutable-input-training.json','training'),('approved-immutable-input-readiness.json','readiness')):verify_receipt(p/name,kind,commit)
assert r['immutable_input_hashes']==dict(profile=PROFILE,training_receipt_sha256=verified['immutable-input-training.json']['sha256'],readiness_receipt_sha256=verified['approved-immutable-input-readiness.json']['sha256'])
assert len(r['expected_logits'])==28
assert set(r['expected_audio49_logits'])==set(r['expected_native_stream49_logits'])==set(a['teacher_ids'])
assert len(r['expected_audio49_logits'])==49
assert set(r['expected_native_control40_logits'])=={x['id']for x in c['teachers']} and len(c['teachers'])==40
for key in ('expected_logits','expected_audio49_logits','expected_native_stream49_logits'):
 assert all(len(x)==64 and all(v in '0123456789abcdef'for v in x)for x in r[key].values())
for key in ('rank','scaling','last_layers','lr','weight_decay','gradient_clip','retention_kl_weight','retention_hidden_mse_weight','actual_component_count'):
 assert r['combined_recipe'][key]=={'rank':8,'scaling':2.,'last_layers':4,'lr':2e-7,'weight_decay':0.,'gradient_clip':1.,'retention_kl_weight':1.,'retention_hidden_mse_weight':10.,'actual_component_count':28}[key]
initial=r['initial_stage_metrics'];final=r['final_metrics'];assert len(initial)==len(final)==28
assert [x['scenario']for x in initial]==[x['scenario']for x in final]
metrics={}
for name in ('nll','top1'):
 b=sum(x[name]for x in initial)/28;f=sum(x[name]for x in final)/28
 assert math.isfinite(b) and math.isfinite(f)
 metrics[name]=dict(parent520_unweighted_component_mean=b,after8_unweighted_component_mean=f,delta=f-b)
assert 'torch'not in sys.modules
ob=json.loads(Path('runs/local-migration-20261023/native-control40-v11-progress-observation-v1.json').read_text())
assert next(x for x in ob['stages']if x['name']=='bounded8_training')['exit_code']==0
result=dict(schema='dl-native-control40-completed-fit-review-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=commit,report_passed=True,fit_actual_exit_code=0,training_pid=101677,training_birth=25975025,reference_fingerprints=dict(actual28=28,assistant_audio49=49,native_stream49=49,control40=40),unchanged_C9_exposure_and_immutable_receipt_validators_passed=True,verified_local_backup=str(s),backup_inventory_sha256=sha(s/'inventory.json'),backup_files=len(inv['files']),verified_final_files=verified,optimizer_updates=8,final_Adam_clock=528,trainable_parameters=1894400,all_original_objective_weights_and_exposures_preserved=True,actual28_teacher_forced_metrics=metrics,metrics_scope='All28 original components, unweighted component means on fixed teacher-forced inputs before and after all8 updates. Not the complete weighted154 objective, free-running behavior, or convergence.',Torch_imported=False,model_loaded=False,GPU_work_executed=False,independent_native_restore_passed=False,behavior_qualified=False,model_adopted=False,live_restore_at_review={k:ob[k]for k in ('at_utc','phase','child','fingerprint_progress')})
out=Path('runs/local-migration-20261023/native-control40-v11-completed-fit-review-v1.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
