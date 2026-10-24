import json,hashlib,datetime,sys,subprocess
from pathlib import Path
from portable.context.native_training_cached import verify_receipt,PROFILE
root=Path('runs/local-migration-20261023');s=root/'native-control40-continuous-copyback-v17/20261024T075140.913692Z';p=s/'native-control40-precise-pilot-v11-environment-retry';proof=p/'training';commit='c9efaaac24d983342f7eded46617e6450bc102ce'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(s/'inventory.json')=='17f3c294d9e8801386384af255c11a95637b741231a5b07e896ca35bb0cfd1f4'
inv=json.loads((s/'inventory.json').read_text());by={x['path']:x for x in inv['files']};verified={}
for f in [p/'controller.json',p/'strict_proof_verification.log',p/'independent_restore.log']+[proof/name for name in('report.json','reload-report.json','immutable-input-training.json','immutable-input-reload.json','approved-immutable-input-readiness.json','adapters.safetensors','training-state.pt')]:
 key=str(f.relative_to(s));e=by[key];digest=sha(f);assert f.stat().st_size==e['bytes'] and digest==e['sha256'];verified[str(f.relative_to(p))]=dict(bytes=f.stat().st_size,sha256=digest)
c=json.loads((p/'controller.json').read_text());r=json.loads((proof/'report.json').read_text());v=json.loads((proof/'reload-report.json').read_text())
expected=['whole_workset','complete40_precise_numerical','complete40_independent_BF16_gradient','combined_readiness','bounded8_training','independent_restore','strict_proof_verification']
assert c['source_commit']==r['repo_commit']==commit and c['phase']=='complete_restored_pilot_only' and c['passed'] is True
assert [x['name']for x in c['stages']]==expected and all(x['exit_code']==0 and x['command']==c['planned_commands'][x['name']]for x in c['stages'])
assert c['optimizer_update_limit']==r['updates_this_run']==r['optimizer_steps']==8 and not c['partial_candidate_resumed'] and c['original_parent_restart']
assert c['model_adopted'] is c['behavior_qualified'] is c['extension_approved'] is False
fit,restore,strict=c['stages'][4:];assert fit['pid']==r['training_pid']==v['training_pid']==101677 and fit['birth']==25975025
assert restore['pid']==v['pid']==115807 and restore['birth']==26750060 and v['pid']!=v['training_pid'] and '--reload'in restore['command']
assert strict['pid']==121918 and strict['birth']==27070269 and 'from evals.native_control_stage import verify_checkpoint'in strict['command'][-1]
assert (p/'strict_proof_verification.log').read_text().strip()=='Full retained154/native49/control40 proof and fresh restore verified'
assert r['schema']=='dl-native-control40-retained154-stream49-proof-v1' and v['schema']=='dl-native-control40-retained154-stream49-reload-v1'
for key in ('passed','fresh_process','all_valid_logits_exact','optimizer_rng_state_restored','optimizer_and_rng_independently_compared_exact','all49_auxiliary_logits_exact','all49_native_stream_logits_exact','all40_native_control_logits_exact'):assert v[key] is True
assert v['actual']==r['expected_logits'] and len(v['actual'])==28
assert v['actual_audio49_logits']==r['expected_audio49_logits'] and len(v['actual_audio49_logits'])==49
assert v['actual_native_stream49_logits']==r['expected_native_stream49_logits'] and len(v['actual_native_stream49_logits'])==49
assert v['actual_native_control40_logits']==r['expected_native_control40_logits'] and len(v['actual_native_control40_logits'])==40
assert v['parent_adam_parameter_steps_restored']==r['adam_parameter_steps_after']==[528]
for file,key in (('adapters.safetensors','checkpoint_sha256'),('training-state.pt','training_state_sha256')):assert sha(proof/file)==r[key]==v[key]
assert v['data_sha256']==r['data_sha256']
assert sha(proof/'report.json')=='500781444712ebb6368d534406ee25f663df4b511c6f920fae9e985db9aa5a18'
assert r['immutable_input_hashes']==dict(profile=PROFILE,training_receipt_sha256=sha(proof/'immutable-input-training.json'),readiness_receipt_sha256=sha(proof/'approved-immutable-input-readiness.json'))
assert v['immutable_input_hashes_receipt_sha256']==sha(proof/'immutable-input-reload.json')
for name,kind in (('immutable-input-training.json','training'),('immutable-input-reload.json','reload'),('approved-immutable-input-readiness.json','readiness')):verify_receipt(proof/name,kind,commit)
for name in ('evals/context_actual_train.py','portable/context/native_training_cached.py','evals/native_control_stage.py'):
 assert Path(name).read_bytes()==subprocess.check_output(['git','show',commit+':'+name])
assert 'torch'not in sys.modules
ob=json.loads((root/'native-control40-v11-progress-observation-v1.json').read_text());assert ob['phase']=='complete_restored_pilot_only' and ob['passed'] is True and ob['controller']is None
result=dict(schema='dl-native-control40-complete-fresh-restore-and-proof-review-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=commit,orchestration_commit=c['orchestration_commit'],training_pid=r['training_pid'],fresh_restore_pid=v['pid'],fresh_restore_birth=restore['birth'],strict_proof_pid=strict['pid'],strict_proof_birth=strict['birth'],all7_actual_exit_codes_zero=True,upstream_supervisor_actually_ended=True,completed_updates=8,exact_restored_inventories=dict(original_histories=28,assistant_acoustic_teachers=49,native_stream_teachers=49,control_native_text_and_acoustic_teachers=40),all4_complete_inventory_dictionaries_exact=True,all_named_adapters_Adam_moments_and_state_and_all4_RNG_families_independently_compared_exact=True,Adam_clocks=[528],exact_code_and_immutable_input_receipts_verified=True,source_assertions_reviewed='Pinned C9 context_actual_train.py compares all adapter arrays, full optimizer state_dict, Torch CPU/CUDA, Python and NumPy RNG before replaying every inventory.',full_native_restore_passed=True,strict_proof_passed=True,verified_Mac_snapshot=str(s),snapshot_files=len(inv['files']),inventory_sha256=sha(s/'inventory.json'),verified_final_files=verified,model_or_Torch_loaded_on_Mac=False,additional_GPU_or_optimizer_work_executed=False,behavior_qualified=False,model_adopted=False,training_extension_approved=False,scope='Complete independent native training-state restoration and existing strict proof, not native behavioral qualification, convergence or adoption.')
out=root/'native-control40-v11-full-restore-review-v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
