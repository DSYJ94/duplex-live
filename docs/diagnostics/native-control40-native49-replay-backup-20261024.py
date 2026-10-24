import json,hashlib,datetime
from pathlib import Path
watch=json.loads(Path('runs/local-migration-20261023/native-control40-continuous-copyback-v17/controller.json').read_text());closed=watch['snapshots'][-1];s=Path(closed['path']);digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert digest(s/'inventory.json')==closed['inventory_sha256'];inv=json.loads((s/'inventory.json').read_text());rows={x['path']:x for x in inv['files']}
p=s/'native-control40-precise-pilot-v11-environment-retry';log=p/'independent_restore.log';report=p/'training/report.json'
for f in (log,report):
 row=rows[str(f.relative_to(s))];assert f.stat().st_size==row['bytes'] and digest(f)==row['sha256']
assert digest(report)=='500781444712ebb6368d534406ee25f663df4b511c6f920fae9e985db9aa5a18'
r=json.loads(report.read_text());ref=r['expected_native_stream49_logits'];actual={};control=0
for line in log.read_text().splitlines():
 try:x=json.loads(line)
 except ValueError:continue
 if not isinstance(x,dict):continue
 if 'native_stream_restore_fingerprint'in x:
  key=x['native_stream_restore_fingerprint'];assert key not in actual and ref[key]==x['sha256'];actual[key]=x['sha256']
 control+=int('control40_restore_fingerprint'in x)
assert len(ref)==len(actual)==49 and set(actual)==set(ref)
out=dict(schema='dl-native-control40-independent-native49-replay-backup-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=r['repo_commit'],training_pid=r['training_pid'],restore_pid=115807,restore_birth=26750060,all49_native_stream_ids_complete=True,all49_reference_and_fresh_process_hashes_exact=True,all49_fresh_process_fingerprints_backed_up=True,control_replays_in_this_snapshot=control,snapshot=str(s),snapshot_inventory_sha256=closed['inventory_sha256'],restore_log_sha256=digest(log),training_report_sha256=digest(report),checkpoint_sha256=r['checkpoint_sha256'],training_state_sha256=r['training_state_sha256'],fresh_native49_fingerprints=actual,full_independent_restore_passed=False,behavior_qualified=False,model_adopted=False,scope='Full49 native-stream replay digest comparison and backup only. Complete control40/optimizer/RNG/all other restoration checks and final reload report/proof remain required; no subset acceptance.')
p=Path('runs/local-migration-20261023/native-control40-v11-restored49-backup-v1.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='fresh_native49_fingerprints'}))
