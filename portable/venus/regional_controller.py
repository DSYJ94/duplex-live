"""One approved public-only regional native diagnostic; no training or API backend."""
import hashlib,json,os,shlex,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from portable.venus.guard import PROJECT,PRIOR_SECONDS,PRIOR_USD

R=Path('runs/venus-20261026/us-v1');R.mkdir(exist_ok=True)
NAME='duplex-venus-us-20261026-v1';ZONE=sys.argv[1] if len(sys.argv)>1 else 'us-central1-a'
IMAGE='common-cu129-ubuntu-2204-nvidia-580-v20261019'
ROOT='/workspace/migration';guard=None;vm=None;a=None;events=[];created=False
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
remote=ROOT+'/duplex-venus-'+source[:7]+'-us-v1'
awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
def run(argv,timeout=60):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-2500:])
 return p.stdout
def cloud(*args):return json.loads(run(['gcloud','compute',*args,'--project='+PROJECT,'--format=json','--quiet']))
def record(event,**kw):
 row=dict(at_utc=datetime.now(timezone.utc).isoformat(),event=event,**kw);events.append(row)
 (R/'events.json').write_text(json.dumps(events,indent=2)+'\n');print(json.dumps(row),flush=True)
def ssh(cmd,timeout=60):return run(a+[cmd],timeout)
def sync(local,dest,timeout=90):return run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=core','--exclude=core.*','-e',shlex.join(a[:-1]),str(local),a[-1]+':'+dest],timeout)
def props(unit):return dict(x.split('=',1) for x in ssh('sudo systemctl show '+unit+' -p ActiveState -p Result -p ExecMainStatus -p MainPID -p InvocationID').splitlines())
def wait_unit(unit,deadline):
 while True:
  p=props(unit)
  if p['ActiveState'] in ('failed','inactive'):return p
  if time.time()>deadline:raise TimeoutError('Bounded unit deadline: '+unit)
  if guard.poll() is not None:raise RuntimeError('Independent cost guard ended')
  time.sleep(5)
def copyback():
 # Full unique runtime output; public weights and reproducible venv excluded.
 ssh("mkdir -p /workspace/migration/venus-receipts; for u in duplex-venus-setup-us-v1 duplex-venus-app-us-v1 duplex-venus-benchmark-us-v1; do sudo journalctl -u $u --no-pager -o cat > /workspace/migration/venus-receipts/$u.log; done")
 code="""import hashlib,json
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
root=Path('/workspace/migration');out=[]
for base in [root/'venus-receipts',root/'venus-runtime-v1'/'diagnostic-v1']:
 if base.exists():
  for p in sorted(base.rglob('*')):
   if p.is_file():out.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
for p in [root/'venus-runtime-v1'/'setup-complete.json',root/'venus-runtime-v1'/'pip-freeze.txt']:
 if p.exists():out.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
print(json.dumps(out))"""
 rows=json.loads(ssh('/usr/bin/python3 -c '+shlex.quote(code),90));(R/'unique-inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
 dest=R/'copyback';dest.mkdir(exist_ok=True)
 paths=R/'copyback-files.txt';paths.write_text(''.join(x['path']+'\n' for x in rows))
 run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=core','--exclude=core.*','--files-from='+str(paths),'-e',shlex.join(a[:-1]),a[-1]+':'+ROOT+'/',str(dest)+'/'],180)
 assert rows and all((dest/x['path']).stat().st_size==x['bytes'] and hashlib.file_digest((dest/x['path']).open('rb'),'sha256').hexdigest()==x['sha256'] for x in rows)
 (R/'copyback-verified.json').write_text(json.dumps(dict(all_hashes_verified=True,files=len(rows),bytes=sum(x['bytes'] for x in rows)),indent=2)+'\n')
 record('unique_output_copyback_hashes_pass',files=len(rows),bytes=sum(x['bytes'] for x in rows))
try:
 auth=Path('docs/NATIVE-REGIONAL-PLACEMENT-AUTHORIZATION-2026-10-26.json').read_bytes();grant=json.loads(auth)
 assert grant['status']=='APPROVED' and ZONE in grant['resources']['supported_catalog_zones']
 assert not (R/'lease.json').exists(),'Never duplicate a prior real allocation'
 price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text())
 assert 0<price['hourly_estimate_usd']<=7.25 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
 allv=cloud('instances','list');assert all(v['status'] in ('TERMINATED','STOPPED') for v in allv)
 assert not any(v['name']==NAME for v in allv)
 from portable.context.gcp_budget_watchdog import read_runpod_pods
 pods=read_runpod_pods(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
 assert all(v['status'] in ('EXITED','TERMINATED','STOPPED') for v in pods)
 quota=cloud('regions','describe','us-central1');q={x['metric']:x for x in quota['quotas']}
 assert q['NVIDIA_A100_GPUS']['limit']>=1 and q['NVIDIA_A100_GPUS']['usage']==0
 pinned=json.loads(Path('runs/venus-20261026/us-image-pinned-v1.json').read_text());assert pinned['name']==IMAGE and pinned['status']=='READY'
 run(['gcloud','compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=3h','--project='+PROJECT,'--quiet'])
 startup=R/'bootstrap.sh';startup.write_text('#!/bin/sh\nset -eu\ntest "$(curl -sf -H \'Metadata-Flavor: Google\' http://metadata.google.internal/computeMetadata/v1/instance/name)" = '+NAME+'\nmkdir -p '+ROOT+'\nchmod 1777 '+ROOT+'\nsystemd-run --unit=duplex-venus-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
 op=json.loads(run(['gcloud','compute','instances','create',NAME,'--project='+PROJECT,'--zone='+ZONE,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-native-bounded','--image='+IMAGE,'--image-project=deeplearning-platform-release','--boot-disk-size=100GB','--boot-disk-type=pd-balanced','--boot-disk-device-name='+NAME,'--no-boot-disk-auto-delete','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(startup),'--max-run-duration=7200s','--instance-termination-action=STOP','--format=json','--async','--quiet'],90));op=op[0] if isinstance(op,list) else op;created=True
 record('approved_us_allocation_submitted',operation=op['name'],zone=ZONE,source=source,image=IMAGE,image_id=pinned['id'],hourly_estimate_usd=price['hourly_estimate_usd'],prior_new_gpu_seconds_upper=1200)
 deadline=time.time()+960
 while op['status']!='DONE':
  if time.time()>deadline:raise TimeoutError('Observe same pending operation; no duplicate')
  time.sleep(5);op=cloud('operations','describe',op['name'],'--zone='+ZONE)
 if op.get('error'):raise RuntimeError(json.dumps(op['error']))
 vm=cloud('instances','describe',NAME,'--zone='+ZONE)
 while vm['status']!='RUNNING':
  if time.time()>deadline:raise TimeoutError('No duplicate while allocation pending')
  time.sleep(5);vm=cloud('instances','describe',NAME,'--zone='+ZONE)
 start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+7200;setup_end=start+1800
 disk=cloud('disks','describe',NAME,'--zone='+ZONE);assert int(disk['sizeGb'])==100
 record('actual_running',vm_id=vm['id'],disk_id=disk['id'],last_start=vm['lastStartTimestamp'],native_stop_unix=end)
 for _ in range(30):
  try:keys=cloud('instances','get-guest-attributes',NAME,'--zone='+ZONE,'--query-path=hostkeys/')
  except RuntimeError:keys=[]
  if keys:break
  time.sleep(3)
 else:raise RuntimeError('No authenticated guest hostkeys')
 (R/'.private').mkdir(mode=0o700,exist_ok=True);known=R/'.private/known-hosts';alias='gcp-venus-us-v1';known.write_text(''.join(alias+' '+v['key']+' '+v['value']+'\n' for v in keys));known.chmod(0o600)
 a=['ssh','-i','runs/local-migration-20261024/.private/gcp-native-ssh','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-o','UserKnownHostsFile='+str(known),'-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','daryl_aurium_labs_com@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP']]
 (R/'ssh.json').write_text(json.dumps(a)+'\n')
 for _ in range(12):
  try:ssh('test -d '+ROOT+' && nvidia-smi',20);break
  except RuntimeError:time.sleep(3)
 else:raise RuntimeError('Strict SSH/GPU not ready')
 sync(R/'source.bundle',ROOT+'/source.bundle')
 ssh('git init --bare '+ROOT+'/source.git && git --git-dir='+ROOT+'/source.git fetch '+ROOT+'/source.bundle HEAD && git --git-dir='+ROOT+'/source.git worktree add --detach '+remote+' '+source)
 birth=run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip()
 lease=dict(resource_key='us-a10040',zone=ZONE,vm_id=str(vm['id']),disk_id=str(disk['id']),disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=1200,prior_gpu_seconds=PRIOR_SECONDS,prior_scope_usd=PRIOR_USD,authorization_sha256=hashlib.sha256(auth).hexdigest(),controller_pid=os.getpid(),controller_birth=birth)
 (R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');sync(R/'lease.json',ROOT+'/venus-lease-us-v1.json')
 guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),'--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 until=time.time()+90
 while True:
  if guard.poll() is not None:raise RuntimeError('Actual independent Mac guard exited')
  if (R/'guard.json').exists() and time.time()-json.loads((R/'guard.json').read_text())['checked_unix']<30:break
  if time.time()>until:raise TimeoutError('No fresh guard')
  time.sleep(1)
 ssh('sudo systemd-run --unit=duplex-venus-guest-guard-us-v1 --property=RuntimeMaxSec=7300 --working-directory='+remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+ROOT+'/venus-live-guard-v1.json --lease '+ROOT+'/venus-lease-us-v1.json')
 assert props('duplex-venus-guest-guard-us-v1')['ActiveState']=='active'
 ssh('sudo systemctl stop duplex-venus-bootstrap-stop.timer');record('both_independent_guards_live',mac_pid=guard.pid)
 # One isolated host venv on official driver580 image; no change to original runtimes.
 ssh('mkdir -p '+ROOT+'/venus-runtime-v1 '+ROOT+'/venus-public-v1 '+ROOT+'/venus-inputs-v1 && df -h '+ROOT)
 setup='set -eu; sudo apt-get update -qq; sudo apt-get install -y -qq python3-venv libsndfile1; /usr/bin/python3 -m venv '+ROOT+'/venus-runtime-v1/venv; '+ROOT+'/venus-runtime-v1/venv/bin/pip install --disable-pip-version-check --no-cache-dir -r '+remote+'/portable/venus/requirements.txt; '+ROOT+'/venus-runtime-v1/venv/bin/pip freeze > '+ROOT+'/venus-runtime-v1/pip-freeze.txt; '+ROOT+'/venus-runtime-v1/venv/bin/python -c '+shlex.quote("import json,torch,transformers,stepaudio2;assert torch.cuda.is_available() and torch.cuda.device_count()==1;assert torch.__version__.startswith('2.4.0');assert transformers.__version__=='4.51.3';json.dump(dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),optimizer_updates=0),open('/workspace/migration/venus-runtime-v1/setup-complete.json','x'))")
 ssh('sudo systemd-run --unit=duplex-venus-setup-us-v1 --uid=daryl_aurium_labs_com --property=RuntimeMaxSec=1500 /bin/bash -c '+shlex.quote(setup))
 modelbase=Path.home()/'.cache/duplex-live/public-models/inclusionAI--Realtime-Venus@4681004131ea'
 run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=core','--exclude=core.*','--files-from=runs/venus-20261026/public-model-files.txt','-e',shlex.join(a[:-1]),str(modelbase)+'/',a[-1]+':'+ROOT+'/venus-public-v1/'],900)
 sync('runs/venus-20261026/inputs/',ROOT+'/venus-inputs-v1/',90)
 sync('runs/venus-20261026/public-model-manifest.json',ROOT+'/public-model-manifest.json')
 code="""import hashlib,json
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
rows=json.loads(Path('/workspace/migration/public-model-manifest.json').read_text());base=Path('/workspace/migration/venus-public-v1')
assert len(rows)==35
for x in rows:
 p=base/x['path'];assert p.stat().st_size==x['bytes'] and digest(p)==x['sha256'],x['path']
print(json.dumps(dict(all_hashes_verified=True,files=len(rows),bytes=sum(x['bytes'] for x in rows))))"""
 out=ssh('/usr/bin/python3 -c '+shlex.quote(code),180)
 (R/'model-hashes.json').write_text(out);record('destination_public_model_hashes_pass',receipt=json.loads(out))
 terminal=wait_unit('duplex-venus-setup-us-v1',setup_end)
 (R/'setup-journal.log').write_text(ssh('sudo journalctl -u duplex-venus-setup-us-v1 --no-pager -o cat',60))
 assert terminal['Result']=='success' and terminal['ExecMainStatus']=='0',terminal
 record('isolated_runtime_actual_success',terminal=terminal)
 py=ROOT+'/venus-runtime-v1/venv/bin/python'
 manifest=ROOT+'/venus-inputs-v1/container-manifest.json'
 out=ssh('cd '+remote+' && '+py+' -m portable.venus.benchmark --preflight-only --manifest '+manifest,60);assert json.loads(out)['preflight_passed'];record('all18_inputs_preflight_pass',result=json.loads(out))
 env='HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME='+ROOT+'/venus-runtime-v1/hf-cache DUPLEX_LIVE_GUARD_FILE='+ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+ROOT+'/venus-lease-us-v1.json'
 ssh('sudo systemd-run --unit=duplex-venus-app-us-v1 --uid=daryl_aurium_labs_com --working-directory='+remote+' --property=RuntimeMaxSec='+str(max(1,int(end-time.time()-60)))+' /usr/bin/env '+env+' '+py+' -u -m portable.venus.native_app --model-dir '+ROOT+'/venus-public-v1/Realtime-Venus-Audio --port 8032')
 record('one_native_model_launch',terminal=props('duplex-venus-app-us-v1'))
 while True:
  try:
   health=json.loads(ssh('curl -sf --max-time 4 http://127.0.0.1:8032/health',10))
   if health.get('ready'):break
  except RuntimeError:pass
  state=props('duplex-venus-app-us-v1')
  if state['ActiveState'] in ('failed','inactive'):raise RuntimeError('Actual model startup failed: '+str(state))
  if time.time()>setup_end:raise TimeoutError('Thirty-minute setup limit')
  time.sleep(3)
 (R/'health.json').write_text(json.dumps(health,indent=2)+'\n');record('native_model_ready',profile=health)
 ssh('sudo systemd-run --unit=duplex-venus-benchmark-us-v1 --uid=daryl_aurium_labs_com --working-directory='+remote+' --property=RuntimeMaxSec=1200 '+py+' -u -m portable.venus.benchmark --manifest '+manifest+' --output '+ROOT+'/venus-runtime-v1/diagnostic-v1')
 record('actual_complete10_started',terminal=props('duplex-venus-benchmark-us-v1'))
 terminal=wait_unit('duplex-venus-benchmark-us-v1',min(end-60,time.time()+1250));record('actual_complete10_terminal',terminal=terminal)
 copyback()
 assert terminal['Result']=='success' and terminal['ExecMainStatus']=='0',terminal
 record('all10_collected_unqualified',results=json.loads((R/'copyback/venus-runtime-v1/diagnostic-v1/results.json').read_text()))
except BaseException as e:
 record('controller_failure',error_type=type(e).__name__,error=str(e)[-2500:])
 if a:
  try:copyback()
  except Exception as x:record('copyback_failed_disk_preserved',error=str(x)[-1000:])
 raise
finally:
 if guard and guard.poll() is None:
  (R/'stop-requested').write_text('closing\n')
  try:guard.wait(timeout=150)
  except subprocess.TimeoutExpired:record('guard_stop_observation_timeout',pid=guard.pid)
 if created:
  try:
   current=cloud('instances','describe',NAME,'--zone='+ZONE)
   if vm is not None:assert str(current['id'])==str(vm['id'])
   if current['status'] not in ('TERMINATED','STOPPED'):
    run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+PROJECT,'--quiet','--discard-local-ssd=True'],150)
   current=cloud('instances','describe',NAME,'--zone='+ZONE);(R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n');record('provider_stopped',vm_id=current['id'],status=current['status'])
  except Exception as x:record('provider_observation_error',error=str(x)[-1500:])
 awake.terminate()
