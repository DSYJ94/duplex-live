"""One explicitly approved same-VM British native evaluation, ZERO updates.

Offline --plan has no provider calls. --execute never retries a start. Output
failure remains a nonzero result, even when provider cleanup succeeds.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error

from . import british_speech_guard as g, results_recovery as recovery

ATTEMPT=int(os.environ.get('DUPLEX_BRITISH_START_ATTEMPT','1'))
R=g.PLAN.parent if ATTEMPT==1 else g.PLAN.parent/('capacity-attempt-%02d'%ATTEMPT)
NAME='duplex-venus-large-gpu-20261104-eua-v67'
DISK='duplex-venus-large-data-20261104-eua-v67'
ZONE='europe-west4-a'
OUT='/workspace/migration/venus-british-native-v83'
OLD='/workspace/migration/venus-large-capacity-v65/native-v75'
BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v75/british-native-v83')
REFS=Path('runs/british-reference-replacement-20261106')
PRICE_SCHEMA=Path('docs/NATIVE-BRITISH-GCP-PRICE-SKU-SCHEMA-2026-11-06.json')


def save(name,value):
    R.mkdir(parents=True,exist_ok=True)
    (R/name).write_text(json.dumps(value,indent=2)+'\n')


def quote():
    schema=json.loads(PRICE_SCHEMA.read_text());found={};page='';now=datetime.now(timezone.utc)
    token=recovery.run([recovery.GCLOUD,'auth','print-access-token']).strip()
    for _ in range(20):
        url='https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?currencyCode=USD&pageSize=5000'
        if page:url+='&pageToken='+urllib.parse.quote(page,safe='')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Bearer '+token}),timeout=45) as response:data=json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code!=429 or attempt==2:raise
                time.sleep(20*(attempt+1))
        for row in data.get('skus',[]):
            key=row['skuId']
            if key not in schema:continue
            expected=schema[key]
            prices=[p for p in row['pricingInfo'] if datetime.fromisoformat(p['effectiveTime'].replace('Z','+00:00'))<=now]
            p=max(prices,key=lambda p:p['effectiveTime'])['pricingExpression'];v=p['tieredRates'][0]['unitPrice']
            if row['description']!=expected['description'] or row['serviceRegions']!=expected['regions'] or p['usageUnit']!=expected['usage_unit'] or row['category']['usageType']!='OnDemand':raise ValueError('Exact current Netherlands price units/SKUs required')
            found[key]=int(v.get('units',0))+v.get('nanos',0)/1e9
        if set(found)==set(schema):break
        page=data.get('nextPageToken','')
        if not page:break
    if set(found)!=set(schema):raise ValueError('Complete all-in GPU quote required')
    hourly=found['AC33-B5A5-C20F']+12*found['896E-DB80-1131']+170*found['A78A-A063-7D5A']+(375*found['EFD6-4D88-8C94']+250*found['5CFD-16B2-4F41'])/730
    result=dict(at_utc=now.isoformat(),source='Official Cloud Billing Catalog exact Netherlands SKUs',hourly_USD=hourly,rates=found)
    save('price.json',result);return result


def price_admission(price,now):
    age=now-datetime.fromisoformat(price['at_utc']).timestamp()
    if price.get('source')!='Official Cloud Billing Catalog exact Netherlands SKUs' or not 0<=age<3600 or not 0<price['hourly_USD']<=5.9:
        raise ValueError('Fresh all-in price <=5.9USD/hour required')
    return price['hourly_USD']


def capacity_retry_after(operation,previous,current,attempt):
    """A terminal stockout with no actual start cannot consume GPU hours."""
    errors=operation.get('error',{}).get('errors',[])
    if attempt not in (2,3) or operation.get('status')!='DONE' or not errors or any(e['code'] not in ('ZONE_RESOURCE_POOL_EXHAUSTED','ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS') for e in errors):
        raise ValueError('Only bounded terminal capacity-only retries admitted')
    if any(v['id']!=g.VM_ID or v['status']!='TERMINATED' for v in (previous,current)) or any(current.get(k)!=previous.get(k) for k in ('lastStartTimestamp','lastStopTimestamp')):
        raise ValueError('No retry after partial start/access/unknown state')
    return datetime.fromisoformat(operation['endTime']).timestamp()+600


def common_args(source,worktree):
    data='/workspace/migration/venus-training-v1'
    return ['--root',OUT,'--model','/workspace/migration/venus-public-v1/Realtime-Venus-Audio',
        '--lease','/workspace/migration/british83-lease.json','--source',source,
        '--plan',worktree+'/evals/fixtures/venus-control-repair-source-v2-20261027.json',
        '--timelines',data+'/paced-timelines-v22','--approval',data+'/paced-data-approval-v22.json',
        '--word-quality',data+'/paced-word-quality-v22.json',
        '--old-plan',worktree+'/evals/fixtures/venus-control-corpus-v4-20261026.json',
        '--old-timelines',data+'/timelines-v7-repaired','--old-approval',data+'/data-approval-v7.json',
        '--voice-reference',OUT+'/references/rp-ben-16k.wav','--voice-reference-sha256','552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0']


def stages(source,worktree):
    args=common_args(source,worktree)
    return [('audition','training.venus.british_audition',args),
            ('main','training.venus.british_behavior',args+['--arm','main']),
            ('control','training.venus.british_behavior',args+['--arm','control']),
            ('combine','training.venus.british_behavior',args+['--combine'])]


def make_plan(vm,source):
    p=dict(resource_key=g.KEY,authorization_sha256=g.sha(g.AUTH),vm_id=g.VM_ID,disk_id=g.DISK_ID,zone=ZONE,
        disk_size_gib=250,prior_GPU_seconds_upper=45500,prior_gpu_seconds=134651,prior_scope_usd=165.7900361111111,
        actual_GPU_interval=13,maximum_native_GPU_seconds=3500,shutdown_reserve_seconds=100,storage_deadline_unix=g.DEADLINE,
        new_optimizer_updates=0,required_native_cases=64,required_voice_audition_cases=9,qualified=False,
        checkpoint_hashes=g.CHECKPOINTS,training_math_source='b39ce3ebe167b059b67fa3d131978e5da7c41ac1',execution_source=source,
        selected_profile='rp-ben',reference_sha256='552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0',
        runtime_file_sha256={f:g.sha(f) for f in g.RUNTIME_FILES},backup_receipt_sha256=g.sha(g.BACKUP),
        audition_reference_paths={'rp-ben':OUT+'/references/rp-ben-16k.wav','british-alice':OUT+'/references/british-alice-16k.wav','british-george-comparator':'/workspace/migration/venus-training-v1/george-reference-v33/reference-16k.wav'})
    return p


def validate_outputs(root,source):
    paired=json.loads((root/'native/paired-native-report.json').read_text())
    audition=json.loads((root/'audition/voice-audition.json').read_text())
    for report,count in ((paired,64),(audition,9)):
        if not report.get('complete') or len(report.get('cases',[]))!=count or report.get('source')!=source or report.get('new_optimizer_updates')!=0 or report.get('word_limit')!=.05 or report.get('peak_limit')!=.995 or report.get('qualified') is not False or report.get('adopted') is not False:
            raise ValueError('Actual complete source-bound native reports required; collection is not adoption')
    rows=json.loads((root/'native/results.json').read_text())['cases']
    identities=[{r['id']:r['input_sha256'] for r in rows if r['condition']==arm} for arm in ('main','control')]
    if len(rows)!=64 or len(identities[0])!=32 or identities[0]!=identities[1]:raise ValueError('Whole paired32 input identities required')
    return True


def copyback(c,remaining,*,require_complete=False):
    """New outputs only, JSON first, bounded batches; never copy weight links."""
    script='''import pathlib,hashlib,json,os
r=pathlib.Path(%r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
print(json.dumps(dict(files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()],links=[dict(path=str(p.relative_to(r)),target=os.readlink(p)) for p in sorted(r.rglob('*')) if p.is_symlink()])))
'''%OUT
    inv=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(script),60));save('remote-inventory.json',inv)
    if sum(row['bytes'] for row in inv['files'])>100_000_000_000:raise ValueError('Artifact transfer allowance exceeded')
    for row in inv['files']+inv['links']:recovery.checked_path(BACKUP,row['path'])
    if any(not row['target'].startswith(OLD+'/') for row in inv['links']):raise ValueError('Unapproved artifact link target')
    BACKUP.mkdir(parents=True,exist_ok=True)
    rows=sorted(inv['files'],key=lambda x:(not x['path'].endswith(('.json','.jsonl','.log')),x['path']))
    work=list(recovery.batches(rows))
    work.append([dict(path=x['path'],bytes=0,target=x['target']) for x in inv['links']])
    ssh=[('ControlMaster=no' if x.startswith('ControlMaster=') else 'ControlPath=none' if x.startswith('ControlPath=') else x) for x in c.ssh_argv]
    for n,batch in enumerate(work):
        if not batch:continue
        files=R/('copyback-%03d.files'%n);files.write_bytes(b''.join(x['path'].encode()+b'\0' for x in batch))
        seconds=min(60,int(remaining()-30))
        if seconds<1:raise TimeoutError('Preserve STOP margin; partial output remains on disk')
        argv=['/opt/homebrew/bin/rsync','-ar','--no-owner','--no-group','--partial-dir=.rsync-partial',*['--exclude='+x for x in sorted(recovery.EXCLUDED|{'core','core.*'})],
              '--from0','--files-from='+str(files),'-e',shlex.join(ssh[:-1]),ssh[-1]+':'+OUT+'/',str(BACKUP)+'/']
        c.run(argv,seconds)
        for row in batch:
            path=recovery.checked_path(BACKUP,row['path'])
            if 'target' in row:
                if not path.is_symlink() or os.readlink(path)!=row['target']:raise ValueError('Artifact link changed')
            elif path.is_symlink() or path.stat().st_size!=row['bytes'] or recovery.sha(path)!=row['sha256']:raise ValueError('Original output hash mismatch')
        save('copyback-%03d-verified.json'%n,dict(files=len(batch),bytes=sum(x['bytes'] for x in batch)))
    if require_complete:validate_outputs(BACKUP,json.loads((R/'claim.json').read_text())['source'])
    save('copyback-verified.json' if require_complete else 'partial-copyback-verified.json',dict(all_hashes_verified=True,all_link_targets_verified=True,files=len(inv['files']),bytes=sum(x['bytes'] for x in inv['files']),links=len(inv['links']),inventory_sha256=g.sha(R/'remote-inventory.json'),backup=str(BACKUP),complete_native_cohorts=require_complete,qualified=False))


def stop_exact(c):
    vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
    if vm['id']!=g.VM_ID:raise ValueError('Never stop a replacement machine')
    if vm['status']!='TERMINATED':c.run([recovery.GCLOUD,'compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--discard-local-ssd=True','--quiet'],150)
    vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-final.json',vm)
    if vm['status']!='TERMINATED':raise ValueError('GPU stop unverified; preserve original state')
    return vm


def execute():
    g.authority()  # Missing/changed approval fails BEFORE all provider calls.
    from . import controller as c
    from .source_controller import prepare_ssh
    from .oslogin_access import ensure as ensure_key
    from portable.venus.guard import bounded_runpod_read
    retry_baseline=None
    if ATTEMPT>1:
        previous_root=g.PLAN.parent if ATTEMPT==2 else g.PLAN.parent/('capacity-attempt-%02d'%(ATTEMPT-1))
        initial=json.loads((previous_root/'start-operation.json').read_text());initial=initial[0] if isinstance(initial,list) else initial
        op=c.cloud('compute','operations','describe',initial['name'],'--zone='+ZONE)
        previous=json.loads((previous_root/'provider-final.json').read_text())
        current=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
        after=capacity_retry_after(op,previous,current,ATTEMPT)
        retry_baseline=previous
        save('previous-capacity-terminal.json',op)
        save('capacity-wait.json',dict(attempt=ATTEMPT,not_before_unix=after,actual_new_GPU_intervals=0))
        while time.time()<after:time.sleep(min(30,after-time.time()))
        if not (R/'price.json').exists():
            previous_price=json.loads((previous_root/'price.json').read_text())
            try:price_admission(previous_price,time.time());save('price.json',previous_price)
            except ValueError:quote()
    hourly=price_admission(json.loads((R/'price.json').read_text()),time.time())
    needs_recovery=not g.BACKUP.exists()
    if needs_recovery:
        from .british_files_first import precheck
        precheck()
    else:g.complete_backup()
    if time.time()+3800>=g.DEADLINE:raise ValueError('Whole native phase cannot fit storage deadline')
    if not all(v['status']=='TERMINATED' for v in c.cloud('compute','instances','list')):raise ValueError('Other compute active')
    vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);disk=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE)
    recovery.validate_original(vm,disk,attached=True)
    old=json.loads((recovery.PRIOR/'provider-final.json').read_text())
    expected_stop=(retry_baseline or old)['lastStopTimestamp']
    # GCP advances lastStopTimestamp on a terminal failed start as well. An
    # unchanged lastStart plus the exact previously verified stockout snapshot
    # distinguishes it from a real new running interval.
    if vm['lastStartTimestamp']!=old['lastStartTimestamp'] or vm['lastStopTimestamp']!=expected_stop:raise ValueError('An unknown GPU interval has occurred')
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip())
    if isinstance(pods,dict):pods=pods['pods']
    if any(p['status'] not in ('EXITED','STOPPED','TERMINATED') for p in pods):raise ValueError('Runpod overlap')
    source=c.run(['git','rev-parse','HEAD']).strip()
    paths=['training/venus','portable/venus','evals/fixtures',str(g.AUTH),str(g.PROPOSAL)]
    if c.run(['git','-c','core.filemode=false','status','--porcelain','--untracked-files=no','--',*paths]).strip():raise ValueError('Source bytes must be committed')
    # Validate all local prepared reference bytes BEFORE paying for a start.
    from .british_audition import REFS as hashes
    for profile in ('rp-ben','british-alice'):
        if recovery.sha(REFS/(profile+'-16k.wav'))!=hashes[profile]:raise ValueError('Prepared reference changed')
    R.mkdir(parents=True,exist_ok=True)
    with (R/'claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),source=source,authorization_sha256=g.sha(g.AUTH)),f)
    c.R=R;c.NAME=NAME;c.events=[];c.UNITS=[];c.source=source;c.remote='/workspace/migration/duplex-british-'+source[:12]
    c.run(['git','bundle','create',str(R/'source.bundle'),'HEAD'],120)
    ensure_key(time.time()+3700,R)
    boot=R/'bootstrap.sh';boot.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-british83-bootstrap-stop --on-active=300s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
    watch=None;file_watch=None;submitted=False;ssh_ready=False;complete=False;backed=False;end=None
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
    try:
        c.run([recovery.GCLOUD,'compute','instances','add-metadata',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--metadata-from-file=startup-script='+str(boot),'--quiet'],60)
        c.cloud('compute','instances','set-scheduling',NAME,'--zone='+ZONE,'--max-run-duration=3500s','--instance-termination-action=STOP','--discard-local-ssds-at-termination-timestamp=true','--no-restart-on-failure','--maintenance-policy=TERMINATE')
        submitted=True
        op=c.cloud('compute','instances','start',NAME,'--zone='+ZONE,'--async');save('start-operation.json',op)
        recovery.wait_operation(op,180)
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-start.json',vm)
        if vm['status']!='RUNNING' or vm['lastStartTimestamp']==old['lastStartTimestamp']:raise ValueError('No new actual GPU start')
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+3500
        if needs_recovery:
            save('file-lease.json',dict(start_unix=end-3500,stop_unix=end,controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip()))
            file_watch=subprocess.Popen([sys.executable,'-u','-m','training.venus.british_controller','--file-guard'],stdout=(R/'file-guard.log').open('x'),stderr=subprocess.STDOUT)
            until=time.time()+60
            while not (R/'file-guard.json').exists():
                if file_watch.poll() is not None or time.time()>until:raise RuntimeError('Files-first guard not ready')
                time.sleep(1)
        prepare_ssh(vm,ZONE,'83british',require_gpu=True);ssh_ready=True
        if c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip():raise ValueError('Model already active')
        if needs_recovery:
            from .british_files_first import recover
            recover(c,end,file_watch);g.complete_backup()
        c.sync(R/'source.bundle','/workspace/migration/british83-source.bundle')
        c.ssh('git --git-dir=/workspace/migration/source.git fetch /workspace/migration/british83-source.bundle HEAD && git --git-dir=/workspace/migration/source.git worktree add --detach '+c.remote+' '+source,90)
        c.ssh('mkdir -p '+OUT+'/references '+c.remote+'/'+str(R)+' '+c.remote+'/'+str(g.BACKUP.parent))
        for profile in ('rp-ben','british-alice'):c.sync(REFS/(profile+'-16k.wav'),OUT+'/references/'+profile+'-16k.wav')
        c.sync(g.BACKUP,c.remote+'/'+str(g.BACKUP))
        plan=make_plan(vm,source)
        with g.PLAN.open('x') as f:json.dump(plan,f,indent=2)
        lease=dict(plan,execution_plan_sha256=g.sha(g.PLAN),creation_timestamp=vm['creationTimestamp'],stop_unix=end,
            verified_hourly_price_usd=hourly,controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        save('lease.json',lease);g.resource(lease)
        c.sync(g.PLAN,c.remote+'/'+str(g.PLAN));c.sync(R/'lease.json','/workspace/migration/british83-lease.json')
        watch=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),'--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT)
        c.guard=watch
        until=time.time()+90
        while not (R/'guard.json').exists():
            if watch.poll() is not None or time.time()>until:raise RuntimeError('Independent Mac guard failed before model')
            time.sleep(1)
        c.ssh('sudo systemd-run --unit=duplex-british83-guest --property=RuntimeMaxSec=3500 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard /workspace/migration/venus-live-guard-v1.json --lease /workspace/migration/british83-lease.json')
        if c.props('duplex-british83-guest')['ActiveState']!='active':raise ValueError('Guest guard not active')
        c.ssh('sudo systemctl stop duplex-british83-bootstrap-stop.timer')
        save('both-guards-ready.json',dict(mac_pid=watch.pid,guest=c.props('duplex-british83-guest'),native_STOP_unix=end))
        if file_watch:
            if file_watch.wait(timeout=30)!=0:raise RuntimeError('Files-first guard handoff failed')
        # Original proof metadata and saved checkpoints stay original, never rewritten.
        setup='import pathlib;root=pathlib.Path('+repr(OUT)+');old=pathlib.Path('+repr(OLD)+');'
        setup+='\nfor arm in ("main","control"):\n d=root/arm;d.mkdir(exist_ok=False)\n for name in ("training-terminal.json","final-native-restore.json","step-0060.pt"):(d/name).symlink_to(old/arm/name)'
        c.ssh('/usr/bin/python3 -c '+shlex.quote(setup))
        py='/workspace/migration/venus-runtime-v1/venv/bin/python'
        runtime="import torch,importlib.metadata as m,hashlib,pathlib;assert torch.__version__=='2.4.0+cu121' and torch.cuda.is_available() and torch.cuda.device_count()==1;assert m.version('minicpmo-utils')=='1.0.6';assert hashlib.sha256(pathlib.Path('/workspace/migration/venus-public-v1/Realtime-Venus-Audio/modeling_minicpmo.py').read_bytes()).hexdigest()=='b7b4070cead9b88347f6e39d8333706ec007e2a832d6bcde5755063bf1453774';p=m.distribution('minicpmo-utils').locate_file('stepaudio2/token2wav.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='0598c25ad9883f75d6d5ca1e9276c03adbfa0357a29442164198ea6931a06efb';print('Pinned native source, tokenizer, Torch and one CUDA GPU verified')"
        save('runtime-source-check.json',dict(stdout=c.ssh(py+' -c '+shlex.quote(runtime),30)))
        preflight='cd '+c.remote+' && '+py+' -m unittest evals.harness.test_venus_completion_python_runtime -q'
        c.ssh(preflight,30)
        env=['env','DUPLEX_LIVE_GUARD_FILE=/workspace/migration/venus-live-guard-v1.json','DUPLEX_LIVE_LEASE_FILE=/workspace/migration/british83-lease.json','HF_HOME=/workspace/migration/venus-runtime-v1/hf-cache',py,'-u','-m']
        for label,module,args in stages(source,c.remote):
            seconds=int(end-time.time()-360)
            if seconds<120:raise TimeoutError('Full remaining stages/backup cannot fit; no subset acceptance')
            unit='duplex-british83-'+label
            c.start_unit(unit,shlex.join(env+[module]+args),seconds)
            c.wait_unit(unit,end-330)
        complete=True;copyback(c,lambda:end-time.time(),require_complete=True);backed=True
    except BaseException as error:
        save('failure.json',dict(error_type=type(error).__name__,error=str(error)[-1800:],source=source,qualified=False));raise
    finally:
        try:
            if ssh_ready:
                for unit in c.UNITS:
                    try:c.ssh('sudo systemctl stop '+unit,20)
                    except Exception:pass
                if not backed and end and end-time.time()>60:
                    partial_deadline=min(end,time.time()+120)
                    try:copyback(c,lambda:partial_deadline-time.time())
                    except Exception as error:save('partial-copyback-failure.json',dict(error=str(error)[-1500:]))
            if submitted:
                final=stop_exact(c)
                if final.get('lastStartTimestamp')!=old.get('lastStartTimestamp'):
                    elapsed=datetime.fromisoformat(final['lastStopTimestamp']).timestamp()-datetime.fromisoformat(final['lastStartTimestamp']).timestamp()
                    save('actual-usage.json',dict(seconds=elapsed,prior_GPU_upper_seconds=45500,qualified=False,complete_native_stages=complete,full_copyback_verified=backed))
        finally:
            if watch:
                (R/'stop-requested').write_text('closed\n')
                try:watch.wait(timeout=25)
                except subprocess.TimeoutExpired:watch.terminate()
            if file_watch:
                try:file_watch.wait(timeout=25)
                except subprocess.TimeoutExpired:file_watch.terminate()
            awake.terminate()
    if not complete or not backed:raise RuntimeError('Incomplete native phase; never imply qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    p.add_argument('--attempt',type=int,choices=(1,2,3),default=ATTEMPT)
    for mode in ('plan','quote','execute','file-guard'):m.add_argument('--'+mode,action='store_true')
    a=p.parse_args()
    ATTEMPT=a.attempt;os.environ['DUPLEX_BRITISH_START_ATTEMPT']=str(ATTEMPT)
    R=g.PLAN.parent if ATTEMPT==1 else g.PLAN.parent/('capacity-attempt-%02d'%ATTEMPT)
    if a.plan:print(json.dumps(dict(stages=[dict(name=n,module=module) for n,module,args in stages('SOURCE','WORKTREE')],required_native_cases=64,required_separate_audition_cases=9,maximum_native_seconds=3500,shutdown_reserve_seconds=100,authorization_present=g.AUTH.exists(),cloud_mutations=0)))
    elif a.quote:print(json.dumps(quote()))
    elif a.file_guard:
        from .british_files_first import file_guard
        file_guard()
    else:execute()
