"""CPU-only retrieval of exact linked training data before the retained disk expires.

No model, optimizer, GPU allocation, credential upload or public-weight copy.
Missing owner authorization fails before any provider call.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time

from . import results_recovery as r

ROOT=Path('runs/isambard-cutover-20261106/data-rescue-v2')
AUTH=Path('docs/NATIVE-ISAMBARD-DATA-RESCUE-RESUME-AUTHORIZATION-2026-11-06.json')
PROPOSAL=Path('docs/NATIVE-ISAMBARD-DATA-RESCUE-RESUME-PROPOSAL-2026-11-06.json')
CACHE=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/context-v52/cache')
DEST=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/isambard-data-rescue-v1')
MANIFEST_SHA='1bf09867b5d1ce78ed4c3504057773d20e0025bb0662e7442311fd79b08a4807'
CPU='duplex-venus-data-rescue-20261106-v2'
MAX_BYTES=20_000_000_000
BASE=Path('/workspace/migration')
CPU_SECONDS=2600
PRIOR_CPU_SECONDS_UPPER=100
CPU_AGGREGATE_SECONDS_UPPER=2800
PRIOR_RESCUE=Path('runs/isambard-cutover-20261106/data-rescue-v1')
PRIOR_FINAL_SHA='48e5ae369827e55e883aabdd1088db31f205aa819e646b638358b955be9da23b'
PREVIOUS_GPU_RECEIPT=Path('runs/venus-large-capacity-20261104/british-native-v85/provider-after-interruption.json')
EXTRAS=('venus-training-v1/paced-timelines-v22','venus-training-v1/timelines-v7-repaired',
        'venus-training-v1/george-reference-v33','venus-british-native-v85')
SINGLE=('venus-training-v1/paced-data-approval-v22.json','venus-training-v1/paced-word-quality-v22.json',
        'venus-training-v1/data-approval-v7.json')


def authority():
    a=json.loads(AUTH.read_text());p=json.loads(PROPOSAL.read_text())
    if a.get('status')!='APPROVED' or not a.get('human_reply_verbatim') or a.get('proposal_path')!=str(PROPOSAL) or a.get('proposal_sha256')!=r.sha(PROPOSAL):
        raise ValueError('Exact CPU-only data-rescue approval required')
    expected=dict(phase_ceiling_USD=175,project_ceiling_USD=350,recovery_maximum_USD=5,
        maximum_additional_CPU_running_intervals=3,maximum_CPU_running_seconds=2600,
        prior_CPU_seconds_upper=100,remaining_aggregate_CPU_seconds_upper=2700,
        combined_CPU_seconds_upper=2800,
        same_disk_id=r.DISK_ID,original_GPU_id=r.GPU_ID,new_GPU_intervals=0,
        new_optimizer_updates=0,maximum_transfer_bytes=MAX_BYTES)
    if any(a.get(k)!=v or p.get(k)!=v for k,v in expected.items()):raise ValueError('CPU rescue scope changed')
    if time.time()+3000>=r.DEADLINE:raise ValueError('Retained disk deadline cannot fit rescue')
    return a


def prior_usage():
    """Bind this second attempt to the stopped, removed first machine."""
    p=PRIOR_RESCUE/'provider-final.json'
    if r.sha(p)!=PRIOR_FINAL_SHA:raise ValueError('Original CPU terminal receipt changed')
    vm=json.loads(p.read_text())
    seconds=(datetime.fromisoformat(vm['lastStopTimestamp'])-datetime.fromisoformat(vm['lastStartTimestamp'])).total_seconds()
    restored=json.loads((PRIOR_RESCUE/'original-binding-restored.json').read_text())
    removed=json.loads((PRIOR_RESCUE/'temporary-cpu-removed.json').read_text())
    if vm['id']!='3473476360381416918' or vm['status']!='TERMINATED' or not 0<seconds<=PRIOR_CPU_SECONDS_UPPER:raise ValueError('First CPU running time/identity differs')
    if restored!=dict(original_vm_id=r.GPU_ID,disk_id=r.DISK_ID,status='TERMINATED',auto_delete=False) or not removed.get('original_disk_preserved') or removed.get('name')!=vm['name']:raise ValueError('First CPU cleanup not proven')
    if (PRIOR_RESCUE/'inventory.json').exists() or list(PRIOR_RESCUE.glob('batch-*-verified.json')):raise ValueError('Prior payload must be accounted before another transfer')
    if PRIOR_CPU_SECONDS_UPPER+CPU_SECONDS+100!=CPU_AGGREGATE_SECONDS_UPPER:raise ValueError('Original aggregate CPU time changed')
    return dict(prior_actual_seconds=seconds,prior_seconds_upper=PRIOR_CPU_SECONDS_UPPER,
                this_native_max_seconds=CPU_SECONDS,shutdown_reserve_seconds=100,
                combined_seconds_upper=CPU_AGGREGATE_SECONDS_UPPER)


def configure():
    r.R=ROOT;r.CPU=CPU;r.RUN_SECONDS=CPU_SECONDS;r.ENTRYPOINT='training.venus.data_rescue'
    r.AUTH=AUTH;r.PROPOSAL=PROPOSAL
    r.require_authorization=authority


def request(manifest_path=None):
    manifest_path=Path(manifest_path) if manifest_path else CACHE/'manifest.json'
    if r.sha(manifest_path)!=MANIFEST_SHA:raise ValueError('Original304 cache manifest changed')
    m=json.loads(manifest_path.read_text());rows=[]
    for row in m['scenes']:
        path=PurePosixPath(row['file'])
        if path.parts[0]=='old':
            if len(path.parts)!=2 or '..' in path.parts:raise ValueError('Unexpected cache path')
            rows.append(dict(path='venus-training-v1/paced-prefix-cache-v24/'+path.name,sha256=row['sha256']))
    if len(m['scenes'])!=304 or len(rows)!=268 or len({x['path'] for x in rows})!=268:raise ValueError('Whole old268 required')
    return dict(cache_manifest_sha256=MANIFEST_SHA,required_cache=rows)


def allowed(path):
    p=PurePosixPath(path)
    forbidden={'.private','credentials','owner-conversations','owner_conversations','session-token','api keys.md'}
    return not p.is_absolute() and '..' not in p.parts and not any(x.lower() in forbidden or x.lower().startswith('.env') or x=='core' or x.startswith('core.') or x.endswith(('.key','.pem','.env','.core')) for x in p.parts)


def inventory(request_path,output):
    authority();q=json.loads(Path(request_path).read_text());expected=request('/workspace/migration/data-rescue-cache-manifest.json')
    # The remote has a Git-deployed source request, not the Mac's cache directory.
    if q!=expected:raise ValueError('Bound cache request differs')
    base=BASE.resolve();files={};links=[];cache_sources=[]
    def add(path,required_sha=None):
        rel=str(path.relative_to(base))
        if not allowed(rel):return
        if path.is_symlink():
            if required_sha:raise ValueError('Exact cache target must be a real file')
            target=path.resolve()
            if target.is_relative_to(base) and allowed(str(target.relative_to(base))):
                links.append(dict(path=rel,target=os.readlink(path)))
                # Preserve referenced synthetic audio/data, never linked weights
                # or anything outside the named native training workspace.
                if target.is_file() and target.is_relative_to(base/'venus-training-v1') and target.suffix in ('.wav','.npy','.npz','.json','.jsonl'):add(target)
            return
        if not path.resolve().is_relative_to(base):raise ValueError('Native source parent escapes approved workspace')
        if not path.is_file():return
        row=dict(path=rel,bytes=path.stat().st_size,sha256=r.sha(path))
        if required_sha and row['sha256']!=required_sha:raise ValueError('Original cached file hash differs: '+rel)
        files[rel]=row
    for row in q['required_cache']:
        p=base/row['path']
        if not p.is_file():raise ValueError('Missing exact source cache: '+row['path'])
        target=p.resolve()
        if not target.is_relative_to(base/'venus-training-v1') or not allowed(str(target.relative_to(base))):raise ValueError('Exact cache reference escapes approved native data')
        if p.is_symlink():links.append(dict(path=row['path'],target=os.readlink(p)))
        add(target,row['sha256'])
        cache_sources.append(dict(requested_path=row['path'],stored_path=str(target.relative_to(base)),sha256=row['sha256']))
    for rel in EXTRAS:
        top=base/rel
        if not top.is_dir() or top.is_symlink() or not top.resolve().is_relative_to(base):raise ValueError('Missing/linked named native directory: '+rel)
        for p in top.rglob('*'):add(p)
    for rel in SINGLE:
        p=base/rel
        if not p.is_file():raise ValueError('Missing native approval receipt: '+rel)
        add(p)
    total=sum(x['bytes'] for x in files.values())
    if total>MAX_BYTES:raise ValueError('Rescue exceeds exact20GB transfer cap')
    ordered=sorted(files.values(),key=lambda x:(not x['path'].startswith('venus-training-v1/paced-prefix-cache-v24/'),not x['path'].endswith(('.json','.jsonl')),x['path']))
    Path(output).write_text(json.dumps(dict(files=ordered,links=links,bytes=total,required_cache=268,cache_sources=cache_sources),indent=2)+'\n')


def admit_inventory(inv,q):
    files={x['path']:x for x in inv['files']}
    if len(files)!=len(inv['files']) or inv['bytes']!=sum(x['bytes'] for x in files.values()) or not 0<inv['bytes']<=MAX_BYTES:raise ValueError('Unique bounded complete file inventory required')
    if any(not allowed(p) or x['bytes']<0 for p,x in files.items()):raise ValueError('Private or unsafe inventory file')
    expected={x['path']:x['sha256'] for x in q['required_cache']}
    actual={x['requested_path']:x['sha256'] for x in inv['cache_sources']}
    if len(inv['cache_sources'])!=268 or actual!=expected:raise ValueError('All268 exact data references required')
    if any(files.get(x['stored_path'],{}).get('sha256')!=x['sha256'] for x in inv['cache_sources']):raise ValueError('Every cached reference needs actual inventoried bytes')


def groups(rows):
    batch=[];size=0
    for row in rows:
        if batch and size+row['bytes']>128*1024**2:yield batch;batch=[];size=0
        batch.append(row);size+=row['bytes']
    if batch:yield batch


def process_probe(proc=Path('/proc')):
    """Inspect processes directly; Docker need not be installed on this image."""
    native=[];containers=[]
    for p in proc.glob('[0-9]*'):
        try:args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError,PermissionError):continue
        if int(p.name)==os.getpid():continue
        if any(a.startswith(('training.venus.','portable.venus.native_app')) or PurePosixPath(a).name in ('torchrun','native_app.py','large_train.py') for a in args):native.append(int(p.name))
        if args and (PurePosixPath(args[0]).name.startswith('containerd-shim') or PurePosixPath(args[0]).name in ('conmon','runc','crun')):containers.append(int(p.name))
    return dict(native_model_pids=native,container_process_pids=containers)


def execute():
    authority();usage=prior_usage();q=request();ROOT.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
    r.save('aggregate-time-admission.json',usage)
    price=json.loads((ROOT/'price.json').read_text());hourly=price['hourly_usd']
    computed=2*price['rates']['core']['usd']+8*price['rates']['ram']['usd']
    if not (computed==hourly and price['source']=='Official Cloud Billing Catalog' and price['machine_type']=='e2-standard-2' and price['region']=='europe-west4' and 0<=time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600 and 0<hourly<=.12 and hourly*2800/3600+MAX_BYTES/1e9*.15<5):raise ValueError('Fresh CPU price and bounded whole-copy cost required')
    if any(v['status']!='TERMINATED' for v in r.cloud('compute','instances','list')):raise ValueError('Other GCP compute active')
    vm=r.cloud('compute','instances','describe',r.GPU,'--zone='+r.ZONE);disk=r.cloud('compute','disks','describe',r.DISK,'--zone='+r.ZONE);r.validate_original(vm,disk,attached=True)
    previous=json.loads(PREVIOUS_GPU_RECEIPT.read_text())
    if any(vm[k]!=previous[k] for k in ('lastStartTimestamp','lastStopTimestamp')):raise ValueError('Unknown original GPU interval')
    source=r.run(['git','rev-parse','HEAD']).strip()
    if r.run(['git','-c','core.filemode=false','status','--porcelain','--',__file__,str(PROPOSAL),str(AUTH)]).strip():raise ValueError('Rescue source and exact approval must be committed')
    if any(x['name']==CPU for x in r.cloud('compute','instances','list')):raise ValueError('Observe existing CPU, never duplicate')
    with (ROOT/'claim.json').open('x') as f:json.dump(dict(controller_pid=os.getpid(),source=source,authorization_sha256=r.sha(AUTH)),f)
    (ROOT/'request.json').write_text(json.dumps(q,indent=2)+'\n')
    r.run(['git','bundle','create',str(ROOT/'source.bundle'),'HEAD'],120)
    from .oslogin_access import ensure
    ensure(time.time()+CPU_SECONDS+200,ROOT)
    watch=None;detached=False
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
    try:
        detached=True
        r.run([r.GCLOUD,'compute','instances','detach-disk',r.GPU,'--disk='+r.DISK,'--zone='+r.ZONE,'--project='+r.PROJECT,'--quiet'],90)
        r.validate_original(r.cloud('compute','instances','describe',r.GPU,'--zone='+r.ZONE),r.cloud('compute','disks','describe',r.DISK,'--zone='+r.ZONE),attached=False)
        boot=ROOT/'bootstrap.sh';boot.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-results-stop --on-active='+str(CPU_SECONDS-90)+'s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        op=r.cloud('compute','instances','create',CPU,'--zone='+r.ZONE,'--machine-type=e2-standard-2','--provisioning-model=STANDARD','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+r.DISK+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE,duplex-results-recovery-claim='+r.sha(ROOT/'claim.json'),'--metadata-from-file=startup-script='+str(boot),'--max-run-duration='+str(CPU_SECONDS)+'s','--instance-termination-action=STOP','--async')
        r.save('create-operation.json',op);r.wait_operation(op)
        cpu=r.cloud('compute','instances','describe',CPU,'--zone='+r.ZONE);r.validate_cpu(cpu);r.save('provider-start.json',cpu)
        if cpu['status']!='RUNNING':raise ValueError('CPU did not actually start')
        end=datetime.fromisoformat(cpu['lastStartTimestamp']).timestamp()+CPU_SECONDS
        r.save('lease.json',dict(vm_id=cpu['id'],stop_unix=end,controller_pid=os.getpid(),controller_birth=r.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip()))
        watch=subprocess.Popen([sys.executable,'-u','-m','training.venus.data_rescue','--guard'],stdout=(ROOT/'guard.log').open('x'),stderr=subprocess.STDOUT)
        from . import controller as c
        from .source_controller import prepare_ssh
        c.R=ROOT;c.NAME=CPU
        prepare_ssh(cpu,r.ZONE,'isambard-data-rescue',require_gpu=False,readiness_timeout=240)
        ssh=c.ssh_argv
        if c.ssh('sudo systemctl is-active duplex-results-stop.timer').strip()!='active':raise ValueError('Guest shutdown timer not active')
        if not (ROOT/'guard.json').exists() or watch.poll() is not None or time.time()>json.loads((ROOT/'guard.json').read_text())['expires_unix']:raise ValueError('CPU guard missing/stale')
        c.sync(ROOT/'source.bundle','/workspace/migration/data-rescue-source.bundle')
        work='/workspace/migration/data-rescue-source-'+source[:12]
        c.ssh('git --git-dir=/workspace/migration/source.git fetch /workspace/migration/data-rescue-source.bundle HEAD && git --git-dir=/workspace/migration/source.git worktree add --detach '+work+' '+source)
        state=json.loads(c.ssh('cd '+work+' && /usr/bin/python3 -m training.venus.data_rescue --probe-models',30));r.save('pre-transfer-runtime.json',state)
        if state['native_model_pids'] or state['container_process_pids']:raise ValueError('Unexpected model/container process on CPU')
        c.sync(ROOT/'request.json','/workspace/migration/data-rescue-request.json')
        c.sync(CACHE/'manifest.json','/workspace/migration/data-rescue-cache-manifest.json')
        c.ssh('cd '+work+' && /usr/bin/python3 -m training.venus.data_rescue --inventory --request /workspace/migration/data-rescue-request.json --output /workspace/migration/data-rescue-inventory.json',180)
        inv=json.loads(c.ssh('cat /workspace/migration/data-rescue-inventory.json',30));admit_inventory(inv,q);r.save('inventory.json',inv)
        for n,batch in enumerate(groups(inv['files'])):
            if watch.poll() is not None or end-time.time()<180 or time.time()>json.loads((ROOT/'guard.json').read_text())['expires_unix']:raise RuntimeError('Rescue guard/time exhausted')
            files=ROOT/('batch-%03d.files'%n);files.write_bytes(b''.join(x['path'].encode()+b'\0' for x in batch))
            argv=['/opt/homebrew/bin/rsync','-ar','--no-owner','--no-group','--partial-dir=.rsync-partial',*['--exclude='+x for x in sorted(r.EXCLUDED|{'core','core.*'})],'--from0','--files-from='+str(files),'-e',shlex.join(ssh[:-1]),ssh[-1]+':/workspace/migration/',str(DEST)+'/']
            r.run(argv,min(300,int(end-time.time()-120)))
            for row in batch:
                p=r.checked_path(DEST,row['path'])
                if p.is_symlink() or p.stat().st_size!=row['bytes'] or r.sha(p)!=row['sha256']:raise ValueError('Recovered native artifact differs')
            r.save('batch-%03d-verified.json'%n,dict(files=len(batch),bytes=sum(x['bytes'] for x in batch)))
        for row in inv['links']:
            p=r.checked_path(DEST,row['path']);p.parent.mkdir(parents=True,exist_ok=True)
            if p.is_symlink() and os.readlink(p)==row['target']:continue
            if p.exists() or p.is_symlink():raise ValueError('Never overwrite differing original link')
            p.symlink_to(row['target'])
        r.save('copyback-verified.json',dict(all_hashes_verified=True,files=len(inv['files']),bytes=inv['bytes'],exact_old268_cache_verified=True,cache_sources=inv['cache_sources'],original_cache_manifest_sha256=MANIFEST_SHA,original_link_targets_preserved=inv['links'],new_optimizer_updates=0,new_GPU_jobs=0))
    finally:
        try:
            if detached:
                try:r.restore_binding(watch)
                except Exception as error:
                    r.save('cleanup-incomplete.json',dict(error=str(error),original_disk_must_be_preserved=True));raise
        finally:awake.terminate()


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    for name in ('plan','quote','execute','guard','inventory','probe-models'):m.add_argument('--'+name,action='store_true')
    p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
    if a.plan:print(json.dumps(dict(required_cache_files=len(request()['required_cache']),maximum_copy_bytes=MAX_BYTES,maximum_CPU_seconds=CPU_SECONDS,prior_CPU_seconds_upper=PRIOR_CPU_SECONDS_UPPER,combined_CPU_seconds_upper=CPU_AGGREGATE_SECONDS_UPPER,new_GPU_jobs=0,authorized=AUTH.exists())))
    elif a.quote:
        ROOT.mkdir(parents=True,exist_ok=True);print(json.dumps(r.quote()))
    elif a.guard:authority();r.guard()
    elif a.inventory:inventory(a.request,a.output)
    elif a.probe_models:authority();print(json.dumps(process_probe()))
    else:execute()
