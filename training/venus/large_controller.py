"""One value-selected A10040 Flex interval after completed CPU staging.

Native preparation, independent restore, complete-epoch main/control and full
paired collection run serially. Every failure stops compute and preserves disk;
this controller never retries an allocation or repeats a saved optimizer step.
"""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from . import controller as c
from .large_guard import AUTH, MEMORY_AUTH, PREFIX_AUTH, LAYOUT_AUTH, AGGREGATE_AUTH, REPORTING_REVIEW, REPORTING_REVIEW_SHA, ROOT, KEY, OLD_USD, authority, memory_authority, prefix_authority, layout_authority, aggregate_authority
from .large_inputs import PARENT_SHA, MANIFEST_SHA, sha
from .large_stage import NAME as CPU, DISK, ZONE, REMOTE

R=ROOT/'gpu-01'
NAME='duplex-venus-large-gpu-20261104-a-v65'
OUTPUT=REMOTE+'/native-v65'
BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v65')


def save(name,value):(R/name).write_text(json.dumps(value,indent=2)+'\n')


def observe(op):
    until=time.time()+1020
    while op['status']!='DONE':
        if time.time()>until:raise TimeoutError('Observe SAME queued provider operation; never create again')
        time.sleep(5);op=c.cloud('compute','operations','describe',op['name'],'--zone='+ZONE)
        save('latest-operation.json',op)
    return op


def copyback():
    # Hash only this new native run, never rehash the old81GB archive.
    script="""import hashlib,json,os
from pathlib import Path
r=Path(%r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
print(json.dumps(dict(files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()],links=[dict(path=str(p.relative_to(r)),target=os.readlink(p)) for p in sorted(r.rglob('*')) if p.is_symlink()])))
"""%OUTPUT
    inventory=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(script),600));save('remote-inventory.json',inventory)
    if sum(x['bytes'] for x in inventory['files'])>100_000_000_000:raise ValueError('Approved100GB artifact limit exceeded; preserve disk')
    if BACKUP.exists():
        binding=json.loads((R/'stream-backup-binding.json').read_text())
        if (binding['backup']!=str(BACKUP) or binding['remote_output']!=OUTPUT
                or binding['source']!=json.loads((R/'claim.json').read_text())['source']):
            raise ValueError('Only this exact source/output streamed backup may be completed')
    else:BACKUP.mkdir(parents=True,exist_ok=False)
    argv=['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private',
          '--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*',
          '--partial-dir=.rsync-partial','-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+OUTPUT+'/',str(BACKUP)+'/']
    c.run(argv,1800)
    for row in inventory['files']:
        path=BACKUP/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Native result backup hash differs')
    for row in inventory['links']:
        path=BACKUP/row['path']
        if not path.is_symlink() or os.readlink(path)!=row['target']:raise ValueError('Native result link differs')
    save('copyback-verified.json',dict(all_hashes_verified=True,files=len(inventory['files']),
         bytes=sum(x['bytes'] for x in inventory['files']),links=len(inventory['links']),
         backup=str(BACKUP),inventory_sha256=sha(R/'remote-inventory.json')))


def capacity_retry_after(number,previous,family="prefix",aggregate_interval=7):
    """Only terminal pre-start stockout on SAME VM; no new running interval."""
    if number not in (1,2):raise ValueError('At most two spaced capacity-only retries')
    if family not in ('prefix','layout','aggregate'):raise ValueError('Exact authorized capacity family required')
    if aggregate_interval not in (7,8,9,10,11):raise ValueError('Exact supported aggregate interval required')
    base=('gpu-%02d'%aggregate_interval) if family=='aggregate' else 'gpu-06' if family=='layout' else 'gpu-05'
    old=ROOT/(base if number==1 else base+'-capacity-01')
    vm=json.loads((old/'no-new-running-interval.json').read_text())
    op=json.loads((old/'allocation-terminal.json').read_text())
    errors=op.get('error',{}).get('errors',[])
    if (op['status']!='DONE' or not errors
            or any(x['code'] not in ('ZONE_RESOURCE_POOL_EXHAUSTED','ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS') for x in errors)
            or vm['status']!='TERMINATED' or vm['id']!=previous['id']
            or vm['lastStartTimestamp']!=previous['lastStartTimestamp']):
        raise ValueError('Only exact terminal capacity failure with ZERO new start may retry')
    # Google advances lastStop on a failed start without advancing lastStart.
    # Preserve the original actual-running receipt; do not charge phantom time.
    if vm['lastStopTimestamp']!=previous['lastStopTimestamp']:
        stop=datetime.fromisoformat(vm['lastStopTimestamp'])
        if not datetime.fromisoformat(op['startTime'])<=stop<=datetime.fromisoformat(op['endTime']):
            raise ValueError('Updated stop must belong to this exact failed-start operation')
    requests=[v for v in json.loads((old/'events.json').read_text()) if v['event']=='one_value_selected_GPU_request']
    if len(requests)!=1:raise ValueError('One preserved actual request required')
    return datetime.fromisoformat(requests[0]['at_utc']).timestamp()+600


def main(resume_sizing=False,replace_prestart=False,fallback80=False,resume80_precreate=False,resume_memory=False,resume_prefix=False,prefix_capacity_retry=0,resume_layout=False,layout_capacity_retry=0,resume_aggregate=False,aggregate_capacity_retry=0,aggregate_interval=7,sustained_first=False,sustained_resume8=False):
    global R,OUTPUT,BACKUP,NAME,ZONE,DISK
    from .source_controller import prepare_ssh
    from portable.venus.guard import bounded_runpod_read
    authority()
    if sustained_first and sustained_resume8:raise ValueError('One sustained mode required')
    sustained_run=sustained_first or sustained_resume8
    resume_evidence_path=ROOT/'sustained-saved-state-0008-evidence.json'
    if sustained_run:
        from .sustained import authority as sustained_authority
        sustained_authority();resume_aggregate=True;aggregate_interval=11 if sustained_resume8 else 10
    if sustained_resume8:
        saved_evidence=json.loads(resume_evidence_path.read_text())
        if saved_evidence['vm_id']!='7521667153519853425' or saved_evidence['disk_id']!='2297341858915206463' or saved_evidence['cache_manifest_sha256']!=MANIFEST_SHA or saved_evidence['remote_checkpoint']!=REMOTE+'/recovery-v75/main-step-0008.pt':raise ValueError('Exact saved checkpoint path/resource/data required')
        if saved_evidence['actual_new_updates']!=8 or saved_evidence['checkpoint_sha256']!='82c1d407a236d823b772f9e4c6f126e4eaec3bb6fb8f0d8147d9260c539b1cd3' or saved_evidence['source']!='b39ce3ebe167b059b67fa3d131978e5da7c41ac1':raise ValueError('Actual saved8 evidence required')
    if aggregate_interval>=10 and not sustained_run:raise ValueError('Explicit sustained mode required')
    if aggregate_interval not in (7,8,9,10,11) or aggregate_interval!=7 and not resume_aggregate:
        raise ValueError("Only actual initial, throughput follow-up or exact reporting recovery supported")
    if resume_aggregate:
        aggregate_authority()
        if any((resume_sizing,replace_prestart,fallback80,resume80_precreate,resume_memory,resume_prefix,prefix_capacity_retry,resume_layout,layout_capacity_retry)):
            raise ValueError("Aggregate continuation is one independent same-resource mode")
        resume_layout=True
    if aggregate_capacity_retry and not resume_aggregate:raise ValueError("Aggregate capacity retry requires aggregate mode")
    if resume_layout:
        layout_authority()
        if any((resume_sizing,replace_prestart,fallback80,resume80_precreate,resume_memory,resume_prefix,prefix_capacity_retry)):
            raise ValueError("Layout restart is one independent same-resource mode")
        resume_prefix=True
    if layout_capacity_retry and not resume_layout:raise ValueError("Layout capacity retry requires layout mode")
    if prefix_capacity_retry and not resume_prefix:raise ValueError('Capacity-only retry requires shared-history mode')
    if resume_prefix:
        prefix_authority()
        if any((resume_sizing,replace_prestart,fallback80,resume80_precreate,resume_memory)):
            raise ValueError("Shared-history restart is one independent same-resource mode")
        resume_memory=True
    if resume_memory:
        memory_authority()
        if any((resume_sizing,replace_prestart,fallback80,resume80_precreate)):
            raise ValueError("Memory restart is one independent same-resource mode")
        fallback80=True
    if resume80_precreate and not fallback80:raise ValueError('80GB precreate recovery requires the original fallback mode')
    if fallback80 and (resume_sizing or replace_prestart):raise ValueError('Choose one exact allocation mode')
    if replace_prestart and not resume_sizing:raise ValueError('Replacement is scoped only to actual pretraining sizing failure')
    previous_intervals=[];prior_seconds=0;prior_vm=None
    if resume_sizing:
        old=ROOT/'gpu-01'
        prior_vm=json.loads((old/'provider-final.json').read_text())
        verified=json.loads((old/'copyback-verified.json').read_text())
        if (prior_vm['status']!='TERMINATED' or not verified['all_hashes_verified']
                or not (Path(verified['backup'])/'no-offload-oom.json').exists()
                or (Path(verified['backup'])/'admission.json').exists()):
            raise ValueError('Exact preserved zero-update sizing failure required')
        prior_seconds=math.ceil((datetime.fromisoformat(prior_vm['lastStopTimestamp'])-datetime.fromisoformat(prior_vm['lastStartTimestamp'])).total_seconds()/100)*100
        if prior_seconds!=300:raise ValueError('Carry the actual300second first-interval upper bound')
        receipt=ROOT/'provider-final-01.json'
        if not receipt.exists():receipt.write_bytes((old/'provider-final.json').read_bytes())
        if sha(receipt)!=sha(old/'provider-final.json'):raise ValueError('Original provider receipt differs')
        previous_intervals=[dict(provider_final_path=str(receipt),sha256=sha(receipt),vm_id=prior_vm['id'],
             disk_id=json.loads((old/'lease.json').read_text())['disk_id'],upper_seconds=prior_seconds,optimizer_updates=0)]
        R=ROOT/'gpu-02';OUTPUT=REMOTE+'/native-v66'
        BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v66')
    if fallback80:
        for number in (tuple(range(1,aggregate_interval)) if resume_aggregate else (1,2,3,4,5) if resume_layout else (1,2,3,4) if resume_prefix else (1,2,3) if resume_memory else (1,2)):
            old=ROOT/('gpu-05-capacity-01' if number==5 else 'gpu-%02d'%number)
            terminal=json.loads((old/'provider-final.json').read_text())
            if sustained_resume8 and number==10:
                if terminal['status']!='TERMINATED':raise ValueError('Actual interrupted GPU must be stopped')
                verified=None
            else:verified=json.loads((old/'copyback-verified.json').read_text())
            if not (sustained_resume8 and number==10) and (terminal['status']!='TERMINATED' or not verified['all_hashes_verified'] or (Path(verified['backup'])/'admission.json').exists() and not (sustained_run and number==9)):
                raise ValueError('Both actual zero-update failures must be stopped and hash-backed-up')
            seconds=math.ceil((datetime.fromisoformat(terminal['lastStopTimestamp'])-datetime.fromisoformat(terminal['lastStartTimestamp'])).total_seconds()/100)*100
            receipt=ROOT/('provider-final-%02d.json'%number)
            if not receipt.exists():receipt.write_bytes((old/'provider-final.json').read_bytes())
            if sha(receipt)!=sha(old/'provider-final.json'):raise ValueError('Actual prior receipt differs')
            previous_intervals.append(dict(provider_final_path=str(receipt),sha256=sha(receipt),vm_id=terminal['id'],
                disk_id=json.loads((old/'lease.json').read_text())['disk_id'],upper_seconds=seconds,optimizer_updates=8 if sustained_resume8 and number==10 else 0))
            prior_seconds+=seconds;prior_vm=terminal
        if prior_seconds!=(({7:2700,8:4000,9:4300,10:8700,11:12800}[aggregate_interval]) if resume_aggregate else 2600 if resume_layout else 2100 if resume_prefix else 1600 if resume_memory else 700):raise ValueError('Carry ALL actual prior interval upper bounds')
        R=ROOT/'gpu-03';OUTPUT=REMOTE+'/native-v67'
        BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v67')
        ZONE='europe-west4-a';NAME='duplex-venus-large-gpu-20261104-eua-v67'
        DISK='duplex-venus-large-data-20261104-eua-v67'
        if resume_memory:
            R=ROOT/'gpu-04';OUTPUT=REMOTE+'/native-v68'
            BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v68')
        if resume_prefix:
            R=ROOT/'gpu-05';OUTPUT=REMOTE+'/native-v69'
            BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v69')
        if resume_layout:
            R=ROOT/'gpu-06';OUTPUT=REMOTE+'/native-v70'
            BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v70')
        if resume_aggregate:
            version=75 if sustained_run else 64+aggregate_interval
            R=ROOT/('gpu-%02d'%aggregate_interval);OUTPUT=REMOTE+('/native-v%d'%version)
            BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v%d'%version)
            if sustained_resume8:BACKUP=BACKUP/'resume11'
    retry_after=0
    if prefix_capacity_retry:
        retry_after=capacity_retry_after(prefix_capacity_retry,prior_vm)
        failed=ROOT/('gpu-05' if prefix_capacity_retry==1 else 'gpu-05-capacity-01')
        prior_vm=json.loads((failed/'no-new-running-interval.json').read_text())
        R=ROOT/('gpu-05-capacity-%02d'%prefix_capacity_retry)
    if layout_capacity_retry:
        retry_after=capacity_retry_after(layout_capacity_retry,prior_vm,'layout')
        failed=ROOT/('gpu-06' if layout_capacity_retry==1 else 'gpu-06-capacity-01')
        prior_vm=json.loads((failed/'no-new-running-interval.json').read_text())
        R=ROOT/('gpu-06-capacity-%02d'%layout_capacity_retry)
    if aggregate_capacity_retry:
        retry_after=capacity_retry_after(aggregate_capacity_retry,prior_vm,'aggregate',aggregate_interval)
        failed=ROOT/(('gpu-%02d'%aggregate_interval) if aggregate_capacity_retry==1 else 'gpu-%02d-capacity-01'%aggregate_interval)
        prior_vm=json.loads((failed/'no-new-running-interval.json').read_text())
        R=ROOT/('gpu-%02d-capacity-%02d'%(aggregate_interval,aggregate_capacity_retry))
    if replace_prestart:
        rejected=json.loads((R/'no-new-running-interval.json').read_text())
        if rejected['id']!=prior_vm['id'] or rejected['lastStartTimestamp']!=prior_vm['lastStartTimestamp'] or rejected['status']!='TERMINATED':
            raise ValueError('Exact zero-new-interval scheduling rejection required')
        NAME='duplex-venus-large-gpu-20261104-a-v66'
    if R.exists() and not replace_prestart and not resume80_precreate:raise ValueError('Observe the real existing GPU controller; never duplicate')
    stage=ROOT/'stage'
    proof=json.loads((stage/'stage-verified.json').read_text())
    previous=json.loads((stage/'provider-final.json').read_text())
    if not proof['passed'] or previous['status']!='TERMINATED':raise ValueError('Complete CPU stage and actual STOP required')
    clock=json.loads((stage/'storage-deadline.json').read_text())
    duration=(73500 if sustained_run else 43100)-prior_seconds
    if sustained_run:
        clock=dict(clock,storage_deadline_unix=1794052800.)
    if time.time()+duration+1800>=clock['storage_deadline_unix']:raise ValueError('Original storage clock insufficient')
    price_path=Path('runs/product-progress-watch/current-capacity-prices-20261105-launch-v76.json' if sustained_resume8 else 'runs/product-progress-watch/current-capacity-prices-20261105-launch-v75.json') if sustained_run else Path('runs/product-progress-watch/current-capacity-prices-20261104-launch-'+('v72' if resume_aggregate and aggregate_interval>=8 else 'v71' if resume_aggregate else 'v70' if resume_layout else 'v69' if resume_prefix else 'v68' if resume_memory else 'v65')+'.json')
    price=json.loads(price_path.read_text())
    hourly=price['options']['GCP_A10080_STANDARD_Netherlands_12CPU_170GiB_375SSD_250PD' if fallback80 else 'GCP_A10040_FLEX_US_12CPU_85GiB_250PD']
    if not 0<hourly<=(5.9 if sustained_run else 8) or time.time()-datetime.fromisoformat(price['at_utc']).timestamp()>3600:
        raise ValueError('Refresh actual full GPU component price before allocation')
    source=c.run(['git','rev-parse','HEAD']).strip()
    if c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','portable/venus']).strip():
        raise ValueError('Source must be Git committed before GPU deployment')
    instances=c.cloud('compute','instances','list')
    if any(v['status'] not in ('TERMINATED','STOPPED') for v in instances):raise ValueError('All compute must actually be stopped')
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip())
    if isinstance(pods,dict):pods=pods['pods']
    if any(v['status'] not in ('EXITED','STOPPED','TERMINATED') for v in pods):raise ValueError('No overlapping Runpod GPU')
    quotas={v['metric']:v for v in c.cloud('compute','regions','describe','europe-west4' if fallback80 else 'us-central1')['quotas']}
    q=quotas['NVIDIA_A100_80GB_GPUS' if fallback80 else 'PREEMPTIBLE_NVIDIA_A100_GPUS']
    if q['usage']!=0 or q['limit']<1:raise ValueError('Actual available FlexA100 quota required; quota is not stock')
    rules={v['name']:v for v in c.cloud('compute','firewall-rules','list')}
    allow=rules['duplex-venus-train-iap-v1'];deny=rules['duplex-venus-train-deny-v1']
    if (allow['sourceRanges']!=['35.235.240.0/20'] or allow['allowed']!=[dict(IPProtocol='tcp',ports=['22'])]
            or deny['denied']!=[dict(IPProtocol='all')] or deny['sourceRanges']!=['0.0.0.0/0']
            or 'duplex-venus-train-bounded' not in allow['targetTags'] or 'duplex-venus-train-bounded' not in deny['targetTags']
            or allow['priority']>=deny['priority']):raise ValueError('Existing authenticated IAP/restricted-ingress fence required')
    old_disk_name='duplex-venus-large-data-20261104-a-v65'
    disk=c.cloud('compute','disks','describe',DISK if (resume80_precreate or resume_memory) else old_disk_name if fallback80 else DISK,
                 '--zone='+('us-central1-a' if fallback80 and not (resume80_precreate or resume_memory) else ZONE))
    olddisk=json.loads((stage/'disk.json').read_text())
    expected_disk=json.loads(((ROOT/'gpu-03' if resume_memory else R)/'disk-a.json').read_text()) if (resume80_precreate or resume_memory) else olddisk
    if disk['id']!=expected_disk['id'] or int(disk['sizeGb'])!=250:raise ValueError('Exact preserved250GB disk required')
    if resume80_precreate:
        failure=json.loads((R/'eu80-precreate-failure.json').read_text())
        if (failure['third_GPU_interval_started'] or failure['reason']!='Google requires explicit discard-local-ssds-at-termination-timestamp'
                or disk.get('users') or disk['status']!='READY' or disk['sourceSnapshotId']!='4724490043241393447'
                or (R/'provider-start.json').exists() or any(v['name']==NAME for v in instances)):
            raise ValueError('Only exact observed400-before-VM and same free ready disk may recover')
    if resume_sizing or resume_memory:
        same=[v for v in instances if v['name']==prior_vm['name']]
        if (len(same)!=1 or same[0]['id']!=prior_vm['id'] or same[0]['status']!='TERMINATED'
                or same[0]['lastStartTimestamp']!=prior_vm['lastStartTimestamp']
                or same[0]['lastStopTimestamp']!=prior_vm['lastStopTimestamp']):
            raise ValueError('Only SAME stopped zero-update sizing VM may restart')
        if resume_memory and (prior_vm['id']!='7521667153519853425' or disk['id']!='2297341858915206463'
                or disk.get('users')!=[prior_vm['selfLink']] or disk['sourceSnapshotId']!='4724490043241393447'):
            raise ValueError('Only SAME stopped80VM and attached exact250disk may restart')
        if replace_prestart and any(v['name']==NAME for v in instances):raise ValueError('Replacement name already exists; observe it')
    elif any(v['name']==NAME for v in instances):raise ValueError('Existing GPU name must be observed')
    if resume80_precreate:
        c.events=json.loads((R/'events.json').read_text())
        claim=R/'eu80-precreate-recovery-claim.json';bundle=R/'eu80-precreate-recovery-source.bundle'
    elif replace_prestart:
        c.events=json.loads((R/'events.json').read_text())
        claim=R/'prestart-replacement-claim.json';bundle=R/'prestart-replacement-source.bundle'
    else:
        R.mkdir();c.events=[];claim=R/'claim.json';bundle=R/'source.bundle'
    c.R=R;c.NAME=NAME
    with claim.open('x') as stream:json.dump(dict(source=source,pid=os.getpid(),authority_sha256=sha(AUTH),same_disk_pretraining_replacement=replace_prestart),stream)
    c.run(['git','bundle','create',str(bundle),'HEAD'],120)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);watch=None;created=False;end=None;checkpoint_stream=None
    try:
        if retry_after:
            c.record('capacity_only_spaced_wait',retry=aggregate_capacity_retry or layout_capacity_retry or prefix_capacity_retry,not_before_unix=retry_after,prior_GPU_seconds_upper=prior_seconds)
            while time.time()<retry_after:time.sleep(min(30,retry_after-time.time()))
            current=c.cloud('compute','instances','list')
            same=[v for v in current if v['name']==NAME]
            if (any(v['status'] not in ('TERMINATED','STOPPED') for v in current)
                    or len(same)!=1 or same[0]['id']!=prior_vm['id']
                    or same[0]['lastStartTimestamp']!=prior_vm['lastStartTimestamp']
                    or same[0]['lastStopTimestamp']!=prior_vm['lastStopTimestamp']):
                raise ValueError('Same unchanged stopped provider identity required after spaced wait')
            fresh_pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip())
            if isinstance(fresh_pods,dict):fresh_pods=fresh_pods['pods']
            if any(v['status'] not in ('EXITED','STOPPED','TERMINATED') for v in fresh_pods):raise ValueError('No overlapping GPU after wait')
        if not resume_sizing and not fallback80:
            current=c.cloud('compute','instances','describe',CPU,'--zone='+ZONE)
            if current['id']!=previous['id'] or current['status']!='TERMINATED':raise ValueError('Exact stopped staging CPU required')
            c.run(['gcloud','compute','instances','delete',CPU,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
            if c.cloud('compute','disks','describe',DISK,'--zone='+ZONE).get('users'):raise ValueError('Staged disk still attached')
        boot=R/'bootstrap.sh';boot.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-large65-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        if sustained_run:
            from .oslogin_access import ensure as ensure_key
            ensure_key(time.time()+duration,R)
        else:
            c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=14h')
        if replace_prestart:
            exact=c.cloud('compute','instances','describe',prior_vm['name'],'--zone='+ZONE)
            if exact['id']!=prior_vm['id'] or exact['status']!='TERMINATED':raise ValueError('Only preserved stopped sizing envelope may retire')
            c.run(['gcloud','compute','instances','delete',prior_vm['name'],'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
            if c.cloud('compute','disks','describe',DISK,'--zone='+ZONE).get('users'):raise ValueError('Same staged disk must be free before replacement')
        created=True
        if resume_memory:
            c.cloud('compute','instances','set-scheduling',NAME,'--zone='+ZONE,
                    '--max-run-duration='+str(duration)+'s','--instance-termination-action=STOP',
                    '--discard-local-ssds-at-termination-timestamp=true','--no-restart-on-failure',
                    '--maintenance-policy=TERMINATE')
            op=c.cloud('compute','instances','start',NAME,'--zone='+ZONE,'--async')
        elif fallback80:
            from .large_stage import LOCAL,PARENT
            # No learned state exists on this staging disk. Both unique failed
            # native runs have complete off-machine hash receipts above.
            if not resume80_precreate:
                exact=c.cloud('compute','instances','describe',prior_vm['name'],'--zone=us-central1-a')
                if exact['id']!=prior_vm['id'] or exact['status']!='TERMINATED' or exact['lastStartTimestamp']!=prior_vm['lastStartTimestamp']:
                    raise ValueError('Only exact stopped second sizing instance may retire')
                c.run(['gcloud','compute','instances','delete',prior_vm['name'],'--zone=us-central1-a','--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
                old_disk=c.cloud('compute','disks','describe',old_disk_name,'--zone=us-central1-a')
                if old_disk['id']!=disk['id'] or old_disk.get('users'):raise ValueError('Only old fully preserved temporary disk may retire')
                c.run(['gcloud','compute','disks','delete',old_disk_name,'--zone=us-central1-a','--project='+c.PROJECT,'--quiet'],150)
            snap=c.cloud('compute','snapshots','describe','duplex-venus-context-migrate-20261102-v50')
            if snap['id']!='4724490043241393447' or snap['sourceDiskId']!='4375908109499109167' or snap['status']!='READY':
                raise ValueError('Same approved owned100GB source snapshot required')
            last_submission=0
            for suffix in ('a','b','c'):
                while time.time()<last_submission+600:time.sleep(min(30,last_submission+600-time.time()))
                ZONE='europe-west4-'+suffix;NAME='duplex-venus-large-gpu-20261104-eu'+suffix+'-v67'
                DISK='duplex-venus-large-data-20261104-eu'+suffix+'-v67';c.NAME=NAME
                if not (resume80_precreate and suffix=='a'):
                    created_disk=json.loads(c.run(['gcloud','compute','disks','create',DISK,'--zone='+ZONE,
                        '--source-snapshot='+snap['name'],'--size=250GB','--type=pd-balanced','--labels=duplex-scope=large-v65',
                        '--project='+c.PROJECT,'--quiet','--format=json'],900));save('disk-create-'+suffix+'.json',created_disk)
                disk=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE)
                if disk['sourceSnapshotId']!=snap['id'] or int(disk['sizeGb'])!=250 or disk.get('users'):raise ValueError('Exact new250GB source clone required')
                save('disk-'+suffix+'.json',disk);last_submission=time.time()
                op=c.cloud('compute','instances','create',NAME,'--zone='+ZONE,'--machine-type=a2-ultragpu-1g',
                    '--provisioning-model=STANDARD','--maintenance-policy=TERMINATE',
                    '--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded',
                    '--disk=name='+DISK+',boot=yes,auto-delete=no,mode=rw',
                    '--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE',
                    '--metadata-from-file=startup-script='+str(boot),'--max-run-duration='+str(duration)+'s',
                    '--discard-local-ssds-at-termination-timestamp=true',
                    '--instance-termination-action=STOP','--async')
                op=op[0] if isinstance(op,list) else op;save('create-'+suffix+'.json',op)
                c.record('one_A10080_standard_prestart_attempt',zone=ZONE,operation=op['name'],prior_GPU_seconds_upper=prior_seconds)
                op=observe(op);save('allocation-terminal-'+suffix+'.json',op)
                if not op.get('error'):break
                existing=[v for v in c.cloud('compute','instances','list') if v['name']==NAME]
                if ('ZONE_RESOURCE_POOL_EXHAUSTED' not in json.dumps(op['error']) or any(v.get('lastStartTimestamp') for v in existing)):
                    raise RuntimeError('No fallback after partial start/access/unknown error: '+json.dumps(op['error']))
                if existing:c.run(['gcloud','compute','instances','delete',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
                c.run(['gcloud','compute','disks','delete',DISK,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet'],150)
            else:raise RuntimeError('All three approved Europe zones failed before starting; no GPU interval3 consumed')
        elif resume_sizing and not replace_prestart:
            c.cloud('compute','instances','set-scheduling',NAME,'--zone='+ZONE,'--max-run-duration='+str(duration)+'s',
                    '--instance-termination-action=STOP','--no-restart-on-failure','--maintenance-policy=TERMINATE')
            op=c.cloud('compute','instances','start',NAME,'--zone='+ZONE,'--async')
        else:op=c.cloud('compute','instances','create',NAME,'--zone='+ZONE,'--machine-type=a2-highgpu-1g',
             '--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE',
             '--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded',
             '--disk=name='+DISK+',boot=yes,auto-delete=no,mode=rw',
             '--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE',
             '--metadata-from-file=startup-script='+str(boot),'--max-run-duration='+str(duration)+'s',
             '--instance-termination-action=STOP','--async')
        op=op[0] if isinstance(op,list) else op;save('create.json',op)
        c.record('one_value_selected_GPU_request',source=source,operation=op['name'],maximum_queue_seconds=900,
                 same_VM_sizing_recovery=resume_sizing and not replace_prestart,
                 same_80VM_lossless_memory_restart=resume_memory,
                 same_disk_pretraining_replacement=replace_prestart,prior_GPU_seconds_upper=prior_seconds)
        op=observe(op);save('allocation-terminal.json',op)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-start.json',vm)
        if vm['status']!='RUNNING':raise ValueError('Actual running provider state required')
        if (resume_sizing or fallback80 or resume_memory) and vm['lastStartTimestamp']==prior_vm['lastStartTimestamp']:
            raise ValueError('Do not claim an old provider interval as a new start')
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+duration
        if sustained_run:c.ssh_repair=lambda:ensure_key(end,R,force=True)
        prepare_ssh(vm,ZONE,'66gpu' if resume_sizing else '65gpu')
        c.remote=c.ROOT+'/duplex-large-'+source[:12]
        c.sync(bundle,c.ROOT+'/large65-gpu-source.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/large65-gpu-source.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+source,120)
        plan=dict(resource_key=KEY,authorization_sha256=sha(AUTH),parent_sha256=PARENT_SHA,cache_manifest_sha256=MANIFEST_SHA,
             maximum_all_branch_updates=120 if sustained_run else 984,external_model_API_calls=0,prior_project_USD=OLD_USD,
             vm_id=vm['id'],disk_id=disk['id'],zone=ZONE,creation_timestamp=vm['creationTimestamp'],
             **clock,previous_intervals=previous_intervals,maximum_running_seconds=duration,disk_size_gib=250,
             machine='a2-ultragpu-1g' if fallback80 else 'a2-highgpu-1g',provisioning='STANDARD' if fallback80 else 'FLEX_START',optimizer_work_started=False,vm_name=NAME,disk_name=DISK)
        if sustained_run:
            from .sustained import AUTH as S_AUTH
            plan['sustained_authorization_sha256']=sha(S_AUTH)
        plan['execution_source']=source
        if sustained_resume8:plan.update(optimizer_work_started=True,trained_state_receipt_path=str(resume_evidence_path),trained_state_receipt_sha256=sha(resume_evidence_path),training_source=saved_evidence['source'])
        if resume_memory:plan['memory_authorization_sha256']=sha(MEMORY_AUTH)
        if resume_prefix:plan.update(prefix_authorization_sha256=sha(PREFIX_AUTH),activation_memory_policy='repeat4-prefix-pageable-v2')
        if resume_layout:plan.update(layout_authorization_sha256=sha(LAYOUT_AUTH),activation_memory_policy='repeat4-prefix-pageable-v3-layout')
        if resume_aggregate:plan['aggregate_authorization_sha256']=sha(AGGREGATE_AUTH)
        if resume_aggregate and aggregate_interval==9:
            if sha(REPORTING_REVIEW)!=REPORTING_REVIEW_SHA:raise ValueError('Exact reviewed reporting fix required')
            plan['reporting_recovery_review_sha256']=REPORTING_REVIEW_SHA
        plan_path=ROOT/(('interval-%02d.json'%aggregate_interval) if resume_aggregate else 'interval-06.json' if resume_layout else 'interval-05.json' if resume_prefix else 'interval-04.json' if resume_memory else 'interval-03.json' if fallback80 else 'interval-02.json' if resume_sizing else 'interval-01.json')
        with plan_path.open('x') as f:json.dump(plan,f,indent=2)
        lease=dict(plan,execution_plan_path=str(plan_path),execution_plan_sha256=sha(plan_path),
             prior_capacity_GPU_seconds_upper=prior_seconds,verified_hourly_price_usd=hourly,stop_unix=end,
             prior_gpu_seconds=89151+prior_seconds,prior_scope_usd=OLD_USD,controller_pid=os.getpid(),
             controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        save('lease.json',lease)
        c.ssh('mkdir -p '+c.remote+'/'+str(ROOT))
        if sustained_resume8:c.sync(resume_evidence_path,c.remote+'/'+str(resume_evidence_path))
        for row in previous_intervals:c.sync(row['provider_final_path'],c.remote+'/'+row['provider_final_path'])
        c.sync(plan_path,c.remote+'/'+str(plan_path));lease_path=c.ROOT+'/large65-lease.json';c.sync(R/'lease.json',lease_path)
        watch=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),
             '--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key',
             '--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT)
        c.guard=watch
        until=time.time()+100
        while not (R/'guard.json').exists():
            if watch.poll() is not None or time.time()>until:raise RuntimeError('Independent GPU guard failed to arm')
            time.sleep(1)
        c.ssh('sudo systemd-run --unit=duplex-large65-guest --property=RuntimeMaxSec='+str(duration)+' --working-directory='+c.remote+
              ' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lease_path)
        if c.props('duplex-large65-guest')['ActiveState']!='active':raise RuntimeError('Guest GPU guard absent')
        c.ssh('sudo systemctl stop duplex-large65-bootstrap-stop.timer')
        if fallback80 and not resume_memory:
            c.ssh('mkdir -p '+REMOTE)
            c.sync(str(LOCAL/'cache')+'/',REMOTE+'/cache/',900);c.sync(PARENT,REMOTE+'/parent466.pt',180)
        if c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip():raise ValueError('GPU already has a model process')
        c.record('actual_GPU_running_both_guards_live',vm_id=vm['id'],disk_id=disk['id'],source=source,native_STOP_unix=end)
        env='env DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lease_path+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache '
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        model_source=saved_evidence['source'] if sustained_resume8 else source
        common=' --model '+c.MODEL+' --cache '+REMOTE+'/cache --action-index '+REMOTE+'/cache/action-index.json --data-admission '+c.remote+'/docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json --opal-source '+c.ROOT+'/opal-upstream-v34 --lease '+lease_path+' --source '+model_source
        def execute(unit,module,args,seconds):
            if seconds<1:raise TimeoutError('No authorized running allowance remains')
            c.start_unit(unit,env+py+' -u -m '+module+args,seconds);c.wait_unit(unit,min(time.time()+seconds+15,end-180))
        sizing_limit=2700 if resume_aggregate and aggregate_interval>=8 else 1200
        sizing_remaining=sizing_limit if fallback80 else sizing_limit-prior_seconds-int(time.time()-datetime.fromisoformat(vm['lastStartTimestamp']).timestamp())-100
        if sustained_first:
            execute('duplex-large75-reuse','training.venus.sustained',' --output '+OUTPUT+' --source '+source,300)
        elif sustained_resume8:
            execute('duplex-large76-resume-proof','training.venus.resume_native',common+' --root '+OUTPUT+' --checkpoint '+saved_evidence['remote_checkpoint']+' --evidence '+c.remote+'/'+str(resume_evidence_path),900)
        else:
            execute('duplex-large65-prepare','training.venus.large_native',common+(' --measured-joint-schedule' if resume_aggregate and aggregate_interval>=8 else '')+' --stage prepare --output '+OUTPUT+' --parent '+REMOTE+'/parent466.pt'+(' --activation-offload --host-offload-limit-gib 48 --activation-memory-policy repeat4-prefix-pageable-v3-layout' if resume_layout else ' --activation-offload --host-offload-limit-gib 48 --activation-memory-policy repeat4-prefix-pageable-v2' if resume_prefix else ' --activation-offload --host-offload-limit-gib 48 --activation-memory-policy repeat4-pageable-v1' if resume_memory else ' --activation-offload --host-offload-limit-gib 128' if fallback80 else ' --activation-offload' if resume_sizing else ''),min(sizing_limit,sizing_remaining))
        if not sustained_resume8:execute('duplex-large65-restore','training.venus.large_native',common+' --stage restore --output '+OUTPUT,900)
        if resume_aggregate:
            from .checkpoint_stream import CheckpointStream
            checkpoint_stream=CheckpointStream(c.ssh_argv,OUTPUT,BACKUP,R,source).start()
        for arm in ('main','control'):
            execute('duplex-large65-train-'+arm,'training.venus.large_train',common+' --arm '+arm+' --root '+OUTPUT+' --parent '+REMOTE+'/parent466.pt'+(' --resume '+saved_evidence['remote_checkpoint'] if sustained_resume8 and arm=='main' else ''),int(end-time.time()-1800))
            execute('duplex-large65-final-'+arm,'training.venus.large_native',common+' --stage final-restore --output '+OUTPUT+'/'+arm,900)
        c.record('both_complete_training_arms_and_independent_final_restores',source=source,adopted=False)
        evaluate=' --root '+OUTPUT+' --model '+c.MODEL+' --lease '+lease_path+' --source '+model_source+' --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --word-quality '+c.OUT+'/paced-word-quality-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674'
        for arm in ('main','control'):
            execute('duplex-large65-native-'+arm,'training.venus.large_behavior',evaluate+' --arm '+arm,int(end-time.time()-600))
        execute('duplex-large65-combine','training.venus.large_behavior',' --combine --root '+OUTPUT+' --source '+model_source,120)
        c.record('complete64_native_speech_collection',source=source,adopted=False)
    except BaseException as error:
        c.record('capacity_interval_failure',error_type=type(error).__name__,error=str(error)[-1500:]);raise
    finally:
        if c.ssh_argv:
            try:
                c.ssh('mkdir -p '+OUTPUT+'/journals')
                for unit in c.UNITS:
                    state=c.props(unit)
                    if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit)
                    c.ssh('sudo journalctl -u '+unit+' --no-pager -o cat > '+OUTPUT+'/journals/'+unit+'.log')
                if checkpoint_stream:checkpoint_stream.stop()
                copyback();c.record('all_native_outputs_offmachine_hash_verified',backup=str(BACKUP))
            except Exception as error:c.record('copyback_incomplete_unique_disk_preserved',error_type=type(error).__name__,error=str(error)[-1000:])
        if created:
            actual=[v for v in c.cloud('compute','instances','list') if v['name']==NAME]
            if actual:
                no_new_start=(resume_sizing or resume_memory) and actual[0].get('lastStartTimestamp')==prior_vm['lastStartTimestamp']
                if no_new_start and actual[0]['status']=='TERMINATED':
                    save('no-new-running-interval.json',actual[0])
                elif (not actual[0].get('lastStartTimestamp') or no_new_start) and not resume_memory:
                    save('cancelled-before-start.json',actual[0])
                    c.run(['gcloud','compute','instances','delete',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
                elif actual[0]['status']!='TERMINATED':
                    c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet']+(['--discard-local-ssd=True'] if fallback80 else []),150)
                if actual[0].get('lastStartTimestamp') and not no_new_start:
                    final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-final.json',final)
                    elapsed=(datetime.fromisoformat(final['lastStopTimestamp'])-datetime.fromisoformat(final['lastStartTimestamp'])).total_seconds()
                    c.record('actual_GPU_stopped',vm_id=final['id'],actual_seconds=elapsed,upper_seconds=math.ceil(elapsed/100)*100)
        if watch:
            (R/'stop-requested').write_text('closed\n')
            try:watch.wait(timeout=35)
            except subprocess.TimeoutExpired:watch.terminate()
        awake.terminate()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume-sizing',action='store_true');p.add_argument('--replace-prestart',action='store_true');p.add_argument('--fallback80',action='store_true');p.add_argument('--resume80-precreate',action='store_true');p.add_argument('--resume-memory',action='store_true');p.add_argument('--resume-prefix',action='store_true');p.add_argument('--prefix-capacity-retry',type=int,choices=(1,2),default=0);p.add_argument('--resume-layout',action='store_true');p.add_argument('--layout-capacity-retry',type=int,choices=(1,2),default=0);p.add_argument('--resume-aggregate',action='store_true');p.add_argument('--aggregate-capacity-retry',type=int,choices=(1,2),default=0);p.add_argument('--aggregate-interval',type=int,choices=(7,8,9),default=7);p.add_argument('--sustained-first',action='store_true');p.add_argument('--sustained-resume8',action='store_true');a=p.parse_args();main(a.resume_sizing,a.replace_prestart,a.fallback80,a.resume80_precreate,a.resume_memory,a.resume_prefix,a.prefix_capacity_retry,a.resume_layout,a.layout_capacity_retry,a.resume_aggregate,a.aggregate_capacity_retry,a.aggregate_interval,a.sustained_first,a.sustained_resume8)
