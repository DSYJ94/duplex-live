"""One queued A100 envelope; never resubmit on timeout or partial startup."""
import json,time,math
from pathlib import Path
from datetime import datetime
from . import controller as c
from . import tts_placement as placement
from .tts_completion_flex_guard import AUTH,RECOVERY_AUTH,COMPAT_AUTH,sha,read_scope

NAME='duplex-venus-tts-flex-20261104-a-v62'
DISK='duplex-venus-tts-flex-data-20261104-a-v62'
ZONE='us-central1-a'


def configuration(version):
    assert version in ('62','63','64')
    authority=COMPAT_AUTH if version=='64' else RECOVERY_AUTH if version=='63' else AUTH
    return (NAME.rsplit('-v',1)[0]+'-v'+version,DISK.rsplit('-v',1)[0]+'-v'+version,authority,1000 if version=='63' else 1200)


def retire(vm,disk,zone,snap,version='62'):
    name,disk_name,_,_=configuration(version)
    assert zone==ZONE and disk['name']==disk_name and disk['id']!='4375908109499109167'
    if vm:
        actual=c.cloud('compute','instances','describe',name,'--zone='+ZONE)
        assert actual['id']==vm['id'] and actual['name']==name
        assert actual['disks'][0]['source'].endswith('/disks/'+disk_name)
        if actual.get('lastStartTimestamp'):
            if actual['status']!='TERMINATED':
                c.run(['gcloud','compute','instances','stop',name,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
            final=c.cloud('compute','instances','describe',name,'--zone='+ZONE)
            assert final['status']=='TERMINATED'
            seconds=datetime.fromisoformat(final['lastStopTimestamp']).timestamp()-datetime.fromisoformat(final['lastStartTimestamp']).timestamp()
            if not any(e['event']=='tts_stop_gpu_terminal' for e in c.events):
                c.record('flex_envelope_terminal',vm_id=vm['id'],actual_seconds=seconds,upper_seconds=math.ceil(seconds/100)*100)
        # Deleting the exact still-pending envelope cancels its queue request.
        # A timed-out observation is never used as authority to create again.
        c.run(['gcloud','compute','instances','delete',name,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
    actual=c.cloud('compute','disks','describe',disk_name,'--zone='+ZONE)
    assert actual['id']==disk['id'] and actual['sourceSnapshotId']==snap['id'] and not actual.get('users')
    c.run(['gcloud','compute','disks','delete',disk_name,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet'],150)
    c.record('flex_temporary_envelope_retired',vm_id=vm['id'] if vm else None,disk_id=disk['id'],original_disk_preserved=True)


def allocate(version='62'):
    name,disk_name,authority,duration=configuration(version)
    a=json.loads(authority.read_text());p,_=read_scope(a,int(version))
    assert p['compute']['VM_name']==name and p['storage']['disk_name']==disk_name
    assert p['compute']['queue_wait_seconds_max']==3600 and p['storage']['temporary_window_seconds']==7200
    assert (p['storage']['combined_upper_each'] if version in ('62','64') else max(p['storage']['prior_conservative_clone_GiBh']+p['storage']['new_clone_GiBh_upper'],p['storage']['prior_conservative_active_snapshot_GiBh']+p['storage']['new_active_snapshot_GiBh_upper']))<=400
    assert all(v['status']=='TERMINATED' for v in c.cloud('compute','instances','list'))
    assert not any(v['name'].startswith(('duplex-venus-tts-data-','duplex-venus-tts-flex-data-')) for v in c.cloud('compute','disks','list'))
    if (c.R/'storage-deadline.json').exists():
        assert version=='63'
        proof=json.loads((c.R/'price-preflight-resume.json').read_text())
        assert sha(c.R/'storage-deadline.json')==proof['storage_deadline_sha256']
        deadline=json.loads((c.R/'storage-deadline.json').read_text())['storage_deadline_unix']
        assert deadline==proof['storage_deadline_unix'] and time.time()+4900<deadline
    else:
        deadline=time.time()+7200
        with (c.R/'storage-deadline.json').open('x') as f:json.dump(dict(storage_deadline_unix=deadline),f)
    snap=None;disk=None
    try:
        snap=placement.snapshot()
        assert time.time()+3600+duration+300<deadline,'Full queue, GPU and copyback time required before creating a clone'
        c.record('one_flex_clone_request',name=disk_name,zone=ZONE)
        c.run(['gcloud','compute','disks','create',disk_name,'--zone='+ZONE,'--source-snapshot='+snap['name'],'--size=100GB','--type=pd-balanced','--labels=duplex-scope=tts-v'+version,'--project='+c.PROJECT,'--quiet','--format=json'],900)
        disk=c.cloud('compute','disks','describe',disk_name,'--zone='+ZONE);placement.save('disk.json',disk)
        assert disk['sourceSnapshotId']==snap['id'] and int(disk['sizeGb'])==100 and not disk.get('users')
        assert time.time()+3600+duration+300<deadline,'Remaining window must cover whole queue and evaluation'
        c.NAME=name
        op=c.cloud('compute','instances','create',name,'--zone='+ZONE,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=3600s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+disk_name+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(c.R/'bootstrap.sh'),'--max-run-duration='+str(duration)+'s','--instance-termination-action=STOP','--async')
        op=op[0] if isinstance(op,list) else op;placement.save('create.json',op)
        c.record('one_queued_flex_request',operation=op['name'],queue_seconds=3600,storage_deadline_unix=deadline)
        until=min(time.time()+3660,deadline-duration-300)
        while op['status']!='DONE':
            assert time.time()<until,'Observe or cancel this exact envelope; no resubmission after timeout'
            time.sleep(10);op=c.cloud('compute','operations','describe',op['name'],'--zone='+ZONE);placement.save('latest-operation.json',op)
        placement.save('terminal-operation.json',op)
        assert not op.get('error'),op.get('error')
        vm=c.cloud('compute','instances','describe',name,'--zone='+ZONE)
        assert vm['status']=='RUNNING' and vm.get('lastStartTimestamp')
        placement.save('allocation.json',dict(vm=vm,disk=disk,snapshot=snap,zone=ZONE,storage_deadline_unix=deadline))
        return vm,disk,ZONE,snap,deadline
    except BaseException:
        # Read actual state, including when a create/observe call timed out.
        # No model has been submitted while allocate() owns this envelope.
        try:
            matches=[v for v in c.cloud('compute','instances','list') if v['name']==name]
            if disk is None:
                disks=[v for v in c.cloud('compute','disks','list') if v['name']==disk_name]
                disk=disks[0] if disks else None
            if disk:retire(matches[0] if matches else None,disk,ZONE,snap,version)
            if snap:placement.retire_snapshot(snap)
        except Exception as error:
            c.record('flex_cleanup_needs_actual_state_recovery',error=str(error)[-1000:],storage_deadline_unix=deadline,no_model_submitted=True)
        raise
