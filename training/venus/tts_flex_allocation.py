"""One queued A100 envelope; never resubmit on timeout or partial startup."""
import json,time,math
from pathlib import Path
from datetime import datetime
from . import controller as c
from . import tts_placement as placement
from .tts_completion_flex_guard import AUTH,sha

NAME='duplex-venus-tts-flex-20261104-a-v62'
DISK='duplex-venus-tts-flex-data-20261104-a-v62'
ZONE='us-central1-a'


def retire(vm,disk,zone,snap,version='62'):
    assert version=='62' and zone==ZONE and disk['name']==DISK and disk['id']!='4375908109499109167'
    if vm:
        actual=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
        assert actual['id']==vm['id'] and actual['name']==NAME
        assert actual['disks'][0]['source'].endswith('/disks/'+DISK)
        if actual.get('lastStartTimestamp'):
            if actual['status']!='TERMINATED':
                c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
            final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            assert final['status']=='TERMINATED'
            seconds=datetime.fromisoformat(final['lastStopTimestamp']).timestamp()-datetime.fromisoformat(final['lastStartTimestamp']).timestamp()
            if not any(e['event']=='tts_stop_gpu_terminal' for e in c.events):
                c.record('flex_envelope_terminal',vm_id=vm['id'],actual_seconds=seconds,upper_seconds=math.ceil(seconds/100)*100)
        # Deleting the exact still-pending envelope cancels its queue request.
        # A timed-out observation is never used as authority to create again.
        c.run(['gcloud','compute','instances','delete',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
    actual=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE)
    assert actual['id']==disk['id'] and actual['sourceSnapshotId']==snap['id'] and not actual.get('users')
    c.run(['gcloud','compute','disks','delete',DISK,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet'],150)
    c.record('flex_temporary_envelope_retired',vm_id=vm['id'] if vm else None,disk_id=disk['id'],original_disk_preserved=True)


def allocate():
    a=json.loads(AUTH.read_text());assert a['status']=='APPROVED' and a['human_approval']=='Approve queued Flex-start within existing caps'
    assert sha(a['proposal_path'])==a['proposal_sha256']
    p=json.loads(Path(a['proposal_path']).read_text())
    assert p['compute']['VM_name']==NAME and p['storage']['disk_name']==DISK
    assert p['compute']['queue_wait_seconds_max']==3600 and p['storage']['temporary_window_seconds']==7200
    assert p['storage']['combined_upper_each']<=400
    assert all(v['status']=='TERMINATED' for v in c.cloud('compute','instances','list'))
    assert not any(v['name'].startswith(('duplex-venus-tts-data-','duplex-venus-tts-flex-data-')) for v in c.cloud('compute','disks','list'))
    deadline=time.time()+7200
    with (c.R/'storage-deadline.json').open('x') as f:json.dump(dict(storage_deadline_unix=deadline),f)
    snap=None;disk=None
    try:
        snap=placement.snapshot()
        assert time.time()+5100<deadline,'Full queue, GPU and copyback time required before creating a clone'
        c.record('one_flex_clone_request',name=DISK,zone=ZONE)
        c.run(['gcloud','compute','disks','create',DISK,'--zone='+ZONE,'--source-snapshot='+snap['name'],'--size=100GB','--type=pd-balanced','--labels=duplex-scope=tts-v62','--project='+c.PROJECT,'--quiet','--format=json'],900)
        disk=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE);placement.save('disk.json',disk)
        assert disk['sourceSnapshotId']==snap['id'] and int(disk['sizeGb'])==100 and not disk.get('users')
        assert time.time()+5100<deadline,'Remaining window must cover whole queue and evaluation'
        c.NAME=NAME
        op=c.cloud('compute','instances','create',NAME,'--zone='+ZONE,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=3600s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+DISK+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(c.R/'bootstrap.sh'),'--max-run-duration=1200s','--instance-termination-action=STOP','--async')
        op=op[0] if isinstance(op,list) else op;placement.save('create.json',op)
        c.record('one_queued_flex_request',operation=op['name'],queue_seconds=3600,storage_deadline_unix=deadline)
        until=min(time.time()+3660,deadline-1500)
        while op['status']!='DONE':
            assert time.time()<until,'Observe or cancel this exact envelope; no resubmission after timeout'
            time.sleep(10);op=c.cloud('compute','operations','describe',op['name'],'--zone='+ZONE);placement.save('latest-operation.json',op)
        placement.save('terminal-operation.json',op)
        assert not op.get('error'),op.get('error')
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
        assert vm['status']=='RUNNING' and vm.get('lastStartTimestamp')
        placement.save('allocation.json',dict(vm=vm,disk=disk,snapshot=snap,zone=ZONE,storage_deadline_unix=deadline))
        return vm,disk,ZONE,snap,deadline
    except BaseException:
        # Read actual state, including when a create/observe call timed out.
        # No model has been submitted while allocate() owns this envelope.
        try:
            matches=[v for v in c.cloud('compute','instances','list') if v['name']==NAME]
            if disk is None:
                disks=[v for v in c.cloud('compute','disks','list') if v['name']==DISK]
                disk=disks[0] if disks else None
            if disk:retire(matches[0] if matches else None,disk,ZONE,snap)
            if snap:placement.retire_snapshot(snap)
        except Exception as error:
            c.record('flex_cleanup_needs_actual_state_recovery',error=str(error)[-1000:],storage_deadline_unix=deadline,no_model_submitted=True)
        raise
