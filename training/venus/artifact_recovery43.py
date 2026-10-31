"""One artifact-only restart of the stopped comparison VM; never loads a model."""
import hashlib
import json
import math
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from . import controller as c

NAME='duplex-venus-foundation-20261030-v43'
IDENT='456124662276463658'
ZONE='us-central1-a'
R=Path('runs/venus-training-20261029/foundation43-artifact-recovery-v1')
PRIOR=Path('runs/venus-training-20261029/action-v43')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not R.exists(), 'Observe existing recovery; never duplicate it'
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text())
    assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
    previous=json.loads((PRIOR/'provider-final.json').read_text())
    assert vm['id']==previous['id']==IDENT and vm['status']=='TERMINATED' and vm['lastStopTimestamp']==previous['lastStopTimestamp']
    assert all(x['status']=='TERMINATED' for x in c.cloud('compute','instances','list'))
    disk=c.cloud('compute','disks','describe','duplex-venus-train-20261026-v1','--zone='+ZONE)
    assert disk['id']=='4375908109499109167' and int(disk['sizeGb'])==100
    elapsed=datetime.fromisoformat(vm['lastStopTimestamp']).timestamp()-datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    prior=25300+math.ceil(elapsed/100)*100
    assert prior==26600 and prior+600<=30300 and (prior+600)*4/3600+24<70
    from portable.venus.guard import bounded_runpod_read
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip())
    if isinstance(pods,dict):pods=pods.get('pods',[])
    assert all(x['status'] in ('EXITED','STOPPED','TERMINATED') for x in pods)
    rows=json.loads((PRIOR/'unique-inventory.json').read_text())
    missing=[x['path'] for x in rows if not (PRIOR/'copyback'/x['path']).exists() or (PRIOR/'copyback'/x['path']).stat().st_size!=x['bytes']]
    assert len(missing)==77 and all(x.startswith(('foundation-native-v43/','journals/duplex-venus-foundation43-native.log')) for x in missing)
    R.mkdir();c.R=R
    (R/'files.txt').write_text('\n'.join(missing)+'\n')
    c.record('artifact_only_preflight',vm_id=IDENT,missing_files=len(missing),prior_seconds_upper=prior,maximum_seconds=600,new_model_runs=0,new_optimizer_updates=0,source=c.run(['git','rev-parse','HEAD']).strip())
    started=False;timer=None;watchdog_done=threading.Event()
    def stop():
        try:
            current=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            if current['id']==IDENT and current['status']!='TERMINATED':
                c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
        finally:watchdog_done.set()
    try:
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=1h')
        c.run(['gcloud','compute','instances','set-scheduling',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--max-run-duration=600s','--instance-termination-action=STOP','--quiet'],60)
        # Independent Mac timer is armed before the provider request; native STOP
        # and the existing600-second bootstrap timer provide guest-side bounds.
        timer=threading.Timer(600,stop);timer.daemon=True;timer.start()
        started=True  # An unknown request outcome still requires cleanup.
        operation=c.cloud('compute','instances','start',NAME,'--zone='+ZONE,'--async')
        c.record('one_artifact_restart',operation=operation)
        until=time.time()+180
        while True:
            live=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            if live['status']=='RUNNING':break
            assert time.time()<until, 'Observe same operation; do not restart'
            time.sleep(5)
        assert live['id']==IDENT and int(live['scheduling']['maxRunDuration']['seconds'])==600
        c.record('artifact_vm_running',vm_id=IDENT,last_start=live['lastStartTimestamp'],native_stop_seconds=600)
        c.ssh_argv=json.loads((PRIOR/'ssh.json').read_text())
        c.ssh_argv[-1]=c.ssh_argv[-1].split('@',1)[0]+'@'+live['networkInterfaces'][0]['accessConfigs'][0]['natIP']
        for attempt in range(15):
            try:
                c.ssh('true',15);break
            except RuntimeError:
                if attempt==14:raise
                time.sleep(3)
        assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip(), 'Artifact recovery must not run a model'
        c.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=core','--exclude=core.*','--files-from='+str(R/'files.txt'),'-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+c.OUT+'/',str(PRIOR/'copyback')+'/'],180)
        # Verify every original inventory entry, including old state, once.
        for row in rows:
            p=PRIOR/'copyback'/row['path']
            assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'], row['path']
        receipt=dict(all_hashes_verified=True,files=len(rows),bytes=sum(x['bytes'] for x in rows),source='96e85552aa305e23515c03c4a0010b46c0084bf1',recovered_files=len(missing),original_failed_transfer_preserved=True)
        (R/'copyback-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
        c.record('full_original_inventory_recovered_and_verified',**receipt)
    finally:
        if started:stop()
        if timer:timer.cancel()
        final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
        (R/'provider-final.json').write_text(json.dumps(final,indent=2)+'\n')
        assert final['id']==IDENT and final['status']=='TERMINATED'
        if started:
            seconds=datetime.fromisoformat(final['lastStopTimestamp']).timestamp()-datetime.fromisoformat(final['lastStartTimestamp']).timestamp()
            c.record('artifact_restart_stopped',actual_seconds=seconds,focused_seconds_upper=prior+math.ceil(seconds/100)*100)


if __name__=='__main__':main()
