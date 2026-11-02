"""One approved CPU-only copyback; no model imports or GPU restart path."""
import json,os,shlex,subprocess,sys,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from . import controller as c
from .source_controller import prepare_ssh

R=Path('runs/venus-training-20261102/cpu-recovery-v53')
AUTH=Path('docs/NATIVE-VENUS-CONTEXT466-CPU-RECOVERY-AUTHORIZATION-2026-11-02.json')
ZONE='us-central1-f'
OLD='duplex-venus-context-20261102-f-v52'
NAME='duplex-venus-context466-recovery-20261102-v53'
DISK='duplex-venus-context-data-20261102-f-v52'
DISK_ID='6233528902929943250'
REMOTE='/workspace/migration/venus-training-v1/context-v52'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,data):(R/name).write_text(json.dumps(data,indent=2)+'\n')
def stop():c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet'],150)


def observe(op):
    end=time.time()+600
    while op['status']!='DONE':
        if time.time()>end:raise TimeoutError('Observe same operation; no new create')
        time.sleep(3);op=c.cloud('compute','operations','describe',op['name'],'--zone='+ZONE)
    if op.get('error'):raise RuntimeError(str(op['error']))


def guard():
    lease=json.loads((R/'lease.json').read_text())
    try:
        while True:
            vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            if vm['status']=='TERMINATED':return
            assert vm['id']==lease['vm_id'] and vm['machineType'].endswith('/e2-standard-2') and not vm.get('guestAccelerators')
            assert vm['disks'][0]['source'].endswith('/disks/'+DISK)
            assert time.time()<lease['stop_unix']
            birth=c.run(['/bin/ps','-p',str(lease['controller_pid']),'-o','lstart=']).strip()
            assert birth==lease['controller_birth']
            assert all(x['status']=='TERMINATED' or x['id']==vm['id'] for x in c.cloud('compute','instances','list'))
            save('guard.json',dict(checked_unix=time.time(),vm_id=vm['id'],expires_unix=min(time.time()+60,lease['stop_unix'])))
            time.sleep(15)
    finally:
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
        if vm['status']!='TERMINATED':stop()


def main():
    a=json.loads(AUTH.read_text());assert a['status']=='APPROVED' and a['human_approval']=='Approve CPU recovery within existing caps'
    assert sha(a['proposal_path'])==a['proposal_sha256']
    cleanup_deadline=datetime.fromisoformat('2026-11-02T21:58:31.966921+00:00').timestamp()
    duration=min(1800,int(cleanup_deadline-time.time()-300))
    assert duration>=180,'No approved recovery window remains; preserve disk'
    price=json.loads((R/'price.json').read_text());assert price['hourly_usd']<=.12 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    # CPU<=.064;2h100GiB PD<=.028;2h regional snapshot<=.014;
    # reserve remaining .144 for transfer/rounding, inside explicit .25.
    assert price['hourly_usd']*1900/3600+200*.1/730+200*.05/730+.144<.25
    assert all(x['status']=='TERMINATED' for x in c.cloud('compute','instances','list'))
    old=c.cloud('compute','instances','describe',OLD,'--zone='+ZONE)
    disk=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE)
    assert old['id']=='8703684888303569917' and old['status']=='TERMINATED' and disk['id']==DISK_ID and int(disk['sizeGb'])==100
    with (R/'claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),source=c.run(['git','rev-parse','HEAD']).strip(),authorization_sha256=sha(AUTH)),f)
    c.R=R;c.NAME=NAME
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);watch=None;vm=None;verified=False
    try:
        c.run(['gcloud','compute','instances','delete',OLD,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
        assert not c.cloud('compute','disks','describe',DISK,'--zone='+ZONE).get('users')
        bootstrap=R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-cpu53-stop --on-active='+str(max(30,duration-90))+'s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=1h')
        op=c.cloud('compute','instances','create',NAME,'--zone='+ZONE,'--machine-type=e2-standard-2','--provisioning-model=STANDARD','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+DISK+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(bootstrap),'--max-run-duration='+str(duration)+'s','--instance-termination-action=STOP','--async')
        op=op[0] if isinstance(op,list) else op;save('create-operation.json',op);observe(op)
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-start.json',vm)
        assert vm['status']=='RUNNING' and not vm.get('guestAccelerators')
        end=min(datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+duration,cleanup_deadline-120)
        save('lease.json',dict(vm_id=vm['id'],stop_unix=end,controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip()))
        watch=subprocess.Popen([sys.executable,'-u','-m','training.venus.cpu_recovery','--guard'],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT)
        prepare_ssh(vm,ZONE,'53',require_gpu=False)
        until=time.time()+45
        while not (R/'guard.json').exists():
            assert watch.poll() is None and time.time()<until;time.sleep(1)
        assert time.time()<json.loads((R/'guard.json').read_text())['expires_unix']
        assert c.props('duplex-cpu53-stop.timer')['ActiveState']=='active'
        save('both-guards-ready.json',dict(mac_pid=watch.pid,guest_timer='duplex-cpu53-stop.timer',provider_stop_unix=end))
        code="import pathlib,json,hashlib,os;r=pathlib.Path("+repr(REMOTE)+");print(json.dumps(dict(files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()],links=[dict(path=str(p.relative_to(r)),target=os.readlink(p)) for p in sorted(r.rglob('*')) if p.is_symlink()])))"
        inventory=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(code),120));save('remote-inventory.json',inventory)
        assert any(x['path'].startswith('native-paired/') for x in inventory['files'])
        dest=R/'copyback'
        argv=['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*','-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+REMOTE+'/',str(dest)+'/']
        c.run(argv,min(900,int(end-time.time()-120)))
        for row in inventory['files']:
            p=dest/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        for row in inventory['links']:
            p=dest/row['path'];assert p.is_symlink() and os.readlink(p)==row['target']
        verified=True;save('copyback-verified.json',dict(all_hashes_verified=True,all_link_targets_verified=True,files=len(inventory['files']),bytes=sum(x['bytes'] for x in inventory['files']),links=len(inventory['links']),inventory_sha256=sha(R/'remote-inventory.json')))
        print('FULL_COPYBACK_VERIFIED',flush=True)
    finally:
        found=[x for x in c.cloud('compute','instances','list') if x['name']==NAME]
        if found:
            if found[0]['status']!='TERMINATED':stop()
            final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-final.json',final)
            assert final['status']=='TERMINATED'
        if watch:
            try:watch.wait(timeout=25)
            except subprocess.TimeoutExpired:watch.terminate()
        if verified:
            c.run(['gcloud','compute','instances','delete',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],90)
            disk=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE);assert disk['id']==DISK_ID and not disk.get('users')
            c.run(['gcloud','compute','disks','delete',DISK,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet'],90)
            snap=c.cloud('compute','snapshots','describe','duplex-venus-context-migrate-20261102-v50');assert snap['id']=='4724490043241393447' and snap['sourceDiskId']=='4375908109499109167'
            c.run(['gcloud','compute','snapshots','delete',snap['name'],'--project='+c.PROJECT,'--quiet'],90)
            save('cleanup-complete.json',dict(approved_temporary_resources_removed=True,original_disk_preserved='4375908109499109167'))
        awake.terminate()


if __name__=='__main__':guard() if '--guard' in sys.argv else main()
