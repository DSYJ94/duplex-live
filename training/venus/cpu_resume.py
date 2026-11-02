"""One authorized same-CPU restart; recover missing files in bounded batches."""
import json,os,shlex,subprocess,sys,time,hashlib
from pathlib import Path
from datetime import datetime
from . import cpu_recovery as r
from .source_controller import prepare_ssh
c=r.c
R=Path('runs/venus-training-20261102/cpu-resume-v54')
PRIOR=Path('runs/venus-training-20261102/cpu-recovery-v53')
AUTH=Path('docs/NATIVE-VENUS-CONTEXT466-CPU-RESUME-AUTHORIZATION-2026-11-02.json')


def batches(rows):
    batch=[];size=0
    for row in rows:
        if row['bytes']>32*1024**2:raise ValueError('File exceeds approved batch bound')
        if batch and size+row['bytes']>32*1024**2:
            yield batch;batch=[];size=0
        batch.append(row);size+=row['bytes']
    if batch:yield batch


def main():
    a=json.loads(AUTH.read_text());assert a['status']=='APPROVED' and a['human_approval']=='Approve one CPU restart within same caps' and r.sha(a['proposal_path'])==a['proposal_sha256']
    assert time.time()<datetime.fromisoformat('2026-11-02T23:00:00+00:00').timestamp()-900
    price=json.loads((PRIOR/'price.json').read_text());assert price['hourly_usd']<=.12 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    assert (184.92+700)*price['hourly_usd']/3600+400*.15/730+.144<.25
    vm=c.cloud('compute','instances','describe',r.NAME,'--zone='+r.ZONE)
    assert vm['id']=='2060195750767541511' and vm['status']=='TERMINATED' and not vm.get('guestAccelerators') and vm['machineType'].endswith('/e2-standard-2')
    disk=c.cloud('compute','disks','describe',r.DISK,'--zone='+r.ZONE);assert disk['id']==r.DISK_ID
    assert all(v['status']=='TERMINATED' for v in c.cloud('compute','instances','list'))
    inventory=json.loads((PRIOR/'remote-inventory.json').read_text());dest=PRIOR/'copyback'
    missing=[]
    for row in inventory['files']:
        p=Path(row['path']);assert not p.is_absolute() and '..' not in p.parts and not any(x in p.parts for x in ('.private','credentials','owner-conversations','session-token','API keys.md'))
        p=dest/p
        if not p.is_file() or p.stat().st_size!=row['bytes'] or r.sha(p)!=row['sha256']:missing.append(row)
    missing.sort(key=lambda x:(not x['path'].endswith(('.json','.jsonl','.txt','.py','.html')),not x['path'].startswith('native-paired/'),x['path']))
    R.mkdir(parents=True,exist_ok=True)
    with (R/'claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),source=c.run(['git','rev-parse','HEAD']).strip(),missing_files=len(missing),inventory_sha256=r.sha(PRIOR/'remote-inventory.json')),f)
    r.R=R;c.R=R;c.NAME=r.NAME
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);watch=None;verified=False
    try:
        bootstrap=R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-cpu54-stop --on-active=510s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        c.run(['gcloud','compute','instances','add-metadata',r.NAME,'--zone='+r.ZONE,'--project='+c.PROJECT,'--metadata-from-file=startup-script='+str(bootstrap),'--quiet'],90)
        c.run(['gcloud','compute','instances','set-scheduling',r.NAME,'--zone='+r.ZONE,'--project='+c.PROJECT,'--max-run-duration=600s','--instance-termination-action=STOP','--quiet'],90)
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=1h')
        op=c.cloud('compute','instances','start',r.NAME,'--zone='+r.ZONE,'--async');op=op[0] if isinstance(op,list) else op;r.save('start-operation.json',op);r.observe(op)
        vm=c.cloud('compute','instances','describe',r.NAME,'--zone='+r.ZONE);r.save('provider-start.json',vm)
        assert vm['id']=='2060195750767541511' and vm['status']=='RUNNING'
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+600
        r.save('lease.json',dict(vm_id=vm['id'],stop_unix=end,controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip()))
        watch=subprocess.Popen([sys.executable,'-u','-m','training.venus.cpu_resume','--guard'],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT)
        prepare_ssh(vm,r.ZONE,'54',require_gpu=False)
        assert c.props('duplex-cpu54-stop.timer')['ActiveState']=='active'
        assert watch.poll() is None and time.time()<json.loads((R/'guard.json').read_text())['expires_unix']
        c.ssh_argv=[('ControlMaster=no' if x.startswith('ControlMaster=') else 'ControlPath=none' if x.startswith('ControlPath=') else 'ControlPersist=no' if x.startswith('ControlPersist=') else x) for x in c.ssh_argv]
        (R/'ssh-direct.json').write_text(json.dumps(c.ssh_argv)+'\n')
        work=list(batches(missing))
        # Links are copied as links, never followed into old model/cache trees.
        work.append([dict(path=x['path'],bytes=0,link=x['target']) for x in inventory['links']])
        for n,rows in enumerate(work):
            assert watch.poll() is None and end-time.time()>60
            filelist=R/('batch-%03d.files'%n);filelist.write_bytes(b''.join(x['path'].encode()+b'\0' for x in rows))
            argv=['/opt/homebrew/bin/rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*','--from0','--files-from='+str(filelist),'-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+r.REMOTE+'/',str(dest)+'/']
            p=subprocess.run(argv,text=True,capture_output=True,timeout=min(90,int(end-time.time()-30)))
            r.save('batch-%03d-terminal.json'%n,dict(exit_code=p.returncode,stderr=p.stderr,stdout=p.stdout,paths=[x['path'] for x in rows]))
            if p.returncode:raise RuntimeError('Actual batch failed; no automatic retry: '+p.stderr[-1200:])
            for row in rows:
                local=dest/row['path']
                if 'link' in row:assert local.is_symlink() and os.readlink(local)==row['link']
                else:assert local.stat().st_size==row['bytes'] and r.sha(local)==row['sha256']
            print(json.dumps(dict(event='batch_verified',batch=n+1,total=len(work),files=len(rows),bytes=sum(x['bytes'] for x in rows))),flush=True)
        for row in inventory['files']:
            p=dest/row['path'];assert p.stat().st_size==row['bytes'] and r.sha(p)==row['sha256']
        for row in inventory['links']:
            p=dest/row['path'];assert p.is_symlink() and os.readlink(p)==row['target']
        verified=True;r.save('copyback-verified.json',dict(all_hashes_verified=True,all_link_targets_verified=True,files=len(inventory['files']),bytes=sum(x['bytes'] for x in inventory['files']),links=len(inventory['links']),inventory_sha256=r.sha(PRIOR/'remote-inventory.json')))
    finally:
        vm=c.cloud('compute','instances','describe',r.NAME,'--zone='+r.ZONE)
        if vm['status']!='TERMINATED':r.stop()
        final=c.cloud('compute','instances','describe',r.NAME,'--zone='+r.ZONE);r.save('provider-final.json',final)
        if watch:
            try:watch.wait(timeout=20)
            except subprocess.TimeoutExpired:watch.terminate()
        if verified:
            c.run(['gcloud','compute','instances','delete',r.NAME,'--zone='+r.ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],90)
            disk=c.cloud('compute','disks','describe',r.DISK,'--zone='+r.ZONE);assert disk['id']==r.DISK_ID and not disk.get('users')
            c.run(['gcloud','compute','disks','delete',r.DISK,'--zone='+r.ZONE,'--project='+c.PROJECT,'--quiet'],90)
            snap=c.cloud('compute','snapshots','describe','duplex-venus-context-migrate-20261102-v50');assert snap['id']=='4724490043241393447' and snap['sourceDiskId']=='4375908109499109167'
            c.run(['gcloud','compute','snapshots','delete',snap['name'],'--project='+c.PROJECT,'--quiet'],90)
            r.save('cleanup-complete.json',dict(temporary_resources_removed=True,original_disk_preserved='4375908109499109167'))
        awake.terminate()


if __name__=='__main__':
    if '--guard' in sys.argv:r.R=R;r.guard()
    else:main()
