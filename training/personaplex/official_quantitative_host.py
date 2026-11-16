"""One missing-condition collection after reference success; no retry/submission."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .expanded_guard import AUTH,OFFICIAL_AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
from .expanded_launch import container_command
from .expanded_evaluation_host import parent_provider
BASE=Path('/projects/u6ada/duplex-live')


def official_command(work,gpu,environment,backup):
    env=dict(environment,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,GIT_OPTIONAL_LOCKS='0',PYTHONPATH=str(BASE/'official-personaplex-runtime-20261116-v1/pylibs')+':'+str(BASE/'artifacts/personaplex-runtime-v1/pylibs')+':'+str(work))
    cmd=container_command(work,gpu=gpu,environment=env,backup=backup)
    if gpu:
        selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not selected or ','in selected or selected=='-1':raise ValueError('One actual scheduler GPU required')
        if cmd.count('CUDA_VISIBLE_DEVICES=0,1,2,3')!=1:raise ValueError('Exact native command ownership required')
        cmd[cmd.index('CUDA_VISIBLE_DEVICES=0,1,2,3')]='CUDA_VISIBLE_DEVICES='+selected
    return cmd


def main(a):
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:raise ValueError('Exact committed source required')
    parent=parent_provider('8523671')
    terminal=json.loads((a.reference/'job-8523671/host-terminal.json').read_text())
    if parent.get('JobState')!='COMPLETED'or parent.get('ExitCode')!='0:0'or not terminal.get('success')or not terminal.get('complete76'):raise ValueError('Successful full local reference and independent archive required')
    if sha(Path(terminal['backup'])/'archive-inventory.json')!=terminal['inventory_sha256']:raise ValueError('Original reference inventory changed')
    a.root.mkdir(exist_ok=True);a.backup.mkdir(exist_ok=True,parents=True);lock=(a.root/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ['SLURM_JOB_ID'];actual=parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30))
    accounting=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30);prior=reserved=0
    for line in accounting.splitlines():
        f=line.split('|')
        if len(f)<7 or f[0]==job:continue
        ident,name,state,elapsed,allocated,requested,limit=f[:7];tres=dict(v.split('=',1)for v in (allocated or requested).split(',')if '='in v);count=int(tres.get('gres/gpu',0));used=int(elapsed);prior+=count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-')and count:raise ValueError('Never overlap valid voice work')
            if not limit.isdigit():raise ValueError('Actual concurrent bounded reservations required')
            reserved+=count*max(0,int(limit)*60-used)
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20);q=next(r.split()for r in quota.splitlines()if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '));used,hard=int(q[1])*1024,int(q[3])*1024
    if hard-used<8*1024**3:raise ValueError('Full new archive requires actual8GiB quota headroom')
    lease=dict(official_authorization_sha256=sha(OFFICIAL_AUTH),provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=1,job_id=job,node=actual['NodeList'],source=a.source,start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved,home_used_bytes=used,home_hard_bytes=hard)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False);path=folder/'lease.json';path.write_text(json.dumps(lease,indent=2)+'\n');(folder/'provider-admission.txt').write_text(accounting+'\n'+quota)
    env=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(path));watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(path),'--output',str(folder/'guard.json')],env=env)
    try:
        until=time.time()+180
        while not (folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>until:raise ValueError('Actual independent guard unavailable')
            time.sleep(2)
        inside={k:env[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')}
        command=official_command(Path.cwd(),True,inside,a.backup)
        subprocess.run(command+['training.personaplex.expanded_gpu','training.personaplex.official_quantitative','--root',str(a.root),'--questions',str(a.questions),'--lease',str(path),'--source',a.source],env=env,check=True)
        for voice in ('NATF2','RPBen-normalized'):
            report=json.loads((a.root/voice/'results.json').read_text())
            if report['source']!=a.source or not report['complete']or len(report['cases'])!=64:raise ValueError('All missing128 required')
        from .evaluation_result_archive import inventory
        from .matched_archive import independent_filesystems
        if not independent_filesystems(a.root,a.backup):raise ValueError('Independent full archive required')
        stage=a.backup/('.job-'+job+'-partial');stage.mkdir(exist_ok=False);rows=[]
        for voice in ('NATF2','RPBen-normalized'):
            rows.extend(dict(r,path=voice+'/'+r['path'])for r in inventory(a.root/voice))
        rows=sorted(rows,key=lambda r:r['path'])
        names=b''.join(r['path'].encode()+b'\0'for r in rows);subprocess.run(['rsync','-ar','--no-owner','--no-group','--timeout=180','--from0','--files-from=-',str(a.root)+'/',str(stage)+'/'],input=names,check=True)
        if sorted(inventory(stage),key=lambda r:r['path'])!=rows:raise ValueError('Every independent output hash must agree')
        (stage/'archive-inventory.json').write_text(json.dumps(rows,indent=2)+'\n');target=a.backup/('job-'+job);os.rename(stage,target)
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,complete128=True,prior_official32_reused='8203137',backup=str(target),inventory_sha256=sha(target/'archive-inventory.json'),source=a.source,optimizer_updates=0,external_API_calls=0),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','backup','reference','questions'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);main(p.parse_args())
