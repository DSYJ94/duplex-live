"""One sequential zero-update native collection/restore allocation; no retry."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .expanded_guard import AUTH,FEATURE_AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
from .expanded_launch import container_command
from .expanded_evaluation_host import parent_provider


def feature_command(work,gpu,environment,backup):
    cmd=container_command(work,gpu=gpu,environment=dict(environment,OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,GIT_OPTIONAL_LOCKS='0'),backup=backup)
    if gpu:
        selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not selected or ','in selected or selected=='-1'or cmd.count('CUDA_VISIBLE_DEVICES=0,1,2,3')!=1:raise ValueError('One actual scheduler-selected GPU required')
        cmd[cmd.index('CUDA_VISIBLE_DEVICES=0,1,2,3')]='CUDA_VISIBLE_DEVICES='+selected
    return cmd


def archive(source,target):
    from .evaluation_result_archive import inventory
    from .matched_archive import independent_filesystems
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists()or not independent_filesystems(source,target.parent):raise ValueError('New independent immutable cache/verification backup required')
    rows=inventory(source);stage=target.with_name('.'+target.name+'-partial');stage.mkdir(exist_ok=False)
    names=b''.join(r['path'].encode()+b'\0'for r in rows)
    subprocess.run(['rsync','-ar','--no-owner','--no-group','--timeout=180','--from0','--files-from=-',str(source)+'/',str(stage)+'/'],input=names,check=True)
    if inventory(stage)!=rows:raise ValueError('Every independent file hash must match before next stage')
    (stage/'archive-inventory.json').write_text(json.dumps(rows,indent=2)+'\n');receipt=dict(passed=True,every_file_sha256_verified=True,files=len(rows),bytes=sum(r['bytes']for r in rows),inventory_sha256=sha(stage/'archive-inventory.json'),destination=str(target),optimizer_updates=0)
    (stage/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');os.rename(stage,target);return receipt


def main(a):
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:raise ValueError('Exact immutable committed source required')
    parent=parent_provider('8526401');terminal=json.loads((a.parent/'job-8526401/host-terminal.json').read_text())
    if parent.get('JobState')!='COMPLETED'or parent.get('ExitCode')!='0:0'or not terminal.get('success')or not terminal.get('complete128'):raise ValueError('Do not interrupt or overlap valid comparison')
    if sha(Path(terminal['backup'])/'archive-inventory.json')!=terminal['inventory_sha256']:raise ValueError('Full parent backup required')
    a.root.mkdir(exist_ok=True);a.backup.mkdir(parents=True,exist_ok=True);lock=(a.root/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ['SLURM_JOB_ID'];actual=parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30))
    raw=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30);prior=reserved=0
    for line in raw.splitlines():
        f=line.split('|')
        if len(f)<7 or f[0]==job:continue
        ident,name,state,elapsed,allocated,requested,limit=f[:7];tres=dict(v.split('=',1)for v in (allocated or requested).split(',')if '='in v);count=int(tres.get('gres/gpu',0));used=int(elapsed);prior+=count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-')and count:raise ValueError('Another voice allocation is active')
            if not limit.isdigit():raise ValueError('Exact concurrent bounded reservations required')
            reserved+=count*max(0,int(limit)*60-used)
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20);q=next(r.split()for r in quota.splitlines()if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '));used,hard=int(q[1])*1024,int(q[3])*1024
    if hard-used<8*1024**3:raise ValueError('Complete independent backup needs8GiB actual headroom')
    lease=dict(feature_authorization_sha256=sha(FEATURE_AUTH),provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=1,job_id=job,node=actual['NodeList'],source=a.source,start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved,home_used_bytes=used,home_hard_bytes=hard)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False);path=folder/'lease.json';path.write_text(json.dumps(lease,indent=2)+'\n');(folder/'provider-admission.txt').write_text(raw+'\n'+quota)
    env=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(path));watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(path),'--output',str(folder/'guard.json')],env=env)
    try:
        until=time.time()+180
        while not (folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>until:raise ValueError('Independent actual scheduler guard unavailable')
            time.sleep(2)
        inside={k:env[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')};command=feature_command(Path.cwd(),True,inside,a.backup)
        receipts={};terminals={}
        for stage in ('cache','verify'):
            subprocess.run(command+['training.personaplex.expanded_gpu','training.personaplex.native_feature_worker',stage,'--root',str(a.root),'--lease',str(path),'--plan',str(a.plan),'--data',str(a.data),'--source',a.source],env=env,check=True)
            result=json.loads((a.root/stage/'terminal.json').read_text())
            if not result['passed']or not result['actual_CUDA']or result['optimizer_updates']!=0 or result['source']!=a.source:raise ValueError('Exact native zero-update stage required')
            terminals[stage]=result;receipts[stage]=archive(a.root/stage,a.backup/('job-'+job)/stage)
            (folder/(stage+'-backup.json')).write_text(json.dumps(receipts[stage],indent=2)+'\n')
        if terminals['cache']['process_identity']==terminals['verify']['process_identity']or not terminals['verify']['all_features_exact']or not terminals['verify']['all_Adam_RNG_OPAL_restored_exact']:raise ValueError('Actual independent full native restoration required')
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,source=a.source,native_cache_complete=True,fresh_native_restore_passed=True,backups=receipts,optimizer_updates=0,automatic_goal_resumed=False,native_policy_accepted=False),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','backup','parent','plan','data'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);main(p.parse_args())
