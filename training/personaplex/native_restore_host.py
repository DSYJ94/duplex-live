"""Restore-only recovery from independently backed cache; never rerun collection."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .expanded_guard import AUTH,RESTORE_AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
from .native_feature_host import feature_command,archive


def main(a):
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:raise ValueError('Exact immutable recovery source required')
    scope=json.loads(RESTORE_AUTH.read_text());cache=Path(scope['cache_root']);backed=Path(scope['cache_backup']);old=scope['parent_job']
    oldraw=subprocess.check_output(['sacct','-j',old,'-X','-n','-P','-o','JobIDRaw,State,Account,ElapsedRaw,ExitCode'],universal_newlines=True,timeout=30)
    row=next(r.split('|')for r in oldraw.splitlines()if r.startswith(old+'|'))
    if row[1:3]!=['FAILED',ACCOUNT]or int(row[3])!=scope['original_used_GPU_seconds']or row[4]!='1:0':raise ValueError('Exact terminal failed parent and all actual usage required')
    receipt=json.loads((backed/'archive-receipt.json').read_text());terminal=json.loads((cache/'terminal.json').read_text())
    if not receipt['passed']or not receipt['every_file_sha256_verified']or sha(backed/'archive-inventory.json')!=scope['inventory_sha256']or not terminal['passed']or terminal['source']!=scope['cache_source']:raise ValueError('Original complete independent cache proof required')
    for name,key in (('zero-state.pt','snapshot_sha256'),('features.pt','features_sha256')):
        if sha(cache/name)!=scope[key]or sha(backed/name)!=scope[key]:raise ValueError('Both original and independent snapshot/feature hashes required')
    if scope['original_used_GPU_seconds']+scope['maximum_seconds']>scope['original_task_maximum_GPU_seconds']:raise ValueError('No extra GPU-time allowance')
    a.root.mkdir(exist_ok=True);a.backup.mkdir(parents=True,exist_ok=True);lock=(a.root/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ['SLURM_JOB_ID']
    if job==old:raise ValueError('New actual allocation required, not an old container PID')
    actual=parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30));raw=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30);prior=reserved=0
    for line in raw.splitlines():
        f=line.split('|')
        if len(f)<7 or f[0]==job:continue
        ident,name,state,elapsed,allocated,requested,limit=f[:7];tres=dict(v.split('=',1)for v in (allocated or requested).split(',')if '='in v);count=int(tres.get('gres/gpu',0));used=int(elapsed);prior+=count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-')and count:raise ValueError('Never overlap valid voice work')
            if not limit.isdigit():raise ValueError('Actual bounded concurrent reservations required')
            reserved+=count*max(0,int(limit)*60-used)
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20);q=next(r.split()for r in quota.splitlines()if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '));used,hard=int(q[1])*1024,int(q[3])*1024
    if hard-used<2*1024**3:raise ValueError('Verification backup needs actual2GiB headroom')
    parent=dict(job_id=old,state='FAILED',cache_source=scope['cache_source'],cache_root=str(cache),cache_independently_verified=True,inventory_sha256=scope['inventory_sha256'])
    lease=dict(restore_authorization_sha256=sha(RESTORE_AUTH),restore_parent=parent,provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=1,job_id=job,node=actual['NodeList'],source=a.source,start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved,home_used_bytes=used,home_hard_bytes=hard)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False);path=folder/'lease.json';path.write_text(json.dumps(lease,indent=2)+'\n');(folder/'provider-admission.txt').write_text(raw+'\n'+oldraw+'\n'+quota)
    env=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(path));watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(path),'--output',str(folder/'guard.json')],env=env)
    try:
        until=time.time()+180
        while not (folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>until:raise ValueError('Independent actual guard unavailable')
            time.sleep(2)
        inside={k:env[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')};cmd=feature_command(Path.cwd(),True,inside,a.backup)
        pos=cmd.index('--pwd');mounts=[]
        for p in (cache,backed):
            for logical in {str(p),str(p.resolve())}:mounts+=['--bind',str(p.resolve())+':'+logical+':ro']
        cmd[pos:pos]=mounts;cmd+=['training.personaplex.expanded_gpu','training.personaplex.native_feature_worker','verify','--root',str(a.root),'--lease',str(path),'--plan',str(a.plan),'--data',str(a.data),'--source',a.source,'--cache-root',str(cache),'--cache-source',scope['cache_source']]
        child=subprocess.Popen(cmd,env=env);(folder/'launch.json').write_text(json.dumps(dict(host_controller_pid=os.getpid(),actual_container_launcher_pid=child.pid,job_id=job,original_job_id=old,source=a.source,cache_producer=scope['cache_source']),indent=2)+'\n');code=child.wait()
        if code!=0:raise subprocess.CalledProcessError(code,cmd)
        result=json.loads((a.root/'verify/terminal.json').read_text())
        if not result['passed']or not result['actual_CUDA']or not result['all_features_exact']or not result['all_Adam_RNG_OPAL_restored_exact']or result['optimizer_updates']!=0:raise ValueError('Full actual native restore required')
        backup=archive(a.root/'verify',a.backup/('job-'+job)/'verify')
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,source=a.source,cache_producer=scope['cache_source'],parent_job=old,cache_backup=receipt,verify_backup=backup,fresh_native_restore_passed=True,optimizer_updates=0,cache_regenerated=False,automatic_goal_resumed=False,native_policy_accepted=False),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','backup','plan','data'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);main(p.parse_args())
