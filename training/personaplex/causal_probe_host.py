"""One bounded frozen four-condition native diagnostic; no learning or APIs."""
import argparse,json,os,subprocess,time,fcntl
from pathlib import Path
from .expanded_guard import AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
from .expanded_evaluation_host import evaluation_command,parent_provider
SCOPE=Path('docs/NATIVE-CAUSAL-PROBE-EXECUTION-2026-11-16.json')


def main(a):
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:raise ValueError('Exact committed probe source required')
    scope=json.loads(SCOPE.read_text())
    if scope['status']!='OWNER_AUTHORIZED_CAUSE_SEPARATING_PRODUCT_WORK'or scope['required_new_outputs']!=304 or scope['new_optimizer_updates']!=0 or scope['new_external_model_API_calls']!=0:raise ValueError('Exact owner-directed frozen diagnostic scope required')
    lock=(a.root/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ['SLURM_JOB_ID'];actual=parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30))
    if stamp(actual['EndTime'])-stamp(actual['StartTime'])>21600:raise ValueError('Bounded6h fourGPU probe required')
    accounting=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30)
    prior=reserved=0
    for line in accounting.splitlines():
        f=line.split('|')
        if len(f)<7 or f[0]==job:continue
        ident,name,state,elapsed,allocated,requested,limit=f[:7];tres=dict(v.split('=',1)for v in (allocated or requested).split(',')if '='in v);count=int(tres.get('gres/gpu',0));used=int(elapsed);prior+=count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-')and count:raise ValueError('Another voice job is active; never overlap')
            if not limit.isdigit():raise ValueError('All active shared reservations require actual bounded time')
            reserved+=count*max(0,int(limit)*60-used)
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20);q=next(r.split()for r in quota.splitlines()if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '));used,hard=int(q[1])*1024,int(q[3])*1024
    if hard-used<8*1024**3:raise ValueError('Independent result archive requires8GiB actual quota headroom')
    lease=dict(causal_scope_sha256=sha(SCOPE),provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=4,job_id=job,node=actual['NodeList'],source=a.source,start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved,home_used_bytes=used,home_hard_bytes=hard)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False);a.backup.mkdir(parents=True,exist_ok=True)
    path=folder/'lease.json';path.write_text(json.dumps(lease,indent=2)+'\n');(folder/'provider-admission.txt').write_text(accounting+'\n'+quota)
    original=parent_provider('8478956');(folder/'parent-provider.json').write_text(json.dumps(original)+'\n')
    env=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(path));watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(path),'--output',str(folder/'guard.json')],env=env)
    try:
        until=time.time()+180
        while not (folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>until:raise ValueError('Actual scheduler guard unavailable')
            time.sleep(2)
        inside={k:env[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')};inside.update(OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,GIT_OPTIONAL_LOCKS='0',HF_HOME='/projects/u6ada/duplex-live/hf-cache')
        command=evaluation_command(Path.cwd(),gpu=False,environment=inside,backup=a.backup,parent_backup=a.parent_backup)
        code="import json;from pathlib import Path;from training.personaplex.expanded_evaluation import completed;from training.personaplex.foundation_context import preflight;selection=completed(Path(%r),'3c6f699e85adb26ba764e473ee491c041c62f6db',json.loads(Path(%r).read_text()),Path(%r));assert len(preflight(Path.cwd()))==32;Path(%r).write_text(json.dumps(dict(selection=selection,parent_root=%r),indent=2))"%(str(a.training_root),str(folder/'parent-provider.json'),str(a.parent_backup),str(folder/'plan.json'),str(a.training_root))
        subprocess.run(command[:-1]+['-c',code],env=env,check=True)
        command=evaluation_command(Path.cwd(),gpu=True,environment=inside,backup=a.backup,parent_backup=a.parent_backup)
        command+=['torch.distributed.run','--standalone','--nproc_per_node=4','--module','training.personaplex.expanded_gpu','training.personaplex.causal_probe','run','--root',str(a.root),'--source',a.source,'--plan',str(folder/'plan.json'),'--lease',str(path)]
        subprocess.run(command,env=env,check=True)
        from .evaluation_result_archive import inventory
        from .matched_archive import independent_filesystems
        rows=[]
        for model in ('baseline','expanded105M'):
            for condition in ('prefix-text-only','audio-plus-prefix-text'):
                base=a.root/(model+'-'+condition);report=json.loads((base/'results.json').read_text())
                if report['source']!=a.source or not report['complete']or len(report['cases'])!=76:raise ValueError('All304 complete new outputs required')
                rows.extend(dict(row,path=base.name+'/'+row['path'])for row in inventory(base))
        summary=dict(complete304=True,source=a.source,optimizer_updates=0,external_API_calls=0,semantic_review_pending=True,original_audio_only_reused=True)
        (a.root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
        for p in a.root.glob('*.json'):rows.append(dict(path=p.name,kind='file',bytes=p.stat().st_size,sha256=sha(p)))
        rows=sorted(rows,key=lambda r:r['path']);target=a.backup/('job-'+job)
        if target.exists()or not independent_filesystems(a.root,a.backup):raise ValueError('New independent immutable result archive required')
        stage=a.backup/('.job-'+job+'-partial');stage.mkdir(exist_ok=False)
        for i in range(0,len(rows),2000):
            names=b''.join(r['path'].encode()+b'\0'for r in rows[i:i+2000]);subprocess.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--timeout=180','--from0','--files-from=-',str(a.root)+'/',str(stage)+'/'],input=names,check=True)
        if sorted(inventory(stage),key=lambda r:r['path'])!=rows:raise ValueError('All new output hashes must agree on independent backup')
        (stage/'archive-inventory.json').write_text(json.dumps(dict(source=str(a.root),files=rows),indent=2)+'\n');receipt=dict(passed=True,files=len(rows),bytes=sum(r['bytes']for r in rows),inventory_sha256=sha(stage/'archive-inventory.json'),destination=str(target),every_file_sha256_verified=True,optimizer_updates=0)
        (stage/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');os.rename(stage,target)
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,complete304=True,backup=receipt,optimizer_updates=0,semantic_review_pending=True),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','backup','parent-backup','training-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);main(p.parse_args())
