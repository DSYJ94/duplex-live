"""One frozen reference after successful causal outputs; no submission/retry."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from .expanded_guard import AUTH,REFERENCE_AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
from .expanded_launch import container_command
from .expanded_evaluation_host import parent_provider
from .local_reasoner_reference import PUBLIC,PIN

RUNTIME=Path('/projects/u6ada/duplex-live/local-reasoner-runtime-20261116-v1')


def reference_command(work,gpu,environment,backup):
    env=dict(environment,PYTHONPATH=str(RUNTIME/'pylibs')+':'+str(work),USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    cmd=container_command(work,gpu=gpu,environment=env,backup=backup)
    if gpu:
        if cmd.count('CUDA_VISIBLE_DEVICES=0,1,2,3')!=1:raise ValueError('Explicit single GPU command ownership required')
        selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not selected or ','in selected or selected=='-1':raise ValueError('One actual scheduler-selected GPU required')
        cmd[cmd.index('CUDA_VISIBLE_DEVICES=0,1,2,3')]='CUDA_VISIBLE_DEVICES='+selected
    pos=cmd.index('--pwd');cmd[pos:pos]=['--bind',str(PUBLIC)+':'+str(PUBLIC)+':ro','--bind',str(PIN)+':'+str(PIN)+':ro']
    return cmd


def main(a):
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:raise ValueError('Exact committed source required')
    a.root.mkdir(exist_ok=True);lock=(a.root/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    parent=parent_provider('8522764')
    if parent.get('JobState')!='COMPLETED'or parent.get('ExitCode')!='0:0':raise ValueError('Successful causal parent required; never overlap')
    terminal=json.loads((a.parent/'job-8522764/host-terminal.json').read_text())
    receipt_path=a.parent_backup/'job-8522764/archive-receipt.json';receipt=json.loads(receipt_path.read_text())
    if (not terminal.get('success')or not terminal.get('complete304')or terminal.get('backup')!=receipt or not receipt.get('every_file_sha256_verified')
            or sha(receipt_path.parent/'archive-inventory.json')!=receipt['inventory_sha256']):raise ValueError('Full independent parent backup required')
    ready=json.loads((RUNTIME/'cpu-readiness.json').read_text());schema=json.loads((RUNTIME/'model-schema.json').read_text())
    if schema['missing_inference_keys']or schema['mismatched_inference_shapes']or not schema['meta_only_no_weights_allocated']:raise ValueError('Existing full CPU inference schema required')
    job=os.environ['SLURM_JOB_ID'];actual=parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30))
    raw=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30)
    prior=reserved=0
    for line in raw.splitlines():
        f=line.split('|')
        if len(f)<7 or f[0]==job:continue
        ident,name,state,elapsed,allocated,requested,limit=f[:7];tres=dict(v.split('=',1)for v in (allocated or requested).split(',')if '='in v);count=int(tres.get('gres/gpu',0));used=int(elapsed);prior+=count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-')and count:raise ValueError('Other voice allocation/reservation active; never duplicate')
            if not limit.isdigit():raise ValueError('Bound all concurrent shared reservations')
            reserved+=count*max(0,int(limit)*60-used)
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20);q=next(r.split()for r in quota.splitlines()if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '));used,hard=int(q[1])*1024,int(q[3])*1024
    if hard-used<2*1024**3:raise ValueError('Reference backup headroom required')
    lease=dict(reference_authorization_sha256=sha(REFERENCE_AUTH),provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=1,job_id=job,node=actual['NodeList'],source=a.source,start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False);a.backup.mkdir(parents=True,exist_ok=True)
    path=folder/'lease.json';path.write_text(json.dumps(lease,indent=2)+'\n');(folder/'admission.txt').write_text(raw+'\n'+quota);(folder/'cpu-readiness.json').write_text(json.dumps(ready,indent=2)+'\n')
    env=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(path));watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(path),'--output',str(folder/'guard.json')],env=env)
    try:
        until=time.time()+180
        while not(folder/'guard.json').exists():
            if watch.poll()is not None or time.time()>until:raise RuntimeError('Actual reference provider guard unavailable')
            time.sleep(2)
        inside={k:env[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')}
        cmd=reference_command(Path.cwd(),True,inside,a.backup)
        cmd+=['training.personaplex.local_reasoner_reference','--questions',str(a.parent/'questions.json'),'--output',str(a.root/'text-reference'),'--lease',str(path),'--source',a.source]
        subprocess.run(cmd,env=env,check=True,timeout=max(1,lease['stop_unix']-time.time()-240))
        from .evaluation_result_archive import inventory
        from .matched_archive import independent_filesystems
        if not independent_filesystems(a.root,a.backup):raise ValueError('Independent reference backup required')
        result=json.loads((a.root/'text-reference/results.json').read_text())
        if not result['complete76']or len(result['results'])!=76:raise ValueError('Complete reference cohort required')
        rows=inventory(a.root/'text-reference');target=a.backup/('job-'+job);stage=a.backup/('.job-'+job+'-partial');stage.mkdir(exist_ok=False)
        subprocess.run(['rsync','-ar','--no-owner','--no-group','--timeout=120',str(a.root/'text-reference')+'/',str(stage)+'/'],check=True)
        if inventory(stage)!=rows:raise ValueError('Every reference backup hash must agree')
        (stage/'archive-inventory.json').write_text(json.dumps(rows,indent=2)+'\n');os.rename(stage,target)
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,complete76=True,backup=str(target),inventory_sha256=sha(target/'archive-inventory.json'),source=a.source,optimizer_updates=0,external_API_calls=0),indent=2)+'\n')
    finally:watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','backup','parent','parent-backup'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True);main(p.parse_args())
