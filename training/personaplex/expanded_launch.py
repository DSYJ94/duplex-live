"""Host Slurm entrypoint: one admitted allocation, disjoint shard writers and learner.

Credentials never enter this program. A scheduler timeout cannot submit a job.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
from .expanded_guard import AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
from .matched_paths import command,BASE


def main():
    p=argparse.ArgumentParser()
    for n in ('root','data','backup'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source',required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args();job=os.environ['SLURM_JOB_ID'];a.root.mkdir(parents=True,exist_ok=True);a.backup.mkdir(parents=True,exist_ok=True)
    observation=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=30);actual=parse_job(observation)
    raw=subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'],universal_newlines=True,timeout=30);prior=reserved=0
    for line in raw.splitlines():
        fields=line.split('|')
        if len(fields)<6 or fields[0]==job:continue
        ident,state,elapsed,allocated,requested,limit=fields[:6];tres=dict(x.split('=',1)for x in (allocated or requested).split(',')if '='in x);gpus=int(tres.get('gres/gpu',0));seconds=int(elapsed)
        prior+=gpus*seconds
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if not limit.isdigit():raise ValueError('Every concurrent reservation needs a real bounded time limit')
            reserved+=gpus*max(0,int(limit)*60-seconds)
    quota=subprocess.check_output(['quota','-w','-v'],universal_newlines=True,timeout=20);line=next(x.split()for x in quota.splitlines()if x.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '));used,hard=int(line[1])*1024,int(line[3])*1024
    if hard-used<12*1024**3:raise ValueError('Real user HOME quota needs checkpoint publication headroom')
    lease=dict(provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=4,job_id=job,node=actual['NodeList'],source=a.source,start_unix=stamp(actual['StartTime']),stop_unix=stamp(actual['EndTime']),prior_award_GPU_seconds=prior,other_reserved_GPU_seconds=reserved,home_used_bytes=used,home_hard_bytes=hard)
    validate_job(actual,lease,time.time());folder=a.root/('job-'+job);folder.mkdir(exist_ok=False);path=folder/'lease.json';path.write_text(json.dumps(lease,indent=2)+'\n');(folder/'provider-admission.txt').write_text(observation+'\n'+raw+'\n'+quota)
    guard=folder/'guard.json';environment=dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(guard),DUPLEX_LIVE_LEASE_FILE=str(path));watch=subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(path),'--output',str(guard)],env=environment)
    try:
        deadline=time.time()+180
        while not guard.exists():
            if watch.poll()is not None or time.time()>deadline:raise RuntimeError('Actual independent scheduler admission unavailable; no model started')
            time.sleep(2)
        common=['--data',str(a.data),'--root',str(a.root/'codec'),'--model',str(BASE/'public-models/personaplex-fdaf-v1'),'--codec-source',str(BASE/'vendor/moshi-rag-8c6df-v1'),'--interleaver-source',str(BASE/'vendor/moshi-finetune-2acc-v1'),'--source',a.source,'--shards','4']
        env={k:environment[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')};env.update(OMP_NUM_THREADS=8,MKL_NUM_THREADS=8)
        def execute(module,args,gpu=False,parallel=False):
            cmd=command(Path.cwd(),gpu='0,1,2,3'if gpu else None,environment=env);image=cmd.index(str(BASE/'artifacts/native-v75/runtime/pytorch-new.sif'));cmd[image:image]=['--bind',str(a.backup)+':'+str(a.backup)]
            if parallel:cmd+=['torch.distributed.run','--standalone','--nproc_per_node=4','--module',module]+args
            else:cmd+=[module]+args
            subprocess.run(cmd,env=environment,check=True)
        if not (a.root/'codec'/'cache.json').exists():
            (a.root/'codec').mkdir(exist_ok=True)
            execute('training.personaplex.expanded_codec',['text-preflight']+common)
            execute('training.personaplex.expanded_codec',['build']+common+['--lease',str(path)],gpu=True,parallel=True)
            execute('training.personaplex.expanded_codec',['verify']+common)
        capture=['--data',str(a.data),'--cache',str(a.root/'codec'),'--root',str(a.root/'capture'),'--source',a.source,'--shards','4']
        if not (a.root/'capture'/'inputs.json').exists():
            execute('training.personaplex.expanded_capture',['capture']+capture+['--lease',str(path)],gpu=True,parallel=True)
            execute('training.personaplex.expanded_capture',['verify']+capture)
        runner=['--root',str(a.root),'--expanded',str(a.root/'capture'),'--old-inputs',str(BASE/'native-moshi-matched-inputs-v1'),'--lease',str(path),'--backup',str(a.backup),'--source',a.source]
        if not (a.root/'admission.json').exists():execute('training.personaplex.expanded_runner',['admit']+runner,gpu=True,parallel=True)
        checkpoint=a.checkpoint or a.root/'checkpoints'/'init.pt'
        execute('training.personaplex.expanded_runner',['train']+runner+['--checkpoint',str(checkpoint)],gpu=True,parallel=True)
        terminal=json.loads((folder/'train'/'terminal.json').read_text())
        if terminal['training_complete']:execute('training.personaplex.expanded_runner',['verify']+runner+['--checkpoint',terminal['checkpoint']],gpu=True,parallel=True)
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,training=terminal),indent=2)+'\n')
    finally:
        watch.terminate();watch.wait(timeout=30)


if __name__=='__main__':main()
