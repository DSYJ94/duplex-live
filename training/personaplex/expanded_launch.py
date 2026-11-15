"""Host Slurm entrypoint: one admitted allocation, disjoint shard writers and learner.

Credentials never enter this program. A scheduler timeout cannot submit a job.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
from .expanded_guard import AUTH,ACCOUNT,KEY,sha,stamp,parse_job,validate_job
from .matched_paths import command,BASE


def container_command(work,*,gpu,environment,backup):
    work=Path(work);cmd=command(work,gpu='0,1,2,3' if gpu else None,environment=environment)
    image=cmd.index(str(BASE/'artifacts/native-v75/runtime/pytorch-new.sif'))
    mounts=[('--bind',str(backup)+':'+str(backup))]
    # Worktree .git files can point into allocated HOME. Expose ONLY the exact
    # Git metadata directories read-only, never the whole HOME or credentials.
    repositories=[work,BASE/'vendor/moshi-rag-8c6df-v1',BASE/'vendor/moshi-finetune-2acc-v1',BASE/'vendor/context-spanning-79df-v1']
    seen=set()
    for repository in repositories:
        raw=subprocess.check_output(['git','-C',str(repository),'rev-parse','--git-common-dir'],universal_newlines=True).strip();common=Path(raw)
        if not common.is_absolute():common=repository/common
        logical=Path(os.path.abspath(str(common)));physical=logical.resolve()
        if str(logical)not in seen:
            seen.add(str(logical));mounts.append(('--bind',str(physical)+':'+str(logical)+':ro'))
    cmd[image:image]=[part for pair in mounts for part in pair]
    return cmd


def adaptive_stages(execute,runner,folder,root,plan,checkpoint=None):
    """Exact restore -> durable transition -> fresh restore -> actual learning."""
    if checkpoint is None:
        execute('training.personaplex.expanded_runner',['transition']+runner+['--checkpoint',plan['parent_checkpoint']],gpu=True,parallel=True)
        transition=json.loads((folder/'transition'/'transition-terminal.json').read_text())
        if not transition['passed']or transition['new_updates']!=0:raise ValueError('Native transition did not pass')
        checkpoint=Path(transition['checkpoint'])
    if checkpoint.resolve().parent!=(root/'checkpoints').resolve():
        raise ValueError('Adaptive resume must use a checkpoint in its new namespace')
    metadata=json.loads(Path(str(checkpoint)+'.sha256.json').read_text())
    execute('training.personaplex.expanded_runner',['verify-transition']+runner+['--checkpoint',str(checkpoint)],gpu=True,parallel=True)
    verification=json.loads((folder/'verify-transition'/'verify-terminal.json').read_text())
    if (not verification['passed']or verification['new_updates']!=0 or verification['step']!=metadata['step']
            or verification['checkpoint_sha256']!=metadata['sha256']):raise ValueError('Fresh adaptive native restore failed; no learning')
    execute('training.personaplex.expanded_runner',['train-adaptive']+runner+['--checkpoint',str(checkpoint)],gpu=True,parallel=True)


def main():
    p=argparse.ArgumentParser()
    for n in ('root','data','backup'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--adaptive-plan',type=Path);p.add_argument('--source',required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args();job=os.environ['SLURM_JOB_ID'];a.root.mkdir(parents=True,exist_ok=True);a.backup.mkdir(parents=True,exist_ok=True)
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
        env={k:environment[k]for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')};env.update(OMP_NUM_THREADS=8,MKL_NUM_THREADS=8,GIT_OPTIONAL_LOCKS='0')
        def execute(module,args,gpu=False,parallel=False):
            cmd=container_command(Path.cwd(),gpu=gpu,environment=env,backup=a.backup)
            if parallel:cmd+=['torch.distributed.run','--standalone','--nproc_per_node=4','--module','training.personaplex.expanded_gpu',module]+args
            else:cmd+=(['training.personaplex.expanded_gpu',module]if gpu else[module])+args
            subprocess.run(cmd,env=environment,check=True)
        adaptive=None
        if a.adaptive_plan is not None:
            adaptive=json.loads(a.adaptive_plan.read_text())
            if adaptive['source']!=a.source or Path(adaptive['root']).resolve()!=a.root.resolve()or Path(adaptive['backup']).resolve()!=a.backup.resolve():raise ValueError('Exact adaptive source/output namespace required')
            # Fresh scheduler evidence precedes any GPU/model command. No cancel,
            # retry, job submission or automatic fallback exists in this branch.
            parent_raw=subprocess.check_output(['scontrol','show','job','-o',adaptive['parent_job']],universal_newlines=True,timeout=30)
            parent_provider=folder/'parent-provider.json'
            evidence=dict(observed_unix=time.time(),job=parse_job(parent_raw));resume_args=[]
            if a.checkpoint is not None:
                if a.checkpoint.resolve().parent!=(a.root/'checkpoints').resolve():raise ValueError('Checkpoint outside adaptive namespace')
                candidates=[]
                for terminal_path in a.root.glob('job-*/host-terminal.json'):
                    host=json.loads(terminal_path.read_text());terminal=host.get('training',{})
                    if host.get('success')is True and terminal.get('continuation_ready')is True:candidates.append((terminal['actual_new_step'],terminal_path.parent.name[4:]))
                if not candidates:raise ValueError('No successful adaptive parent; no startup retry')
                previous=max(candidates)[1]
                raw_resume=subprocess.check_output(['scontrol','show','job','-o',previous],universal_newlines=True,timeout=30)
                evidence['resume_job']=parse_job(raw_resume);resume_args=['--resume-checkpoint',str(a.checkpoint)]
            parent_provider.write_text(json.dumps(evidence)+'\n')
            execute('training.personaplex.adaptive_handover',['--plan',str(a.adaptive_plan),'--provider-json',str(parent_provider)]+resume_args)
            runner=['--root',str(a.root),'--expanded',str(Path(adaptive['parent_root'])/'capture'),'--old-inputs',str(BASE/'native-moshi-matched-inputs-v1'),'--lease',str(path),'--backup',str(a.backup),'--source',a.source,'--adaptive-plan',str(a.adaptive_plan)]
            adaptive_stages(execute,runner,folder,a.root,adaptive,a.checkpoint)
        else:
            if not (a.root/'codec'/'cache.json').exists():
                (a.root/'codec').mkdir(exist_ok=True)
                pref=a.root/'codec'/'text-preflight.json'
                if not pref.exists():execute('training.personaplex.expanded_codec',['text-preflight']+common)
                else:
                    checked=json.loads(pref.read_text())
                    if not checked['complete1072'] or checked['source_manifest_sha256']!=sha(a.data/'manifest.json') or checked['code_sha256']!=sha(Path('training/personaplex/expanded_codec.py')):raise ValueError('Actual complete CPU preflight differs')
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
