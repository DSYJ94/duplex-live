"""One scheduler allocation, independent host guard, serial native processes."""
import argparse
from datetime import datetime,timezone
import json,os,signal,subprocess,sys,time,uuid
from pathlib import Path
from . import isambard_guard as g

BASE=Path('/projects/u6ada/duplex-live')
ART=BASE/'artifacts/native-v75'

def main(a):
    if getattr(a,'merge_lora',False) and a.stage!='posttrain-evaluate':
        raise ValueError('Inference folding is a separate post-training evaluation only')
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source:raise ValueError('Exact Git execution source required')
    job_id=os.environ['SLURM_JOB_ID']
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',job_id],universal_newlines=True,timeout=20))
    root=Path(a.root);root.resolve().relative_to(BASE.resolve())
    execution=root/('execution-'+job_id);execution.mkdir(parents=True,exist_ok=False)
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
               job_id=job_id,node=job['NodeList'],source=source,start_unix=g.stamp(job['StartTime']),
               stop_unix=g.stamp(job['EndTime']),maximum_seconds=7200 if a.stage in ('evaluate','posttrain-evaluate','retention-source-audit') else 86400,GPU_count=1)
    g.validate_job(job,lease,time.time());lease_path=execution/'lease.json';lease_path.write_text(json.dumps(lease,indent=2)+'\n')
    guard_path=execution/'guard.json'
    watcher=subprocess.Popen([sys.executable,'-B','-u','-m','training.venus.isambard_guard','--lease',str(lease_path),'--output',str(guard_path)],stdout=(execution/'guard.log').open('x'),stderr=subprocess.STDOUT)
    child=None;code=1
    try:
        until=time.time()+40
        while not guard_path.exists():
            if watcher.poll() is not None or time.time()>until:raise RuntimeError('Independent Slurm guard did not start')
            time.sleep(1)
        visible=os.environ['CUDA_VISIBLE_DEVICES']
        if not visible or ',' in visible:raise ValueError('One actual scheduler-selected GPU required')
        work=str(BASE/a.worktree)
        common=['apptainer','exec','--cleanenv','--containall','--nv','--bind',str(BASE)+':'+str(BASE),
                '--bind',str(BASE.resolve())+':'+str(BASE.resolve()),
                '--bind',str(ART/'recovered-data-v1/venus-training-v1')+':/workspace/migration/venus-training-v1:ro','--pwd',work]
        env=dict(CUDA_VISIBLE_DEVICES=visible,SLURM_JOB_ID=job_id,SLURM_JOB_ACCOUNT=g.ACCOUNT,
                 DUPLEX_LIVE_LEASE_FILE=str(lease_path),DUPLEX_LIVE_GUARD_FILE=str(guard_path),DUPLEX_WAV_IO='soundfile-wav-v1',
                 HF_HOME=str(BASE/'hf-cache'),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                 TMPDIR=str(BASE/('tmp/native-'+job_id)))
        Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
        for key,value in env.items():common+=['--env',key+'='+value]
        runtime=[str(ART/'runtime/pytorch-new.sif'),str(BASE/'envs/venus-arm64-v1/bin/python'),'-u','-m']
        model=str(BASE/'public-model-slurm-v1/Realtime-Venus-Audio');data=str(ART/'recovered-data-v1/venus-training-v1')
        args=['--root',str(root),'--source',source,'--lease',str(lease_path),'--model',model,'--opal-source',str(BASE/'opal-upstream-v1'),
              '--plan',work+'/evals/fixtures/venus-control-repair-source-v2-20261027.json','--timelines',data+'/paced-timelines-v22',
              '--approval',data+'/paced-data-approval-v22.json','--word-quality',data+'/paced-word-quality-v22.json',
              '--old-plan',work+'/evals/fixtures/venus-control-corpus-v4-20261026.json','--old-timelines',data+'/timelines-v7-repaired',
              '--old-approval',data+'/data-approval-v7.json','--voice-reference',str(ART/'recovered-data-v1/venus-british-native-v85/references/rp-ben-16k.wav'),
              '--voice-reference-sha256','552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0']
        if a.stage=='evaluate':
            stages=[('training.venus.isambard_native',args+['--stage','schema-selftest'],'schema-selftest')]
            stages += [('training.venus.isambard_native',args+['--arm',arm,'--stage',stage],arm+'-'+stage) for arm in ('main','control') for stage in ('reference','restore')]
            stages += [('training.venus.isambard_native',args+['--arm',arm,'--stage','collect'],arm+'-native') for arm in ('main','control')]
            stages += [('training.venus.isambard_native',args+['--stage','combine'],'combine')]
        elif a.stage=='posttrain-evaluate':
            if getattr(a,'merge_lora',False):args+=['--merge-lora','--baseline-profile','unmerged646']
            stages=[('training.venus.isambard_posttrain',args+['--arm','main','--stage',stage],name)
                    for stage,name in (('collect','main-native'),('compare','compare'))]
        elif a.stage=='retention-source-audit':
            stages=[('data.venus.retention_quality',[
                '--original',str(root/'original'),'--resumed',str(root/'resumed'),
                '--payload',work+'/docs/NATIVE-RETENTION96-SPEECH-PAYLOAD-2026-11-07.json',
                '--output',str(root/'quality-v1'),'--source',source,'--lease',str(lease_path)],'source-audit')]
        else:
            stages=[('training.venus.isambard_train',args+['--epochs',str(a.epochs)],'training'),
                    ('training.venus.isambard_train',args+['--epochs',str(a.epochs),'--final-restore'],'final-restore')]
            if a.resume_checkpoint:
                extra=['--resume-checkpoint',a.resume_checkpoint,'--resume-sha',a.resume_sha,'--resume-step',str(a.resume_step),'--resume-prior-root',a.resume_prior_root]
                stages=[('training.venus.isambard_resume',args+extra+['--stage',stage],'resume-'+stage) for stage in ('reference','restore')]+[(stages[0][0],stages[0][1]+extra,'training'),stages[1]]
        for module,argv,name in stages:
            g.ensure(lease,json.loads(guard_path.read_text()))
            started=time.time();log=(execution/(name+'.log')).open('x')
            invocation=uuid.uuid4().hex
            child=subprocess.Popen(common+['--env','DUPLEX_NATIVE_EXECUTION_ID='+invocation]+runtime+[module]+argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (execution/'current-process.json').write_text(json.dumps(dict(stage=name,pid=child.pid,execution_id=invocation,module=module,source=source,started_unix=started),indent=2)+'\n')
            while child.poll() is None:
                if watcher.poll() is not None:raise RuntimeError('Independent scheduler guard exited')
                g.ensure(lease,json.loads(guard_path.read_text()));time.sleep(5)
            log.close();result=dict(stage=name,pid=child.pid,execution_id=invocation,actual_exit_code=child.returncode,seconds=time.time()-started,source=source)
            (execution/(name+'-terminal.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
            if child.returncode:raise RuntimeError('Actual native stage failed: '+name)
        code=0
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        watcher.terminate()
        try:watcher.wait(timeout=10)
        except subprocess.TimeoutExpired:watcher.kill();watcher.wait()
        (execution/'terminal.json').write_text(json.dumps(dict(actual_exit_code=code,source=source,job_id=job_id,stage=a.stage,ended_utc=datetime.now(timezone.utc).isoformat(),new_optimizer_updates=0 if a.stage in ('evaluate','posttrain-evaluate','retention-source-audit') else None),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','root','worktree'):p.add_argument('--'+name,required=True)
    p.add_argument('--stage',choices=('evaluate','posttrain-evaluate','train','retention-source-audit'),required=True);p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--merge-lora',action='store_true')
    p.add_argument('--resume-checkpoint');p.add_argument('--resume-sha');p.add_argument('--resume-step',type=int);p.add_argument('--resume-prior-root')
    main(p.parse_args())
