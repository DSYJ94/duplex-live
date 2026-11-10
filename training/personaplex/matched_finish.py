"""Final approved window: unchanged positive-state resume, then native evaluation.

Host Python3.6. No scheduler submissions, retries, new optimizer recipe or
external calls. Evaluation carries the SAME actual allocation start/end/cost.
"""
import argparse,json,os,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
from . import matched_guard as g
from . import matched_health
from .matched_job import visible_pair,write
from .matched_paths import command,BASE
from .matched_interrupted import DURABLE
from .matched_generation import completed_states


def evaluation_lease(previous,source,job):
    """A source transition inside one interval cannot reset any budget clock."""
    allowed=(7,8) if previous.get('storage_recovery_authorization_sha256') else (6,)
    if previous['job_id']!=job or previous.get('continuation_interval') not in allowed:
        raise ValueError('Only the last approved actual running window may evaluate here')
    lease=dict(previous,source=source,controller_source=source,trainer_source=previous['source'])
    g.resource(lease)
    for key in ('start_unix','stop_unix','prior_GPU_seconds','maximum_seconds','continuation_interval','GPU_count'):
        if lease[key]!=previous[key]:raise ValueError('Evaluation cannot reset training allocation time')
    return lease


def run_evaluation(root,source,work):
    root=Path(root);job=os.environ['SLURM_JOB_ID']
    old=json.loads((root/('execution-'+job)/'lease.json').read_text())
    if source==old['source']:raise ValueError('Separate immutable evaluation controller source required')
    selected=completed_states(root,old['source'])
    lease=evaluation_lease(old,source,job)
    actual=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True,timeout=20))
    g.validate_job(actual,lease,time.time())
    # Six full32 collections plus local ASR, two bases in parallel. This reserve
    # is not a benchmark or permission to shorten any cohort.
    if lease['stop_unix']-time.time()<10800:
        raise RuntimeError('Full native comparison reserve unavailable; preserve completed training, no subset run')
    evaluation=DURABLE/'runs'/('matched-evaluation-'+job);evaluation.mkdir(exist_ok=False)
    execution=evaluation/('execution-'+job);execution.mkdir()
    write(execution/'lease.json',lease);write(evaluation/'candidate-selection.json',dict(states=selected,trainer_source=old['source'],source=source))
    services=[];control=None;code=1;active={};lock=threading.Lock();failed=threading.Event();errors=[]
    devices=dict(zip(('personaplex','context-spanning'),visible_pair(os.environ['CUDA_VISIBLE_DEVICES'])))
    try:
        watcher,services,control=matched_health.start(lease,execution)
        def ensure():
            if failed.is_set() or watcher.poll() is not None:raise RuntimeError('Evaluation peer or independent scheduler guard stopped')
            g.ensure(lease,json.loads((control/'guard.json').read_text()))
        def worker(base):
            try:
                for capacity in ('baseline','narrow','broad'):
                    for stage in ('collect','audit'):
                        ensure();name=base+'-'+capacity+'-'+stage;ident=uuid.uuid4().hex;child=None;actual_exit=None;began=time.time()
                        env=dict(SLURM_JOB_ID=job,SLURM_JOB_ACCOUNT=g.ACCOUNT,
                            DUPLEX_LIVE_LEASE_FILE=str(execution/'lease.json'),DUPLEX_LIVE_GUARD_FILE=str(control/'guard.json'),
                            DUPLEX_NATIVE_EXECUTION_ID=ident,CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HOME=str(BASE/'hf-cache'))
                        cmd=command(work,gpu=devices[base],environment=env)+['training.personaplex.matched_evaluation',
                            '--source',source,'--trainer-source',old['source'],'--training-root',str(root),
                            '--root',str(evaluation),'--lease',str(execution/'lease.json'),
                            '--base',base,'--capacity',capacity,'--stage',stage]
                        identity=dict(source=source,trainer_source=old['source'],base=base,capacity=capacity,
                            stage=stage,execution_id=ident,started_unix=began,module='training.personaplex.matched_evaluation')
                        try:
                            with (execution/(name+'.log')).open('x') as log:
                                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                                with lock:active[base]=child
                                write(execution/(base+'-current.json'),dict(identity,pid=child.pid))
                                while child.poll() is None:ensure();time.sleep(5)
                                actual_exit=child.returncode
                        finally:
                            if child is not None and child.poll() is None:
                                os.killpg(child.pid,signal.SIGTERM)
                                try:child.wait(timeout=60)
                                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                                actual_exit=child.returncode
                            write(execution/(name+'-terminal.json'),dict(identity,actual_exit_code=actual_exit,ended_unix=time.time()))
                            with lock:active.pop(base,None)
                        if actual_exit!=0:raise RuntimeError('Actual native evaluation child failed: '+name)
            except BaseException as exc:
                failed.set()
                with lock:errors.append(dict(base=base,error=type(exc).__name__+': '+str(exc)))
        threads=[threading.Thread(target=worker,args=(base,)) for base in devices]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        if errors:raise RuntimeError('Native evaluation stopped: '+repr(errors))
        reports={}
        for base in devices:
            for capacity in ('baseline','narrow','broad'):
                name=base+'-'+capacity;path=evaluation/name/'report.json';report=json.loads(path.read_text())
                if report.get('complete') is not True or len(report.get('cases',[]))!=32:
                    raise ValueError('Full six-condition comparison required')
                reports[name]=dict(path=str(path),sha256=g.sha(path),summary=report['summary'])
        write(evaluation/'comparison.json',dict(source=source,trainer_source=old['source'],complete_six_times32=True,
            reports=reports,optimizer_updates=0,external_API_calls=0,connected_backend_tested=False,
            independent_reasoning_semantics_and_listening_still_required=True,fresh20_still_required=True,adopted=False));code=0
    finally:
        if control is not None:matched_health.stop(services,control,execution)
        write(execution/'terminal.json',dict(source=source,trainer_source=old['source'],job_id=job,
            actual_exit_code=code,ended_unix=time.time(),adopted=False))


def main(a):
    from . import matched_resume
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.controller_source:raise ValueError('Exact final host/evaluation producer required')
    # This call preserves the entire approved training resume implementation.
    # It admits only a successful positive prior window, not failed recovery.
    matched_resume.main(a)
    root=DURABLE/'runs'/('native-moshi-matched-resume-'+os.environ['SLURM_JOB_ID'])
    phase=json.loads((root/'phase-result.json').read_text())
    if phase.get('all_four_complete') is not True:
        raise RuntimeError('Final approved training window partial; retain every state, no extra restart')
    run_evaluation(root,source,Path.cwd())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('controller-source','inputs','inputs-sha','previous-root','previous-result-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
