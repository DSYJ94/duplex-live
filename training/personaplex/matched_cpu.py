"""Bounded CPU-only full323 preparation with durable actual host terminals.

The wrapper may survive a lost Mac connection. It records the real host/PIDs
and child exit, and never allocates a GPU or submits a scheduler job.
Compatible with host Python3.6; the actual data module uses container Python.
"""
import argparse,json,os,signal,socket,subprocess,sys,time,uuid
from pathlib import Path
from .matched_paths import BASE,command


def write(path,value):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(str(temporary),str(path))


def stages(work,source,out):
    result=[];snapshot=BASE/'native-control40-workset-v1'
    if not (snapshot/'restore.json').exists():
        incoming=BASE/'incoming-control40-snapshot-v1'
        args=[sys.executable,'-B','-u','-m','training.personaplex.matched_snapshot',
            '--source',source,'--archive',str(incoming/'control40-complete-workset-v1.tar.gz'),
            '--inventory',str(incoming/'inventory.json'),'--out',str(snapshot)]
        if snapshot.exists():args+=['--resume']
        result.append(('restore-source',args))
    result.append(('unit',command(work)+['unittest','-q','evals.harness.test_personaplex_matched_native',
        'evals.harness.test_native_gcp_inputs',
        'evals.harness.test_personaplex_matched_losses','evals.harness.test_personaplex_matched_state',
        'evals.harness.test_personaplex_matched_job','evals.harness.test_personaplex_matched_snapshot',
        'evals.harness.test_personaplex_matched_guard','evals.harness.test_personaplex_matched_paths',
        'evals.harness.test_personaplex_native_capture48','evals.harness.test_personaplex_matched_schedule',
        'evals.harness.test_personaplex_matched_inputs']))
    result.append(('temporary-inputs',command(work)+['training.personaplex.matched_temporary']))
    result.append(('legacy-inputs',command(work)[:-1]+[str(work/'training/personaplex/matched_legacy.py'),
        '--source',source,'--out',str(BASE/'native-moshi-legacy-inputs-v1')]))
    result.append(('prepare',command(work)+['training.personaplex.matched_inputs','--source',source,'--out',str(out),
        '--capture',str(BASE/'native-personaplex-capture48-v1'),
        '--control-engine',str(BASE/'legacy-control40-engine-v1')]))
    return result


def cpu_allocation():
    from .matched_guard import parse_job,stamp,ACCOUNT
    ident=os.environ.get('SLURM_JOB_ID')
    if not ident:raise ValueError('This complete CPU workset preparation requires a scheduler allocation')
    job=parse_job(subprocess.check_output(['scontrol','show','job','-o',ident],universal_newlines=True,timeout=20))
    tres=dict(x.split('=',1) for x in job.get('AllocTRES','').split(',') if '=' in x)
    if (job.get('JobId')!=ident or job.get('Account')!=ACCOUNT or job.get('JobState')!='RUNNING'
            or any(k.startswith('gres/gpu') and v!='0' for k,v in tres.items()) or job.get('NumNodes')!='1'
            or not 0<stamp(job['EndTime'])-stamp(job['StartTime'])<=1800
            or not stamp(job['StartTime'])<=time.time()<stamp(job['EndTime'])-1200):
        raise ValueError('Exact running account/one-node/zero-GPU/30-minute CPU job required')
    return dict(job_id=ident,node=job['NodeList'],GPU_count=0,stop_unix=stamp(job['EndTime']))


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source:raise ValueError('Exact Git CPU producer required')
    allocation=cpu_allocation() if a.require_slurm_cpu else None
    out=Path(a.out);out.resolve().relative_to(BASE.resolve())
    if out.exists():raise ValueError('Preserve existing complete or partial output; inspect before any rerun')
    tracker=Path(a.tracker);tracker.resolve().relative_to(BASE.resolve());tracker.mkdir(exist_ok=False)
    work=Path.cwd();started=time.time();active=None;code=1
    record=dict(source=source,host=socket.gethostname(),manager_pid=os.getpid(),
        started_unix=started,execution_id=uuid.uuid4().hex,maximum_seconds=1200,
        GPU_allocated=False,optimizer_updates=0)
    if allocation:record['actual_Slurm']=allocation
    write(tracker/'manager.json',record)
    try:
        for name,args in stages(work,source,out):
            with (tracker/(name+'.log')).open('x') as log:
                active=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                phase=dict(record,stage=name,pid=active.pid,phase_started_unix=time.time())
                write(tracker/'current.json',phase)
                try:
                    while active.poll() is None:
                        if time.time()>=started+1200:raise subprocess.TimeoutExpired(args,1200)
                        # This is a fresh observation of the SAME actual Popen
                        # child, not a claim based on an old lock or intent file.
                        write(tracker/'current.json',dict(phase,checked_unix=time.time(),actual_child_returncode=None))
                        time.sleep(5)
                    actual=active.wait()
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid,signal.SIGTERM)
                    try:active.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
                    write(tracker/(name+'-terminal.json'),dict(phase,actual_exit_code=active.returncode,timed_out=True))
                    raise
            write(tracker/(name+'-terminal.json'),dict(phase,actual_exit_code=actual,ended_unix=time.time()))
            if actual:raise RuntimeError('Actual CPU stage failed: '+name)
        code=0
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        write(tracker/'terminal.json',dict(record,actual_exit_code=code,ended_unix=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','out','tracker'):p.add_argument('--'+name,required=True)
    p.add_argument('--require-slurm-cpu',action='store_true')
    main(p.parse_args())
