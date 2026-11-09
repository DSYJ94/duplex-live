"""Bounded CPU-only full323 preparation with durable actual host terminals.

The wrapper may survive a lost Mac connection. It records the real host/PIDs
and child exit, and never allocates a GPU or submits a scheduler job.
Compatible with host Python3.6; the actual data module uses container Python.
"""
import argparse,json,os,signal,socket,subprocess,time,uuid
from pathlib import Path
from .matched_paths import BASE,command


def write(path,value):
    temporary=Path(str(path)+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(str(temporary),str(path))


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source:raise ValueError('Exact Git CPU producer required')
    out=Path(a.out);out.resolve().relative_to(BASE.resolve())
    if out.exists():raise ValueError('Preserve existing complete or partial output; inspect before any rerun')
    tracker=Path(a.tracker);tracker.resolve().relative_to(BASE.resolve());tracker.mkdir(exist_ok=False)
    work=Path.cwd();started=time.time();active=None;code=1
    record=dict(source=source,host=socket.gethostname(),manager_pid=os.getpid(),
        started_unix=started,execution_id=uuid.uuid4().hex,maximum_seconds=1200,
        GPU_allocated=False,optimizer_updates=0)
    write(tracker/'manager.json',record)
    try:
        stages=[('unit',['unittest','-q','evals.harness.test_personaplex_matched_native',
            'evals.harness.test_personaplex_matched_losses','evals.harness.test_personaplex_matched_state',
            'evals.harness.test_personaplex_matched_job','evals.harness.test_personaplex_matched_snapshot',
            'evals.harness.test_personaplex_matched_guard','evals.harness.test_personaplex_matched_paths',
            'evals.harness.test_personaplex_native_capture48',
            'evals.harness.test_personaplex_matched_schedule','evals.harness.test_personaplex_matched_inputs']),
            ('prepare',['training.personaplex.matched_inputs','--source',source,'--out',str(out),
                '--capture',str(BASE/'native-personaplex-capture48-v1'),
                '--control-engine',str(BASE/'legacy-control40-engine-v1')])]
        for name,args in stages:
            with (tracker/(name+'.log')).open('x') as log:
                active=subprocess.Popen(command(work)+args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
    main(p.parse_args())
