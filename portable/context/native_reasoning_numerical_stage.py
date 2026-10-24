"""Wait for exact complete32 Engine exits, then one bounded numerical32 stage."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def process(pid):
    p=Path('/proc',str(pid),'stat')
    if not p.exists():return None
    fields=p.read_text().rsplit(')',1)[1].split()
    return dict(state=fields[0],birth=int(fields[19]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','python','inputs','version','semantic','data','model','codec-source','interleaver-source',
            'cache','assets','proof','engine-inputs','engine-control','out','control'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('engine-pid','engine-birth'):p.add_argument('--'+n,type=int,required=True)
    for n in ('source-commit','engine-source-commit','deadline-utc'):p.add_argument('--'+n,required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    def clean():
        return (subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.source_commit
            and not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain'],text=True))
    if not clean()or a.out.exists()or deadline<=time.time()+60:
        raise ValueError('Clean exact source, new output and bounded future deadline required')
    a.control.mkdir(exist_ok=False)
    state=dict(schema='dl-native-reasoning32-numerical-controller-v1',pid=os.getpid(),
        birth=int(Path('/proc/self/stat').read_text().split()[21]),source_commit=a.source_commit,
        started_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=a.deadline_utc,
        engine_controller_pid=a.engine_pid,engine_controller_birth=a.engine_birth,
        engine_source_commit=a.engine_source_commit,phase='waiting_exact_complete32_engine',
        complete32_numerical_passed=False,optimizer_updates=0,provider_requests=0,training_approved=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        tmp=a.control/'controller.partial';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.control/'controller.json')
    child=None
    def stop():
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def interrupted(signum,_):
        stop();state.update(phase='interrupted_preserved',signal=signum);save();raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);save()
    try:
        while time.time()<deadline-60:
            r=json.loads((a.engine_control/'controller.json').read_text());live=process(a.engine_pid)
            if (r['pid']!=a.engine_pid or r['birth']!=a.engine_birth or r['source_commit']!=a.engine_source_commit
                    or live is not None and live['birth']!=a.engine_birth):
                raise ValueError('Original exact32 Engine controller changed')
            if r['phase']=='complete32_engine_only'and r['complete32_engine_independently_verified']is True:
                if len(r['stages'])!=2 or any(s['actual_exit']!=0 for s in r['stages']):
                    raise ValueError('Both complete32 actual Engine exits must be zero')
                if live is None or live['state']=='Z':break
            elif r['phase'].startswith(('failed','interrupted'))or live is None or live['state']=='Z':
                raise RuntimeError('Exact Engine stage ended without complete independent32 approval')
            time.sleep(5)
        else:raise TimeoutError('No complete32 Engine approval before numerical dispatch deadline')
        lock=(a.out.parent/'native-control40-gpu-owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU unexpectedly occupied after exact upstream completion')
        if not clean():raise ValueError('Source changed before dispatch')
        args=[str(a.python),'-u','-m','evals.native_reasoning_numerical']
        for n in ('inputs','version','semantic','data','model','codec_source','interleaver_source','cache','assets','proof','engine_inputs','out'):
            args+=['--'+n.replace('_','-'),str(getattr(a,n))]
        env={k:v for k,v in os.environ.items()if not k.startswith(('OPENAI_','RUNPOD_','AWS_'))and k not in('HF_TOKEN','HUGGING_FACE_HUB_TOKEN')}
        env.update(CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
            PYTHONPATH='/tmp/duplex-reasoning32-asr-cpu-v1/packages:'+str(a.source),CPATH='/usr/local/cuda/include',
            TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',OMP_NUM_THREADS='8')
        with(a.control/'numerical.log').open('x')as log:
            child=subprocess.Popen(args,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lock.fileno(),))
            state.update(phase='complete32_numerical_running',child_pid=child.pid,
                child_birth=process(child.pid)['birth'],command=args);save()
            try:code=child.wait(timeout=max(1,deadline-time.time()))
            finally:
                if child.poll()is None:stop()
        state['actual_exit']=code;save()
        if code:raise RuntimeError('Complete32 numerical child failed')
        result=json.loads((a.out/'numerical.json').read_text());expected=json.loads((a.engine_inputs/'engine-inputs.json').read_text())
        if (not result['complete32']or not result['passed']or result['threshold']!=.001
                or result['causal_prefix_error_required']!=0. or not result['all_model_parameters_unchanged']
                or result['peak_allocated_bytes']>=80_000_000_000
                or [r['id']for r in result['cases']]!=[r['id']for r in expected['cases']]
                or any(not r['passed']or r['causal_prefix_maximum_error']!=0. or r['native_training_text_alignment_max_error']>.001
                    or not r['native_depth_parity']['passed']or any(x['maximum_absolute_error']>.001 for x in r['comparisons'].values())for r in result['cases'])):
            raise ValueError('All32 unchanged native numerical checks required')
        state.update(phase='complete32_numerical_only',complete32_numerical_passed=True)
    except BaseException as error:
        state.update(phase='failed_preserved_no_retry',error_type=type(error).__name__,reason=str(error));raise
    finally:
        stop()
        if child is not None and child.returncode is not None:state['actual_exit']=child.returncode
        save()


if __name__=='__main__':main()
