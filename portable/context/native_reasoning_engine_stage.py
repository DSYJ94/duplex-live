"""Deadline-bounded complete32 native Engine capture and cold verification."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','python','inputs','version','semantic','data','model','codec-source','interleaver-source','cache','control','assets','proof','engine-out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--deadline-utc',required=True)
    a=p.parse_args();deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    def clean():
        return (subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()==a.source_commit
            and not subprocess.check_output(['git','-C',str(a.source),'status','--porcelain'],text=True))
    if not clean()or not a.cache.exists()or a.engine_out.exists()or deadline<=time.time()+60:
        raise ValueError('New output, clean exact Git source and future bounded deadline required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Existing GPU work must finish; never duplicate or contend')
    a.control.mkdir(exist_ok=False)
    state=dict(schema='dl-native-reasoning32-engine-controller-v1',pid=os.getpid(),
        birth=int(Path('/proc/self/stat').read_text().split()[21]),source_commit=a.source_commit,
        started_utc=datetime.now(timezone.utc).isoformat(),deadline_utc=a.deadline_utc,
        phase='starting',stages=[],complete32_engine_independently_verified=False,
        optimizer_updates=0,provider_requests=0,training_approved=False,behavior_qualified=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        tmp=a.control/'controller.partial';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.control/'controller.json')
    args=[str(a.python),'-u','-m','evals.native_reasoning_engine']
    for n in ('inputs','version','semantic','data','model','codec_source','interleaver_source'):
        args+=['--'+n.replace('_','-'),str(getattr(a,n))]
    args+=['--cache',str(a.cache),'--assets',str(a.assets),'--proof',str(a.proof),'--out',str(a.engine_out)]
    env={k:v for k,v in os.environ.items()if not k.startswith(('OPENAI_','RUNPOD_','AWS_'))and k not in('HF_TOKEN','HUGGING_FACE_HUB_TOKEN')}
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',
        PYTHONPATH='/tmp/duplex-reasoning32-asr-cpu-v1/packages:'+str(a.source),
        CPATH='/usr/local/cuda/include',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',OMP_NUM_THREADS='8',
        TORCHINDUCTOR_CACHE_DIR='/opt/control40-native20-v1-inductor',TRITON_CACHE_DIR='/opt/control40-native20-v1-triton')
    child=None
    def stop(*_):
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def interrupted(signum,_):
        stop();state.update(phase='interrupted_preserved',signal=signum);save();raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save()
    try:
        for name,command,device in (('complete32_engine_capture',args,'0'),
                ('independent_complete32_engine_verify',[*args,'--verify-only'],'')):
            if not clean():raise ValueError('Execution source changed')
            stage=dict(name=name,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            state['stages'].append(stage);state['phase']=name
            with(a.control/(name+'.log')).open('x')as log:
                child=subprocess.Popen(command,cwd=a.source,env=dict(env,CUDA_VISIBLE_DEVICES=device),
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                stage.update(pid=child.pid,birth=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]));save()
                try:code=child.wait(timeout=max(1,deadline-time.time()))
                finally:
                    if child.poll()is None:stop()
            stage.update(actual_exit=code,finished_utc=datetime.now(timezone.utc).isoformat());save()
            if code:raise RuntimeError('Complete Engine stage failed: '+name)
        if not clean():raise ValueError('Execution source changed')
        state.update(phase='complete32_engine_only',complete32_engine_independently_verified=True)
    except BaseException as error:
        state.update(phase='failed_preserved_no_retry',error_type=type(error).__name__,reason=str(error));raise
    finally:
        stop();save()


if __name__=='__main__':main()
