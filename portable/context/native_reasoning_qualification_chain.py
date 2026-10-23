"""One bounded CPU chain: typed whole32, full speech, prepared reviewer inputs.

Never submits GPU work, trains, sends provider requests or retries a failed stage.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(a):
    deadline=datetime.fromisoformat(a.deadline_utc).timestamp()
    if not time.time()+60<deadline<time.time()+4*3600:raise ValueError('Finite CPU-only qualification deadline required')
    source=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Clean committed source required')
    commands=[('typed_complete32',[sys.executable,'-m','evals.native_reasoning_teacher_mixed','--config',str(a.config),'--out',str(a.batch)]),
        ('independent_complete32_speech',[sys.executable,'-m','evals.native_reasoning_teacher_mixed_speech','--config',str(a.config),'--batch',str(a.batch),'--out',str(a.speech)]),
        ('prepared_semantic_inputs',[sys.executable,'-m','evals.native_reasoning_semantic_prepared','prepare','--config',str(a.config),'--batch',str(a.batch),'--speech-approval',str(a.speech/'teacher-speech-quality.json'),'--prepared',str(a.prepared)])]
    a.control.mkdir(parents=True,exist_ok=False)
    state=dict(schema='dl-native-reasoning32-CPU-qualification-chain-v1',pid=os.getpid(),
        birth=int(Path('/proc/self/stat').read_text().split()[21]),source_commit=commit,
        deadline_utc=a.deadline_utc,phase='starting',stages=[],provider_requests=0,
        optimizer_updates=0,semantic_approved=False,training_approved=False,passed=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat();p=a.control/'status.partial'
        p.write_text(json.dumps(state,indent=2)+'\n');p.replace(a.control/'status.json')
    save();child=None
    env={k:v for k,v in os.environ.items()if not k.startswith(('OPENAI_','RUNPOD_','AWS_'))
         and k not in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN')}
    env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    try:
        for name,command in commands:
            if time.time()>=deadline:raise TimeoutError('Qualification deadline reached')
            row=dict(name=name,command=command,started_utc=datetime.now(timezone.utc).isoformat())
            state['stages'].append(row);state['phase']=name;save()
            with(a.control/(name+'.log')).open('x')as log:
                child=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row.update(pid=child.pid,birth=int(Path('/proc',str(child.pid),'stat').read_text().split()[21]));save()
                code=child.wait(timeout=max(1,min(600,deadline-time.time())))
            row.update(actual_exit=code,ended_utc=datetime.now(timezone.utc).isoformat());save()
            if code:raise ValueError(name+' failed; preserve actual output and do not retry unchanged')
        state.update(phase='complete32_ready_for_independent_semantic_review',passed=True);save();return state
    except BaseException as error:
        if child is not None and child.poll()is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            state['stages'][-1]['actual_exit']=child.returncode
        state.update(phase='failed_preserved_no_retry',error_type=type(error).__name__,reason=str(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','batch','speech','prepared','control'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True);print(json.dumps(run(p.parse_args())))
