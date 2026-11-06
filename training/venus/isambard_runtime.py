"""Bounded ARM64 runtime preparation in Slurm; no weights or optimizer loaded."""
import argparse
from datetime import datetime,timezone
import json,os,platform,subprocess
from pathlib import Path

BASE=Path('/projects/u6ada/duplex-live')

def main(source,attempt):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='brics.u6ada' or platform.machine()!='aarch64':
        raise ValueError('Actual project ARM64 Slurm allocation required')
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=source:raise ValueError('Exact Git source required')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible or ',' in visible:raise ValueError('One scheduler-selected GPU required')
    out=BASE/('runtime-preparation-v'+str(attempt));out.mkdir(exist_ok=False)
    start=dict(source=source,job_id=os.environ['SLURM_JOB_ID'],started_utc=datetime.now(timezone.utc).isoformat(),model_execution=False,new_optimizer_updates=0)
    (out/'start.json').write_text(json.dumps(start,indent=2)+'\n')
    command=['apptainer','exec','--cleanenv','--containall','--nv','--bind',str(BASE)+':'+str(BASE),
             '--env','CUDA_VISIBLE_DEVICES='+visible,'--env','PYTHONDONTWRITEBYTECODE=1',
             '--env','HF_HOME='+str(BASE/'hf-cache'),
             '--env','DUPLEX_RUNTIME_ATTEMPT='+str(attempt),'--env','DUPLEX_WAV_IO=soundfile-wav-v1',
             '--env','PYTHONPATH='+str(BASE/('source-isambard-runtime-v'+str(attempt))),
             str(BASE/'artifacts/native-v75/runtime/pytorch-new.sif'),'bash',str(BASE/('source-isambard-runtime-v'+str(attempt))/'training/venus/isambard_runtime.sh')]
    code=1
    try:code=subprocess.call(command)
    finally:(out/'terminal.json').write_text(json.dumps(dict(start,actual_exit_code=code,ended_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    raise SystemExit(code)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--attempt',type=int,required=True);a=p.parse_args();main(a.source,a.attempt)
