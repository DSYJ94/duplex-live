"""One Slurm-bounded public model download; no model execution or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

BASE=Path('/projects/u6ada/duplex-live')
MANIFEST=BASE/'artifacts/native-v75/receipts/public-model-manifest.json'
MANIFEST_SHA='f001676decf7db11f095b807fb1a4f4dda5ee07bded662b209d90805c0b60ea4'


def main(source):
    job=os.environ.get('SLURM_JOB_ID')
    if not job or os.environ.get('SLURM_JOB_ACCOUNT')!='brics.u6ada':
        raise ValueError('Run through the authorized project Slurm allocation, never on a login node')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if actual!=source or platform.machine()!='aarch64':raise ValueError('Exact Git source and ARM64 platform required')
    if sys.flags.optimize or hashlib.sha256(MANIFEST.read_bytes()).hexdigest()!=MANIFEST_SHA:raise ValueError('Unoptimized Python and exact35-file public manifest required')
    out=BASE/'staging-20261106-v1'/('slurm-'+job);out.mkdir(exist_ok=False)
    proof=dict(job_id=job,account='brics.u6ada',node=os.environ.get('SLURMD_NODENAME'),
        source=source,started_utc=datetime.now(timezone.utc).isoformat(),new_optimizer_updates=0,
        model_execution=False,public_revision='4681004131ea1f1fb9aca9f0db76be322657cc79')
    (out/'actual-start.json').write_text(json.dumps(proof,indent=2)+'\n')
    child=None;code=1
    try:
        child=subprocess.Popen([sys.executable,'-u','-m','portable.venus.fetch_public',
            '--manifest',str(MANIFEST),'--output',str(BASE/'public-model-slurm-v1')],start_new_session=True)
        try:code=child.wait(timeout=1500)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait();code=124
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
        (out/'actual-terminal.json').write_text(json.dumps(dict(proof,
            actual_exit_code=code,ended_utc=datetime.now(timezone.utc).isoformat(),
            all35_hashes_verified=code==0),indent=2)+'\n')
    raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    main(p.parse_args().source)
