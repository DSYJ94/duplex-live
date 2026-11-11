"""One exclusive recovery8 submission, only after the exact owner amendment.

A timeout leaves the claim intact and UNKNOWN: never retry sbatch automatically.
"""
import argparse,json,re,shlex,subprocess,time
from pathlib import Path
from . import matched_rpc_recovery as recovery
from .matched_rpc_job import HISTORY as OLD_HISTORY
from . import matched_rpc_startup as startup
HISTORY=startup.history(OLD_HISTORY)
from .matched_postupdate import prior_usage
from .matched_interrupted import DURABLE
from .matched_reuse import DATA_SHA
from .matched_storage_quota import storage_check


def command(work,source):
    native=['/usr/bin/python3','-B','-u','-m','training.personaplex.matched_rpc_startup_job',
        '--source',source,'--inputs','/projects/u6ada/duplex-live/native-moshi-matched-inputs-v1',
        '--inputs-sha',DATA_SHA]
    return ['sbatch','--parsable','--account=brics.u6ada','--partition=workq','--nodes=1',
        '--gpus=2','--cpus-per-task=16','--mem=200G','--time=24:00:00','--no-requeue','--export=NONE',
        '--job-name=duplex-matched-rpc-recovery8-v1','--chdir='+str(work),
        '--output='+str(DURABLE/'slurm-rpc-recovery-%j.log'),
        '--wrap=ulimit -c 0\nexec '+' '.join(shlex.quote(x) for x in native)]


def submit(source):
    grant=startup.authority()
    work=Path.cwd();work.resolve().relative_to(DURABLE.resolve())
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=source:
        raise ValueError('Exact committed worker required')
    if subprocess.check_output(['git','status','--porcelain'],universal_newlines=True).strip():
        raise ValueError('No uncommitted deployment changes allowed')
    queue=subprocess.check_output(['squeue','-h','-A','brics.u6ada','-o','%i'],universal_newlines=True,timeout=20)
    if queue.strip():raise ValueError('No other project job may overlap recovery8')
    if prior_usage('0',HISTORY)!=135051:raise ValueError('All seven failed allocations must remain counted')
    storage_check()
    claim=DURABLE/'rpc-recovery8-submission-v1';claim.mkdir(exist_ok=False)
    cmd=command(work,source)
    (claim/'command.json').write_text(json.dumps(dict(source=source,authorization_sha256=grant,
        command=cmd,prior_GPU_seconds=135051,interval=8,started_unix=time.time()),indent=2)+'\n')
    try:
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
    except subprocess.TimeoutExpired:
        (claim/'unknown.json').write_text(json.dumps(dict(state='UNKNOWN_DO_NOT_RESUBMIT',source=source))+'\n')
        raise
    (claim/'response.json').write_text(json.dumps(dict(actual_exit=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    if result.returncode or not re.fullmatch(r'[0-9]+(?:;[^\s;]+)?',result.stdout.strip()):
        raise RuntimeError('Submission failed or ambiguous; inspect actual scheduler before any action')
    job=result.stdout.strip().split(';')[0]
    (claim/'job.json').write_text(json.dumps(dict(job_id=job,source=source,actual_sbatch_exit=0))+'\n')
    return job


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    print(submit(p.parse_args().source),flush=True)
