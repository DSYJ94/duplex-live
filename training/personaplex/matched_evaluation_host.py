"""Evaluation-only approved continuation from four finished native lineages.

Never resumes an optimizer. Real successful training windows and every actual
GPU interval are carried into this NEW scheduler allocation, without replay.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
from . import matched_guard as g,matched_resume as r
from .matched_job import write
from .matched_generation import completed_states
from .matched_interrupted import DURABLE
from .matched_storage_recovery import authority
from .matched_finish import run_evaluation


def completed_parent(root,digest):
    root=r.beneath(root);phase=r.read_pinned(root/'phase-result.json',digest)
    leases=list(root.glob('execution-*/lease.json'))
    if len(leases)!=1:raise ValueError('One actual successful training execution required')
    lease=json.loads(leases[0].read_text());terminal=json.loads((leases[0].parent/'terminal.json').read_text())
    if (phase.get('all_four_complete') is not True or phase.get('partial_lineages')
            or phase.get('source')!=lease['source'] or terminal.get('source')!=lease['source']
            or terminal.get('job_id')!=lease['job_id'] or terminal.get('actual_exit_code')!=0
            or lease.get('storage_recovery_authorization_sha256')!=authority()):
        raise ValueError('Actual successful full training and exact storage grant required')
    positive=0
    for base in ('personaplex','context-spanning'):
        for cap in ('narrow','broad'):
            p=root/base/cap/'train/training-terminal.json'
            if p.exists():
                value=json.loads(p.read_text())
                if value['source']!=lease['source']:raise ValueError('Actual training producer differs')
                updates=value['actual_new_updates']
                if type(updates) is not int or updates<=0:raise ValueError('Actual completed training must contain positive updates')
                positive+=updates
    if positive<=0:raise ValueError('Evaluation continuation needs a genuinely positive completed training window')
    if (root/'rpc-recovery-initial.json').is_file():
        initial=json.loads((root/'rpc-recovery-initial.json').read_text());grant=r.rpc_authority()
        if (initial!={'source':lease['source'],'authorization_sha256':grant,'failed_job':'8285192'}
                or lease.get('rpc_recovery_authorization_sha256')!=grant or lease['continuation_interval']!=7):
            raise ValueError('Exact approved failed6 recovery7 parent required')
        earlier=[]
    elif (root/'storage-recovery-initial.json').is_file():
        initial=json.loads((root/'storage-recovery-initial.json').read_text())
        if initial!={'source':lease['source'],'authorization_sha256':authority(),'failed_job':'8274714'} or lease['continuation_interval']!=6:
            raise ValueError('Exact approved initial recovery window6 required')
        earlier=[]
    else:
        earlier=r.chain(phase['previous_root'],phase['previous_result_sha256'])
        if (earlier[-1]['lease']['source']!=lease['source']
                or earlier[-1]['lease']['continuation_interval']+1!=lease['continuation_interval']):
            raise ValueError('Immutable consecutive successful training windows required')
    completed_states(root,lease['source'])
    return earlier+[dict(root=root,result=phase,lease=lease)]


def new_allocation(parent,source,job,prior):
    if parent['continuation_interval'] not in (6,7):raise ValueError('Only unused authorized7/8 evaluation interval allowed')
    lease=dict(parent,job_id=job['JobId'],node=job['NodeList'],controller_source=source,
        continuation_interval=parent['continuation_interval']+1,prior_GPU_seconds=prior,
        start_unix=g.stamp(job['StartTime']),stop_unix=g.stamp(job['EndTime']),
        maximum_seconds=int(g.stamp(job['EndTime'])-g.stamp(job['StartTime'])))
    if source==parent['source']:raise ValueError('Evaluation producer must be separately recorded')
    g.validate_job(job,lease,time.time());return lease


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.controller_source:raise ValueError('Exact evaluation controller required')
    ordered=completed_parent(a.training_root,a.phase_result_sha);parent=ordered[-1]['lease']
    r.verify_worker(parent['worker_worktree'],parent['source'],source)
    current=os.environ['SLURM_JOB_ID']
    accounting=subprocess.check_output(['sacct','-n','-X','-P','-A',g.ACCOUNT,'--starttime=2026-11-08',
        '--format=JobIDRaw,JobName%100,Account,State,ExitCode,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=30)
    prior=r.allocation_carry(accounting,ordered,current)
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',current],universal_newlines=True,timeout=20))
    if not job.get('JobName','').startswith('duplex-matched-'):raise ValueError('Accounted matched evaluation allocation required')
    allocation=new_allocation(parent,source,job,prior)
    # Keep the NEW allocation separate from the original successful parent.
    record=DURABLE/'runs'/('matched-evaluation-allocation-'+current);record.mkdir(exist_ok=False)
    write(record/'allocation.json',dict(allocation=allocation,source=source,training_root=a.training_root,
        phase_result_sha256=a.phase_result_sha,optimizer_updates=0))
    run_evaluation(a.training_root,source,Path.cwd(),allocation=allocation)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('controller-source','training-root','phase-result-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
