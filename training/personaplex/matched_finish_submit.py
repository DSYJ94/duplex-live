"""One final interval10 after a real successful positive recovery9.

CPU-only admission before sbatch. Never watches, retries or mutates training.
An ambiguous submission leaves its exclusive durable claim UNKNOWN.
"""
import argparse,json,re,shlex,subprocess,time
from pathlib import Path
from . import matched_guard as g,matched_resume as resume,matched_rpc_startup as startup
from . import matched_evaluation_host as evaluation
from .matched_interrupted import DURABLE
from .matched_reuse import DATA_SHA
from .matched_storage_quota import storage_check
from . import paced_health_service as paced

ROOT=DURABLE/'runs/rpc-recovery-8317068'
CLAIM=DURABLE/'matched-final10-submission-v1'
SOURCE='10e7e0664e47c1222e0492a8367b3c000ed5789d'
BACKUP_BASE=Path('/projects/u6ada/duplex-live/native-moshi-postupdate-backup-v1')


def walltime(prior):
    # Keep an additional minute per GPU below the unchanged total ceiling.
    if type(prior) is not int or prior<135137:raise ValueError('Actual full prior allocation carry required')
    seconds=min(86400,((112*3600-prior)//2-60)//60*60)
    if seconds<11400:raise ValueError('Insufficient full comparison/startup reserve; no subset allocation')
    return int(seconds)


def parent(root):
    root=resume.beneath(root)
    if root.resolve()!=ROOT.resolve():raise ValueError('Only the actual recovery9 root can finish')
    result=root/'phase-result.json';digest=g.sha(result);phase=json.loads(result.read_text())
    if type(phase.get('all_four_complete')) is not bool:raise ValueError('Actual terminal phase result required')
    ordered=evaluation.completed_parent(root,digest) if phase['all_four_complete'] else resume.chain(root,digest)
    old=ordered[-1]['lease']
    if (old['job_id']!='8317068' or old['source']!=SOURCE or old['continuation_interval']!=9
            or old.get(startup.KEY)!=startup.authority() or old['prior_GPU_seconds']!=135137):
        raise ValueError('Exact successful positive9 and all original clocks/authorities required')
    return ordered,digest,phase['all_four_complete']



def backup_proofs(root,source):
    """Verify only the two NEW final broad saves, never rehash old archives."""
    status=json.loads((root/'independent-archive-status.json').read_text())
    if status.get('passed') is not True or status.get('job_id')!='8317068' or status.get('source')!=source:
        raise ValueError('Actual independent current-window backup status required')
    proofs={}
    for base in ('personaplex','context-spanning'):
        folder=root/base/'broad/train'
        terminal=folder/'training-terminal.json'
        if not terminal.exists():terminal=folder/'training-window-terminal.json'
        value=json.loads(terminal.read_text())
        if value.get('source')!=source or value.get('actual_new_updates',0)<=0:
            raise ValueError('Each broad lineage must have actual positive saved progress')
        checkpoint=Path(value.get('final_checkpoint',value.get('checkpoint','')))
        if checkpoint.parent!=folder:raise ValueError('Exact current training checkpoint required')
        records=[r for r in status['copies'] if r['checkpoint']==str(checkpoint)]
        if len(records)!=1:raise ValueError('Exactly one independent backup of selected final state required')
        record=records[0];destination=Path(record['destination'])
        destination.resolve().relative_to(BACKUP_BASE.resolve())
        actual=json.loads((destination.parent/'archive-receipt.json').read_text())
        for name in ('source','checkpoint','destination','step','sha256','bytes','files'):
            if actual.get(name)!=record.get(name):raise ValueError('Independent backup receipt differs from status')
        if (record.get('source')!=source or record.get('actual_rsync_exit_code')!=0
                or record.get('hash_verified') is not True or record.get('separate_filesystem') is not True
                or record['sha256']!=value['checkpoint_sha256'] or record['step']!=value['total_lineage_updates']
                or destination.is_symlink() or checkpoint.is_symlink()
                or checkpoint.stat().st_dev==destination.stat().st_dev or g.sha(checkpoint)!=record['sha256']):
            raise ValueError('Complete verified backup on an independent filesystem required')
        for name,expected in record['files'].items():
            if Path(name).name!=name:raise ValueError('No aliased backup manifest paths')
            file=destination.parent/name
            if file.is_symlink() or file.stat().st_size!=expected['bytes'] or g.sha(file)!=expected['sha256']:
                raise ValueError('Full final-state backup hash mismatch')
        if destination.name not in record['files'] or record['files'][destination.name]['sha256']!=record['sha256']:
            raise ValueError('Selected checkpoint missing from complete backup manifest')
        native_name=destination.name+'.native-reference.json'
        other=set(record['files'])-{destination.name,native_name}
        if len(record['files'])!=3 or len(other)!=1 or native_name not in record['files']:
            raise ValueError('Checkpoint, native reference and actual save receipt required')
        native=json.loads((destination.parent/native_name).read_text())
        saved=json.loads((destination.parent/next(iter(other))).read_text())
        if (native.get('source')!=source or native.get('step')!=record['step']
                or native.get('checkpoint_sha256')!=record['sha256'] or saved.get('source')!=source
                or saved.get('checkpoint_sha256')!=record['sha256']
                or saved.get('lineage_updates',saved.get('total_lineage_updates'))!=record['step']):
            raise ValueError('Archived native/save receipt identity differs')
        proofs[base]=record
    return proofs



def forecast(root,seconds,complete):
    """Reuse original measured estimates plus actual final restores; no model."""
    if complete:
        if seconds<11400:raise ValueError('Full native comparison reserve unavailable')
        return dict(training_remaining_seconds=0,required_seconds=11400,model_executed=False)
    original=resume.preflight(resume.ORIGINAL_ROOT,resume.PACKET_SHA)
    # resume_plan validates the actual final restore/state records. It constructs
    # dictionaries only; these unused preview paths are never created or saved.
    preview=root/'unused-CPU-plan-preview'
    rows=resume.resume_plan(root,original,preview,preview/'execution',SOURCE)
    estimates={base:0. for base in ('personaplex','context-spanning')}
    origins={(v['base'],v['capacity']):v for v in original['lineages']}
    for row in rows:
        step=row['restored_step'];value=origins[(row['base'],row['capacity'])]
        remaining=value['estimated_complete196_seconds']*(196-step)/196
        if 0<step<196:remaining+=600
        estimates[row['base']]+=remaining
    needed=max(estimates.values())+11400
    if needed>seconds:raise ValueError('Remaining complete training and full evaluation do not fit; no final partial run')
    return dict(training_remaining_seconds_per_base=estimates,required_seconds=needed,
        actual_final_restores_verified_by_resume_plan=True,model_executed=False)


def command(work,source,digest,complete,seconds):
    native=['/usr/bin/python3','-B','-u','-m','training.personaplex.paced_finish',
        '--mode','evaluate' if complete else 'train','--controller-source',source]
    if complete:native+=['--training-root',str(ROOT),'--phase-result-sha',digest]
    else:native+=['--inputs','/projects/u6ada/duplex-live/native-moshi-matched-inputs-v1',
        '--inputs-sha',DATA_SHA,'--previous-root',str(ROOT),'--previous-result-sha',digest]
    return ['sbatch','--parsable','--account=brics.u6ada','--partition=workq','--nodes=1',
        '--gpus=2','--cpus-per-task=16','--mem=200G','--time='+str(seconds//60),
        '--no-requeue','--export=NONE','--job-name=duplex-matched-final10-v1',
        '--chdir='+str(work),'--output='+str(DURABLE/'slurm-final10-%j.log'),
        '--wrap=ulimit -c 0\nexec '+' '.join(shlex.quote(x) for x in native)]


def submit(source):
    pacing_grant=paced.authority()  # explicit amendment required before any allocation
    grant=startup.authority();work=Path.cwd();work.resolve().relative_to(DURABLE.resolve())
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=source:
        raise ValueError('Exact final controller Git producer required')
    if subprocess.check_output(['git','status','--porcelain'],universal_newlines=True).strip():
        raise ValueError('No uncommitted controller deployment changes')
    ordered,digest,complete=parent(ROOT);old=ordered[-1]['lease']
    if source==old['source']:raise ValueError('Final evaluation controller must be separately recorded')
    resume.verify_worker(old['worker_worktree'],old['source'],source)
    queue=subprocess.check_output(['squeue','-h','-A',g.ACCOUNT,'-o','%i'],universal_newlines=True,timeout=20)
    if queue.strip():raise ValueError('No overlapping project job permitted')
    accounting=subprocess.check_output(['sacct','-n','-X','-P','-A',g.ACCOUNT,'--starttime=2026-11-08',
        '--format=JobIDRaw,JobName%100,Account,State,ExitCode,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=30)
    prior=resume.allocation_carry(accounting,ordered,'0');seconds=walltime(prior)
    storage=storage_check();schedule=forecast(ROOT,seconds,complete)
    backups=backup_proofs(ROOT,old['source'])
    from .matched_rpc_job import prepare_mount_target
    prepare_mount_target(work)
    cmd=command(work,source,digest,complete,seconds)
    CLAIM.mkdir(exist_ok=False)
    (CLAIM/'command.json').write_text(json.dumps(dict(source=source,authorization_sha256=grant,
        command=cmd,paced_authorization_sha256=pacing_grant,
        prior_GPU_seconds=prior,maximum_seconds=seconds,interval=10,
        previous_result_sha256=digest,evaluation_only=complete,storage=storage,schedule=schedule,independent_backups=backups,started_unix=time.time()),indent=2)+'\n')
    try:
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
    except subprocess.TimeoutExpired:
        (CLAIM/'unknown.json').write_text(json.dumps(dict(state='UNKNOWN_DO_NOT_RESUBMIT',source=source))+'\n')
        raise
    (CLAIM/'response.json').write_text(json.dumps(dict(actual_exit=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    if result.returncode or not re.fullmatch(r'[0-9]+(?:;[^\s;]+)?',result.stdout.strip()):
        raise RuntimeError('Final submission failed or ambiguous; never auto-resubmit')
    job=result.stdout.strip().split(';')[0]
    (CLAIM/'job.json').write_text(json.dumps(dict(job_id=job,source=source,actual_sbatch_exit=0))+'\n')
    return job


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    print(submit(p.parse_args().source),flush=True)
