"""Read-only parent admission and explicit adaptive source lineage.

No scheduler mutation, submission, model loading or checkpoint deletion.
The host supplies a fresh terminal provider observation before model startup.
"""
import argparse,copy,json,re,time
from pathlib import Path
from .adaptive_policy import Policy,digest,sha
from .adaptive_evidence import read,backup


def prepare(parent_root,parent_job,parent_backup,target_root,target_backup,source,provider):
    parent_root=Path(parent_root).resolve();target_root=Path(target_root).resolve()
    parent_backup=Path(parent_backup).resolve();target_backup=Path(target_backup).resolve()
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('Exact consumer source required')
    if provider.get('JobId')!=str(parent_job) or provider.get('JobState')!='COMPLETED' or provider.get('Account')!='brics.u6ada':
        raise ValueError('Parent must be provider COMPLETED; never overlap or interrupt a valid worker')
    if parent_root==target_root or parent_root in target_root.parents or target_root in parent_root.parents:
        raise ValueError('Separate sibling output namespace required')
    if parent_backup==target_backup or parent_backup in target_backup.parents or target_backup in parent_backup.parents:
        raise ValueError('Separate sibling backup namespace required')
    folder=parent_root/('job-'+str(parent_job));host,host_hash=read(folder/'host-terminal.json')
    terminal,terminal_hash=read(folder/'train'/'terminal.json')
    if host.get('success')is not True or host.get('training')!=terminal or terminal.get('training_complete')is not False or terminal.get('continuation_ready')is not True:
        raise ValueError('Successful cooperative continuation required, not failed or completed training')
    step=terminal['actual_new_step'];checkpoint=Path(terminal['checkpoint']).resolve()
    if type(step)is not int or not 0<step<196 or checkpoint.parent!=parent_root/'checkpoints':raise ValueError('Exact unfinished native phase checkpoint required')
    if checkpoint.name!='step-%04d.pt'%step or not sha(terminal.get('checkpoint_sha256')):raise ValueError('Latest terminal checkpoint binding required')
    later=[p for p in checkpoint.parent.glob('step-*.pt')if re.fullmatch('step-[0-9]{4}.pt',p.name)and int(p.stem[5:])>step]
    if later:raise ValueError('Newer checkpoint exists; do not roll back or replay saved updates')
    receipt=parent_backup/('step-%04d'%step)/'backup.json'
    backed_step,checkpoint_hash,backup_hash=backup(receipt)
    if backed_step!=step or checkpoint_hash!=terminal['checkpoint_sha256']:raise ValueError('Independent latest backup differs')
    metadata,metadata_hash=read(Path(str(checkpoint)+'.sha256.json'))
    if metadata['sha256']!=checkpoint_hash or metadata['step']!=step or metadata['source']!=terminal['source']:raise ValueError('Parent checkpoint publication differs')
    boundary=step//49*49
    if not boundary:raise ValueError('At least one complete DEV epoch required before adaptive continuation')
    admission,admission_hash=read(parent_root/'admission.json')
    if not admission.get('actual_CUDA')or len(admission.get('all_four_ranks',[]))!=4 or admission.get('total_new_updates')!=196:raise ValueError('Original complete four-rank admission required')
    return dict(schema='dl-adaptive-handover-v1',source=source,parent_source=terminal['source'],parent_job=str(parent_job),
        parent_root=str(parent_root),parent_backup=str(parent_backup),root=str(target_root),backup=str(target_backup),
        parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=checkpoint_hash,parent_backup_receipt=str(receipt),
        parent_backup_sha256=backup_hash,parent_metadata_sha256=metadata_hash,parent_terminal_sha256=terminal_hash,
        parent_host_terminal_sha256=host_hash,admission_sha256=admission_hash,step=step,boundary=boundary,
        policy_sha256=Policy().identity,new_optimizer_updates=0)


def validate_plan(plan,provider):
    fresh=prepare(plan['parent_root'],plan['parent_job'],plan['parent_backup'],plan['root'],plan['backup'],plan['source'],provider)
    if fresh!=plan:raise ValueError('Immutable handover plan no longer matches actual parent')
    return plan


def validate_resume(plan,checkpoint,provider):
    root=Path(plan['root']);checkpoint=Path(checkpoint).resolve();candidates=[]
    if checkpoint.parent!=(root/'checkpoints').resolve():raise ValueError('Resume checkpoint outside its namespace')
    for path in root.glob('job-*/host-terminal.json'):
        host,_=read(path);terminal=host.get('training',{})
        if host.get('success')is True and terminal.get('continuation_ready')is True:
            candidates.append((terminal['actual_new_step'],path.parent.name[4:],terminal))
    if not candidates:raise ValueError('No successful trained adaptive continuation exists; no startup retry')
    step,job,terminal=max(candidates,key=lambda r:r[0])
    if (step<=plan['step']or terminal.get('training_complete')is not False or terminal.get('source')!=plan['source']
            or Path(terminal['checkpoint']).resolve()!=checkpoint or provider.get('JobId')!=job or provider.get('JobState')!='COMPLETED'
            or provider.get('Account')!='brics.u6ada'):raise ValueError('Latest successful terminal adaptive state required; never replay or overlap')
    for path in checkpoint.parent.glob('*.pt'):
        match=re.fullmatch(r'(?:adaptive-)?step-([0-9]{4})\.pt',path.name)
        if match and int(match[1])>step:raise ValueError('Newer saved adaptive updates exist; no rollback')
    folder=checkpoint.stem if checkpoint.stem.startswith('adaptive-')else 'step-%04d'%step
    backed_step,checkpoint_hash,_=backup(Path(plan['backup'])/folder/'backup.json')
    if backed_step!=step or checkpoint_hash!=terminal['checkpoint_sha256']:raise ValueError('Adaptive resume backup differs')
    return terminal


def restore_identity(saved,current,plan):
    """Return original identity ONLY after proving all non-source math is equal."""
    if saved.get('source')!=plan['parent_source']or current.get('source')!=plan['source']:
        raise ValueError('Exact predecessor and consumer source required')
    before,after=copy.deepcopy(saved),copy.deepcopy(current)
    before.pop('source');after.pop('source')
    for value in (before,after):value['capture_provenance'].pop('current_training_source')
    if before!=after:raise ValueError('Native recipe/data/model/authorization differs; source-only transition required')
    return saved


def target_identity(current,plan):
    result=copy.deepcopy(current)
    result['adaptive_lineage']=dict(schema=plan['schema'],parent_source=plan['parent_source'],
        parent_checkpoint_sha256=plan['parent_checkpoint_sha256'],parent_step=plan['step'],
        parent_backup_sha256=plan['parent_backup_sha256'],policy_sha256=plan['policy_sha256'],plan_sha256=digest(plan))
    return result


def native_restore_packet(*,source,step,execution,pid,rank,value,optimizer_exact,rng_exact,recipe_exact,diagnostics_exact,fingerprint_exact,common_adam_exact):
    """Every native field must be independently checked, never inferred from CPU tests."""
    checks=dict(optimizer=optimizer_exact,rng=rng_exact,recipe=recipe_exact,diagnostics=diagnostics_exact,
                native_fingerprint=fingerprint_exact,common_adam=common_adam_exact)
    # Linux PIDs can recur on a different node/allocation. The freshly generated
    # execution identity, not global PID uniqueness, establishes this new run.
    if not all(v is True for v in checks.values())or value['writer_execution']==execution:
        raise ValueError('Fresh exact native restoration required before any transition or learning')
    return dict(source=source,actual_CUDA=True,step=step,execution=execution,pid=pid,rank=rank,
        reference_execution=value['writer_execution'],reference_pid=value['writer_pid'],checks=checks,
        native_data_qualified=True,native_fresh_restore_passed=True,all_Adam_RNG_OPAL_native_fields_exact=True,new_optimizer_updates=0)


def all_native_restored(rows,step):
    if len(rows)!=4 or {r['rank']for r in rows}!=set(range(4))or any(r['step']!=step or not r['native_fresh_restore_passed']or not all(r['checks'].values())or r['new_optimizer_updates']!=0 for r in rows):
        raise ValueError('Every actual native rank must pass before enabling the controller')
    if len({r['pid']for r in rows})!=4 or len({r['source']for r in rows})!=1:raise ValueError('Distinct native rank processes and unanimous source required')
    return dict(passed=True,step=step,rows=rows,new_optimizer_updates=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--provider-json',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--resume-checkpoint',type=Path)
    for key in ('parent-root','parent-job','parent-backup','root','backup','source'):p.add_argument('--'+key)
    a=p.parse_args();observation=json.loads(a.provider_json.read_text())
    if not 0<=time.time()-observation['observed_unix']<=90:raise ValueError('Fresh terminal provider evidence required')
    if a.prepare:
        plan=prepare(a.parent_root,a.parent_job,a.parent_backup,a.root,a.backup,a.source,observation['job'])
        with a.plan.open('x')as stream:json.dump(plan,stream,indent=2);stream.write('\n')
    else:
        plan=validate_plan(json.loads(a.plan.read_text()),observation['job'])
        if a.resume_checkpoint:validate_resume(plan,a.resume_checkpoint,observation['resume_job'])
    print(json.dumps(dict(status='safe_handover_admitted',parent_job=plan['parent_job'],step=plan['step'],plan_sha256=digest(plan),new_optimizer_updates=0)))


if __name__=='__main__':main()
