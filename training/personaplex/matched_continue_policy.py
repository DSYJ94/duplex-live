"""Verify reused numerical proofs plus NEW actual restore/state terminals.

No data/optimizer/acceptance edits; no model or scheduler mutation. Remaining
work decreases only from actual successfully checkpointed and restored steps.
"""
import json,math,time
from pathlib import Path
from . import matched_guard as g
from .matched_reuse import ORIGIN,ORIGINAL_ROOT,DATA_SHA,preflight

PACKET_SHA='3a874ddc35de0b4c67bde8ce106e4c536a225ff9790be6276a729d4e3204621b'


def read_pinned(path,digest):
    path=Path(path)
    if path.is_symlink() or g.sha(path)!=digest:raise ValueError('Continuation proof changed')
    return json.loads(path.read_text())


def restored_entry(row,source,data_sha,original):
    value=read_pinned(row['restore'],row['restore_sha256'])
    terminal=json.loads(Path(row['restore_terminal']).read_text())
    checkpoint=Path(row['checkpoint']);reference=json.loads(Path(str(checkpoint)+'.native-reference.json').read_text())
    for key,expected in [('source',source),('lineage_source',ORIGIN),('base',row['base']),
            ('capacity',row['capacity']),('data_report_sha256',data_sha),('checkpoint_sha256',row['checkpoint_sha256'])]:
        if value.get(key)!=expected:raise ValueError('Fresh restore lineage/source/data/state identity differs: '+key)
    if (value.get('passed') is not True or value.get('actual_CUDA') is not True
            or value.get('all_Adam_RNG_OPAL_fields_exact') is not True or value.get('native_logits_exact') is not True
            or terminal.get('actual_exit_code')!=0 or terminal.get('source')!=source
            or terminal.get('base')!=row['base'] or terminal.get('capacity')!=row['capacity']
            or terminal.get('stage')!='restore' or terminal.get('execution_id')!=value.get('execution_id')
            or value['execution_id']==reference['execution_id']
            or value['reference_execution_id']!=reference['execution_id']
            or reference['checkpoint_sha256']!=row['checkpoint_sha256']
            or (reference['source'] not in (ORIGIN,source) and not row.get('quota_interrupted_state'))):
        raise ValueError('A distinct actual successful native restore is required')
    step=value['restored_training_step']
    if type(step) is not int or not 0<=step<=196 or reference['step']!=step:
        raise ValueError('Actual restored training step differs')
    if step==0:
        if row.get('quota_interrupted_state'):raise ValueError('Interrupted recovery cannot claim zero-update state')
        if str(checkpoint)!=original['original_checkpoint'] or row['checkpoint_sha256']!=original['checkpoint_sha256']:
            raise ValueError('Only original independently admitted zero-update initialization allowed')
    elif row.get('quota_interrupted_state'):
        from .matched_interrupted import validate_interrupted
        validate_interrupted(row,reference,step)
    else:
        validate_training_state(row['previous_training_terminal'],row['previous_training_child_terminal'],
            checkpoint,row['checkpoint_sha256'],step,source,row,reference)
    return step


def validate_training_state(path,child_path,checkpoint,digest,step,source,row,reference):
    value=json.loads(Path(path).read_text());child=json.loads(Path(child_path).read_text())
    if (value['source']!=source or value['total_lineage_updates']!=step
            or value['checkpoint_sha256']!=digest or reference['checkpoint_sha256']!=digest
            or reference['step']!=step or reference['source']!=source
            or child['actual_exit_code']!=0 or child['source']!=source
            or child['base']!=row['base'] or child['capacity']!=row['capacity']
            or child['stage']!='train' or child['execution_id']!=reference['execution_id']):
        raise ValueError('Only actual successful checkpointed training can reduce remaining work')
    expected=value.get('final_checkpoint') if value.get('completed') else value.get('checkpoint')
    if str(checkpoint)!=expected:raise ValueError('Training checkpoint path changed')
    if step==196 and (value.get('completed') is not True or value.get('complete_epochs')!=4):
        raise ValueError('Full196/four epochs required for completion')
    if step<196 and value.get('completed') is not False:raise ValueError('Partial progress cannot claim completion')


def evaluate(a):
    packet=read_pinned(a.joint_admission,a.joint_admission_sha)
    if (packet.get('schema')!='dl-moshi-training-only-joint-v1' or packet.get('source')!=a.source
            or packet.get('lineage_source')!=ORIGIN or packet.get('data_report_sha256')!=a.inputs_sha
            or a.inputs_sha!=DATA_SHA or packet.get('original_joint_packet_sha256')!=PACKET_SHA):raise ValueError('Exact training-only packet required')
    original=preflight(ORIGINAL_ROOT,PACKET_SHA)
    originals={(r['base'],r['capacity']):r for r in original['lineages']}
    entries=packet['entries']
    if len(entries)!=4 or {(r['base'],r['capacity']) for r in entries}!=set(originals):
        raise ValueError('Complete four-lineage matched cohort required')
    estimates={b:0. for b in ('personaplex','context-spanning')};steps={}
    for row in entries:
        key=(row['base'],row['capacity']);old=originals[key]
        step=restored_entry(row,a.source,a.inputs_sha,old)
        # The current window may already have completed the earlier serial arm.
        completed=Path(row['training_terminal'])
        if completed.exists():
            terminal=json.loads(completed.read_text());checkpoint=Path(terminal['final_checkpoint'])
            ref=json.loads(Path(str(checkpoint)+'.native-reference.json').read_text())
            final=json.loads(Path(row['final_restore']).read_text())
            actual=json.loads(Path(row['final_restore_terminal']).read_text())
            if (final.get('source')!=a.source or final.get('lineage_source')!=ORIGIN
                    or final.get('base')!=row['base'] or final.get('capacity')!=row['capacity']
                    or final.get('restored_training_step')!=196 or final.get('passed') is not True
                    or final.get('actual_CUDA') is not True or final.get('data_report_sha256')!=a.inputs_sha
                    or actual.get('stage')!='restore' or actual.get('base')!=row['base'] or actual.get('capacity')!=row['capacity']
                    or final.get('native_logits_exact') is not True or final.get('all_Adam_RNG_OPAL_fields_exact') is not True
                    or final.get('checkpoint_sha256')!=terminal['checkpoint_sha256']
                    or final.get('reference_execution_id')!=ref['execution_id'] or final.get('execution_id')==ref['execution_id']
                    or actual.get('actual_exit_code')!=0 or actual.get('source')!=a.source
                    or actual.get('execution_id')!=final.get('execution_id') or g.sha(checkpoint)!=terminal['checkpoint_sha256']):
                raise ValueError('Completed state needs its actual independent final restore')
            validate_training_state(completed,row['training_child_terminal'],checkpoint,terminal['checkpoint_sha256'],196,a.source,row,ref)
            step=196
        remaining=old['estimated_complete196_seconds']*(196-step)/196
        if 0<step<196:remaining+=600  # conservative per-resume fixed overhead
        if not math.isfinite(remaining) or remaining<0:raise ValueError('Finite remaining schedule required')
        estimates[row['base']]+=remaining;steps['/'.join(key)]=step
    lease=g.check(a.lease)
    if not lease.get('continuation_authorization_sha256'):raise ValueError('Training-only restart requires the recorded amendment')
    available=g.phase_limit(lease)-lease['prior_GPU_seconds']-2*(time.time()-lease['start_unix'])
    required=2*max(estimates.values())+21600
    if required>available:raise RuntimeError('Complete remaining matched schedule exceeds carried authorized GPU time')
    return dict(estimated_serial_seconds_per_base=estimates,actual_verified_steps=steps,
        required_GPU_seconds=required,available_GPU_seconds=available,all_four_native_actual_terminals=True,
        reused_original_source=ORIGIN,new_actual_restore_producer=a.source)
