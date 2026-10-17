"""Explicit native teacher supervision beside unchanged four-reference retention."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def teacher_index(kinds,step,reference_index,teacher_kinds):
    from evals.context_component_kind import retention_index
    names=('private','simple','clarify','cancel')
    if reference_index!=retention_index(kinds,step,4,True):
        raise RuntimeError('Native teacher sampling must match the declared retention-kind clock')
    group=[i for i,k in enumerate(teacher_kinds) if k==names[reference_index]]
    if not group:raise RuntimeError('Native teacher coverage omits a required task kind')
    width=len(set(kinds));period=4*width;cycles,offset=divmod(step,period)
    previous=cycles*width+sum(retention_index(kinds,s,4,True)==reference_index for s in range(offset))
    return group[previous%len(group)]


def prepare(examples,spm,voice,retention,args):
    import torch
    from contextspan.model.sequence_convention import persona_prefix
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_native_teacher_data import load
    approval=args.actual_native_teacher_approval;pre=args.actual_native_teacher_preflight
    gate_file=pre/'native-teacher-math-audit.json';gate=json.loads(gate_file.read_text())
    manifest,rows=load(approval,args.actual_native_teacher_scope,args.assets,args.out/'voice.pt',examples,
        persona_prefix(BASE_INSTRUCTIONS,voice,spm),spm)
    native_data=pre/'native-teacher-data.json'
    if (gate['schema']!='dl-context-native-teacher-math-audit-v1' or not gate['passed']
            or gate['optimizer_steps']!=0 or gate['training_approved'] is not False
            or gate['data_sha256']!=sha256(native_data) or json.loads(native_data.read_text())!=manifest
            or gate['data_loader_code_sha256']!=sha256(Path(__file__).with_name('context_native_teacher_data.py'))
            or gate['math_code_sha256']!=sha256(Path(__file__).with_name('context_native_teacher_math.py'))
            or len(gate['cases'])!=len(rows)):
        raise RuntimeError('Native teacher objective lacks its complete matching numerical preflight')
    for row,checked in zip(rows,gate['cases']):
        if (checked['teacher']!=row['teacher'] or not checked['passed']
                or checked['causal_prefix_maximum_error']!=0.
                or set(checked['comparisons'])!={'hidden','text_logits'}
                or any(not 0<=v['maximum_absolute_error']<=.001 for v in checked['comparisons'].values())
                or checked['text_targets']!=int(row['mask'].sum())):
            raise RuntimeError('Native teacher numerical preflight changed or failed')
    if args.actual_native_teacher_coverage=='four-longest':
        ids=[r[2]['id'] for r in retention]
        selected=[next(row for row in rows if row['teacher']['id']==identity) for identity in ids]
    elif args.actual_native_teacher_coverage=='all-approved':selected=rows
    else:raise RuntimeError('Unknown explicit native teacher coverage')
    if {row['teacher']['kind'] for row in selected}!={'private','simple','clarify','cancel'}:
        raise RuntimeError('Native teacher objective lost task-kind coverage')
    packaged=args.out/'approved-native-teacher-math.json'
    if args.reload:
        if sha256(packaged)!=sha256(gate_file):raise RuntimeError('Packaged native teacher audit changed')
    else:packaged.write_bytes(gate_file.read_bytes())
    policy=dict(mode='native-streamed-user-teacher-text-v1',weight=args.actual_retention_teacher_weight,
        coverage=args.actual_native_teacher_coverage,teachers=[row['teacher'] for row in selected],
        native_input_approval_sha256=sha256(approval),teacher_scope_sha256=sha256(args.actual_native_teacher_scope),
        numerical_preflight_sha256=sha256(gate_file),native_teacher_data_sha256=sha256(native_data),
        code_sha256=sha256(Path(__file__)),sampling='same task-kind exposure as unchanged four-reference retention; rotate every teacher within that kind',
        reference_kl_and_hidden_objective_unchanged=True,assistant_acoustics='approved teacher conditioning; no audio loss',
        prefix_loss_targets=0,span_loss_targets=0,user_loss_targets=0,audio_loss_targets=0)
    batches=[(row['inputs'].cuda(),row['mask'].cuda(),row['targets'].cuda(),row['teacher']) for row in selected]
    return policy,batches
