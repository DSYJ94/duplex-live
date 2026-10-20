"""Derive complete49 assistant-only audio targets from existing approved histories.

This grants no optimizer approval. Retain every original28 actual component,
all25 repaired old native teachers, all24 new teachers and KL1/hidden10.
The rejected standalone joint pilot is not a parent or a data source.
"""
import hashlib
import json
from pathlib import Path
from evals.moshi_model_download import sha256

ROOT=Path('/projects/u6si/duplex-live')
GROUPS=(
 dict(name='old25',count=25,cache='native-old25-codes-7543730',scope='native-old25-native-scope-7543889',
      approval='native-old25-native-approval-7544805',numerical='native-old25-native-math-7545975'),
 dict(name='new24',count=24,cache='native-v2-codes-7545905',scope='native-v2-native-scope-7545907',
      approval='native-v2-native-approval-7545909',numerical='native-v2-native-math-7546608'))


def digest(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def complete_inventory(records):
    ids=[r['teacher']['id'] for r in records]
    if (len(records)!=49 or len(set(ids))!=49
            or any(sum(r['group']==g['name'] for r in records)!=g['count'] for g in GROUPS)
            or any(r['teacher']['source_split']!='train' or r['teacher']['source_model']!='gpt-live-1'
                or r['assistant_audio_targets_per_head']<1 or r['user_audio_loss_targets']!=0
                or r['prefix_loss_targets']!=0 or r['supplied_span_loss_targets']!=0 for r in records)):
        raise ValueError('Assistant-audio audit requires all25+24 approved TRAIN teachers with strict ownership')
    return True


def load(assets,voice_file,spm,voice):
    import torch
    from contextspan.model.sequence_convention import persona_prefix,assemble_training_sequence
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_training_data import load_training
    from evals.context_native_teacher_data import load as approved_native
    from evals.context_observed_teacher import canonical
    from evals.context_retention_teacher import select_teachers
    from evals.context_native_v2_teacher_objective import validate_old25_lineage
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.context_native_learning_curve import PINS
    pi,parent=verify_checkpoint(ROOT/'runs/context-training-proof-7510331')
    ci,current=verify_checkpoint(ROOT/'runs/context-training-proof-7558824')
    if ci['checkpoint_sha256']!=PINS[1]['checkpoint'] or ci['training_state_sha256']!=PINS[1]['state']:
        raise ValueError('Changed reference for repaired25/additional24 approval lineage')
    validate_old25_lineage(parent['retention_teacher_text'],current['retention_teacher_text'])
    data_file=ROOT/'runs/context-training-proof-7510331/data.json';actual=json.loads(data_file.read_text())
    if (len(actual['components'])!=28 or current['data_sha256']!=parent['data_sha256']
            or sha256(data_file)!=parent['data_sha256'] or sha256(voice_file)!=parent['voice_sha256']
            or current['voice_sha256']!=parent['voice_sha256']):
        raise ValueError('Original28 components or native520 voice changed')
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm);initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
    batches=[];records=[];groups=[]
    for group in GROUPS:
        policy=current['retention_teacher_text'] if group['name']=='old25' else current['additional_native_v2']['teacher']
        runs=ROOT/'runs';cache=runs/group['cache'];scope=runs/group['scope']/'report.json';approval=runs/group['approval']/'report.json';numerical=runs/group['numerical']
        gate_file=numerical/'native-teacher-math-audit.json';data=numerical/'native-teacher-data.json'
        if (sha256(scope)!=policy['teacher_scope_sha256'] or sha256(approval)!=policy['native_input_approval_sha256']
                or sha256(gate_file)!=policy['numerical_preflight_sha256'] or sha256(data)!=policy['native_teacher_data_sha256']):
            raise ValueError('Changed complete native teacher input/numerical approval')
        examples,_=load_training([cache],spm,prefix,text_padding='legacy-epad0',request_policy='native-prelead-epad')
        manifest,approved=approved_native(approval,scope,assets,voice_file,examples,prefix,spm)
        encoded=json.dumps(manifest,indent=2)+'\n';gate=json.loads(gate_file.read_text())
        if (len(approved)!=group['count'] or not gate['passed'] or gate['schema']!='dl-context-native-teacher-math-audit-v2'
                or gate['optimizer_steps']!=0 or hashlib.sha256(encoded.encode()).hexdigest()!=policy['native_teacher_data_sha256']
                or gate['data_sha256']!=policy['native_teacher_data_sha256'] or len(gate['cases'])!=group['count']
                or [c['teacher'] for c in gate['cases']]!=[r['teacher'] for r in approved]
                or any(not c['passed'] or c['causal_prefix_maximum_error']!=0.
                    or any(not 0<=v['maximum_absolute_error']<=.001 for v in c['comparisons'].values()) for c in gate['cases'])):
            raise ValueError('Complete unchanged native numerical approvals required')
        selected={e['record']['id']:canonical(e,spm) for e in select_teachers(examples,True)}
        for row,source in zip(approved,manifest['cases']):
            e=selected[row['teacher']['id']];saved=torch.load(Path(source['capture'])/'teacher-engine-inputs.pt',weights_only=True,map_location='cpu')
            clean=e['clean_codes'].clone();clean[9:17]=saved['encoded_user_codes'][0]
            assembled,tm,am=assemble_training_sequence(clean,torch.ones(clean.shape[1],dtype=torch.bool),e['record']['spans'],spm)
            codes=torch.cat([prefix,assembled],dim=1)[None]
            tm=torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),tm])[None]
            am=torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),am])[None]
            delayed=torch.cat([initial,_delay_sequence([0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1],codes,initial)],dim=2)[:,:,:-1]
            if (not torch.equal(delayed,row['inputs']) or not torch.equal(codes[:,:9],e['codes'][None,:9])
                    or not torch.equal(tm[0],row['mask']) or not torch.equal(codes[0,0],row['targets'])
                    or not torch.equal(am[0],e['audio_mask']) or tm[:,:prefix.shape[1]].any()
                    or am[:,:prefix.shape[1]].any() or (am&~tm).any()):
                raise ValueError('New audio ownership changed approved input/text/prefix/span masks')
            targets=codes[:,1:9].transpose(1,2)[am]
            if (targets<0).any() or (targets>=2048).any() or not am.any():raise ValueError('Invalid assistant acoustic targets')
            record=dict(group=group['name'],teacher=row['teacher'],source_capture=source['capture'],
                native_input_sha256=digest(delayed),codes_sha256=digest(codes),text_mask_sha256=digest(tm),
                audio_mask_sha256=digest(am),assistant_targets_sha256=digest(targets),
                assistant_audio_targets_per_head=int(am.sum()),user_audio_loss_targets=0,prefix_loss_targets=0,supplied_span_loss_targets=0)
            records.append(record);batches.append(dict(codes=codes,text_mask=tm,audio_mask=am,delayed_inputs=delayed,record=record))
        groups.append(dict(group=group,scope_sha256=sha256(scope),approval_sha256=sha256(approval),numerical_sha256=sha256(gate_file),native_data_manifest=manifest))
    complete_inventory(records)
    proof=dict(schema='dl-native-all49-assistant-audio-data-v1',cases=records,groups=groups,parent=pi,
        approval_lineage_reference=ci,original28_actual_components=actual['components'],original28_data_sha256=sha256(data_file),
        old25_and_new24_text_input_and_masks_exact=True,original_audio_words_receipts_and_folds_unchanged=True,
        only_assistant_eight_audio_codebooks_are_auxiliary_targets=True,user_audio_loss_targets=0,prefix_loss_targets=0,
        supplied_span_loss_targets=0,heldout_labels=0,owner_data_used=False,optimizer_updates=0,training_approved=False,
        required_original_objectives=dict(actual_histories=28,old_native_teachers=25,new_native_teachers=24,kl_weight=1.,hidden_mse_weight=10.),
        rejected_joint_7530408_extended=False,loader_sha256=sha256(Path(__file__)))
    return proof,batches
