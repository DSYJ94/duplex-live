"""Read-only fixed-history comparison of native520, clock8 and clock40.

All 81 inputs are approved TRAIN/reference inputs. Development recordings are
never loaded. No likelihood measurement constitutes native behavior approval.
"""
import json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.context_native_retention_drift import differences, validate_inventory
from evals.context_warm_ordinary_clock import DATA_SHA256

PINS = (
    dict(job='7556562',steps=8,source='3061dcba1de571d2dad58d466f7c3f3ca43e41ec',
         checkpoint='1ad464e2b5ca2258ef777f013ccdfe057721a65cd974e17a32a1070b1f05016b',
         state='3303bf1624cf3c36ff071e9fb69a239b3b93fda6a0d914db67848b7543cad675',
         reload='440b665312b3125f57dd26531212db5f2c90e8b2d5a08dcf1d04b5e18b7c8b9a',
         report='9785b624a31004638662efa80013704f8ad1ed65e7a102387aa7061e96acd3cf'),
    dict(job='7558824',steps=40,source='99cc7394ebde78e2762d92813dddbc7a3feaaaed',
         checkpoint='b4361124025b339c02e354b7729699a520da74400927eb1aaa9a5029136264b6',
         state='9712d0d858d638630d02fbb2f2909da822c0c3f9d7b47202a1637a1785ee6c9d',
         reload='20c73529c9dacf72e62b3d8f2f3ee2ae0cf51310633d77aadba429d6bacb0492',
         report='6fde1c6be34ede4c857efdd57361f7a9cf61cca47f613f26bb7b2d5909d3bea3'))


def validate_lineage(parent, children, parent_checkpoint, data_hash):
    if parent['optimizer_steps']!=520 or data_hash!=DATA_SHA256 or len(children)!=2:
        raise ValueError('Learning curve requires exact native520 and both declared children')
    for pin,(identity,child) in zip(PINS,children):
        if (identity['checkpoint_sha256']!=pin['checkpoint']
                or identity['training_state_sha256']!=pin['state']
                or identity['reload_report_sha256']!=pin['reload'] or identity['report_sha256']!=pin['report']
                or child['optimizer_steps']!=pin['steps'] or child['repo_commit']!=pin['source']
                or not child['passed'] or child['data_sha256']!=data_hash
                or parent['data_sha256']!=data_hash
                or any(child[k]!=parent[k] for k in ['frozen_hashes','voice_sha256','assets_manifest_sha256'])
                or child['warm_start']['checkpoint_sha256']!=parent_checkpoint
                or not child['additional_native_v2']['complete24_per8']
                or child['additional_native_v2']['learning_rate']!=2e-6
                or child['warm_ordinary_retention']['weight']!=64
                or child['warm_ordinary_clock_retention']['weight']!=64):
            raise ValueError('Learning curve state, source, approved data or retained recipe changed')
    eight,forty=[row[1] for row in children]
    if (forty['initial_optimizer_steps']!=8 or forty['updates_this_run']!=32
            or Path(forty['resumed_from_training_state']).parent.name!='context-training-proof-7556562'
            or forty['source_transition']['training_state_sha256']!=PINS[0]['state']
            or forty['source_transition']['from_repo_commit']!=PINS[0]['source']
            or forty['source_transition']['to_repo_commit']!=PINS[1]['source']
            or any(eight[k]!=forty[k] for k in ['retention_teacher_text','additional_native_v2',
                'warm_ordinary_retention','warm_ordinary_clock_retention','native_protocol'])):
        raise ValueError('The forty-update state is not the exact unchanged-recipe continuation')


def target_metrics(z,mask,target):
    import torch
    from torch.nn import functional as F
    logits=z[0,0,mask].float();y=target[mask]
    if not len(y) or not torch.isfinite(logits).all():raise ValueError('Invalid approved target rows')
    ce=F.cross_entropy(logits,y,reduction='none');weight=torch.where(y==3,.5,1.)
    pred=logits.argmax(-1);content=y>=5;pad=y==3
    row=dict(frames=len(y),half_pad_nll=float((ce*weight).sum()/weight.sum()),
        top1_matches=int((pred==y).sum()),nonpad_frames=int((~pad).sum()),
        spoken_frames=int(content.sum()),pad3_frames=int(pad.sum()),
        spoken_top1_matches=int((pred[content]==y[content]).sum()),pad3_top1_matches=int((pred[pad]==3).sum()))
    if content.any():
        first=int(content.nonzero()[0]);row['first_spoken_target_probability']=float(logits[first].softmax(-1)[y[first]])
    if not __import__('math').isfinite(row['half_pad_nll']):raise ValueError('Nonfinite fixed-history loss')
    return row


def run(lm,learned,actual,old,new,retention,references,audits,args,metadata,forward):
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.moshi_training_smoke import tensor_hashes
    validate_inventory(old,new,retention)
    if len(actual)!=28 or len(audits)!=28 or any(not a['passed'] for a in audits):
        raise ValueError('Learning curve must include all28 independently approved actual histories')
    if Path(args.warm_start_proof).name!='context-training-proof-7510331':
        raise ValueError('Only accepted native520 can be the diagnostic parent')
    parent_identity,parent=verify_checkpoint(args.warm_start_proof)
    runs=Path(args.warm_start_proof).parent
    children=[verify_checkpoint(runs/('context-training-proof-'+p['job'])) for p in PINS]
    validate_lineage(parent,children,parent_identity['checkpoint_sha256'],metadata['data_sha256'])
    for _,c in children:
        if (c['retention_teacher_text']!=metadata['recipe']['retention_teacher_text']
                or c['additional_native_v2']!=metadata['recipe']['additional_native_v2']
                or metadata['recipe']['retention_kl_weight']!=1.
                or metadata['recipe']['retention_hidden_mse_weight']!=10.):
            raise ValueError('Diagnostic input masks/teacher targets or legacy losses changed')
    states=[load_file(runs/('context-training-proof-'+p['job'])/'adapters.safetensors',device='cuda') for p in PINS]
    saved={n:p.detach().clone() for n,p in learned.items()};before=tensor_hashes(learned.items())
    for state in states:
        if set(state)!=set(saved) or any(state[n].shape!=p.shape or state[n].dtype!=p.dtype
                or not torch.isfinite(state[n]).all() for n,p in learned.items()):
            raise ValueError('Changed adapter layout or nonfinite saved state')
    rows=[]
    inputs=[('actual28',i,*b,dict(id='actual-'+str(i)+'-'+a['scenario'],kind=a.get('target_kind','actual')))
        for i,(b,a) in enumerate(zip(actual,audits))]
    inputs += [('old25',i,*b) for i,b in enumerate(old)]+[('new24',i,*b) for i,b in enumerate(new)]
    inputs += [('legacy4',i,x,m,None,r) for i,(x,m,r) in enumerate(retention)]
    if len(inputs)!=81:raise ValueError('Complete81 comparison required')
    def install(state):
        for n,p in learned.items():p.copy_(state[n])
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            for group,index,x,mask,target,record in inputs:
                install(saved);ph,pz=forward(x);check=differences(ph,pz,ph,pz,mask)
                if check['warm_parent_kl']!=0 or check['warm_parent_hidden_mse']!=0 or check['text_top1_matches']!=check['frames']:
                    raise ValueError('Native520 self-comparison is not exact')
                row=dict(group=group,index=index,id=record['id'],kind=record['kind'],self_comparison=check,
                    approved_tensor_hashes=tensor_hashes([('inputs',x),('mask',mask)]+([] if target is None else [('targets',target)])),children=[])
                if target is not None:row['native520_targets']=target_metrics(pz,mask,target)
                for pin,state in zip(PINS,states):
                    install(state);ch,cz=forward(x);result=dict(job=pin['job'],optimizer_steps=pin['steps'],**differences(ph,pz,ch,cz,mask))
                    if target is not None:
                        result['targets']=target_metrics(cz,mask,target)
                        for label,m in [('nonpad',mask & (target!=3)),('pad3',mask & (target==3))]:
                            result[label]=differences(ph,pz,ch,cz,m) if m.any() else None
                    else:
                        base_h,prob=references[index]
                        result['legacy_unadapted_kl_parent']=float(F.kl_div(pz[0,0,mask].float().log_softmax(-1),prob,reduction='batchmean'))
                        result['legacy_unadapted_kl_child']=float(F.kl_div(cz[0,0,mask].float().log_softmax(-1),prob,reduction='batchmean'))
                        result['legacy_unadapted_hidden_mse_child']=float((ch[:,mask].float()-base_h.float()).square().mean())
                    row['children'].append(result)
                rows.append(row);print(json.dumps(row),flush=True)
    finally:
        with torch.no_grad():install(saved)
        for p in learned.values():p.grad=None
    if tensor_hashes(learned.items())!=before:raise RuntimeError('Read-only comparison changed native520 parameters')
    if tensor_hashes((n,p) for n,p in lm.named_parameters() if n not in learned)!=metadata['frozen_hashes']:
        raise RuntimeError('Read-only comparison changed frozen base')
    report=dict(schema='dl-native-fixed-history-learning-curve-v1',metadata=metadata,parent_identity=parent_identity,
        children=[dict(pin=p,verified_identity=i) for p,(i,_) in zip(PINS,children)],cases=rows,
        complete_actual_histories=28,complete_old_teachers=25,complete_new_teachers=24,complete_legacy_references=4,
        optimizer_updates=0,new_checkpoint_written=False,parent_parameters_restored_exact=True,
        frozen_base_parameters_unchanged=True,source_code_sha256=sha256(Path(__file__)),
        development_recordings_loaded=False,evaluation_outputs_used_for_training=False,
        training_approved=False,candidate_adopted=False,native_behavior_evidence=False,
        interpretation='Pointwise comparison on approved fixed histories only; no causal attribution or native behavioral qualification')
    (args.out/'native-learning-curve-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete_inputs=81,optimizer_updates=0,parent_parameters_restored_exact=True)),flush=True)
