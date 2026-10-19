"""Read-only parent/candidate comparison on all approved teacher inputs and old references.

This never trains on evaluation outputs, changes legacy retention, or authorizes
another optimizer run. Teacher words, audio, input masks and approvals stay intact.
"""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def validate_lineage(parent, child, parent_checkpoint, data_hash):
    if (parent['optimizer_steps']!=520 or not child['passed'] or child['optimizer_steps']!=8
            or not child['additional_native_v2']['complete24_per8']
            or child['warm_start']['checkpoint_sha256']!=parent_checkpoint
            or child['data_sha256']!=parent['data_sha256'] or child['data_sha256']!=data_hash
            or child['frozen_hashes']!=parent['frozen_hashes']
            or child['voice_sha256']!=parent['voice_sha256']):
        raise ValueError('Drift diagnostic requires accepted native520 and its exact restored full24 eight-update child')


def validate_inventory(old, new, retention):
    if (len(old)!=25 or len(new)!=24 or len(retention)!=4
            or len({r[3]['id'] for r in old})!=25
            or len({r[2]['id'] for r in retention})!=4
            or {r[3]['id'] for r in new}!={'teacher-native2-train-%03d'%i for i in range(24)}):
        raise ValueError('Complete approved25/24 teacher and original four-reference inputs required')


def differences(parent_h, parent_z, child_h, child_z, mask):
    import torch
    a=parent_z[0,0,mask].float().log_softmax(-1)
    b=child_z[0,0,mask].float().log_softmax(-1)
    frames=int(mask.sum())
    if not frames:raise ValueError('Empty comparison mask')
    matches=int((a.argmax(-1)==b.argmax(-1)).sum())
    result=dict(warm_parent_kl=float((a.exp()*(a-b)).sum(-1).mean()),
        warm_parent_hidden_mse=float((parent_h[:,mask].float()-child_h[:,mask].float()).square().mean()),
        text_top1_matches=matches,text_top1_agreement=matches/frames,frames=frames)
    if not all(__import__('math').isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite drift diagnostic')
    return result


def run(lm, learned, old, new, retention, references, args, metadata, forward):
    import torch
    from torch.nn import functional as F
    from safetensors.torch import load_file
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.context_teacher_text_balance import loss as teacher_loss
    from evals.moshi_training_smoke import tensor_hashes
    validate_inventory(old,new,retention)
    frame_counts=sorted({int(b[1].sum()) for b in old+new+retention})
    rounding=[dict(frames=n,legacy_float_mean=float(torch.ones(n,device='cuda',dtype=torch.float32).mean())) for n in frame_counts]
    print(json.dumps(dict(identity_mean_reproduction=rounding)),flush=True)
    proof=Path(args.native_v2_retention_drift_proof)
    identity,child=verify_checkpoint(proof)
    parent=json.loads((args.warm_start_proof/'report.json').read_text())
    validate_lineage(parent,child,sha256(args.warm_start_proof/'adapters.safetensors'),metadata['data_sha256'])
    candidate=load_file(proof/'adapters.safetensors',device='cuda')
    saved={name:p.detach().clone() for name,p in learned.items()}
    if set(candidate)!=set(saved):raise ValueError('Incomplete candidate adapter parameters')
    for name,p in learned.items():
        if candidate[name].shape!=p.shape or candidate[name].dtype!=p.dtype or not torch.isfinite(candidate[name]).all():
            raise ValueError('Invalid candidate adapter tensor')
    before=tensor_hashes(learned.items());rows=[]
    def install(state):
        for name,p in learned.items():p.copy_(state[name])
    batches=[('old25',i,*b) for i,b in enumerate(old)]+[('new24',i,*b) for i,b in enumerate(new)]
    batches += [('legacy-reference',i,inputs,mask,None,record) for i,(inputs,mask,record) in enumerate(retention)]
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            for group,index,inputs,mask,target,record in batches:
                install(saved);ph,pz=forward(inputs)
                check=differences(ph,pz,ph,pz,mask)
                if check['warm_parent_kl']!=0 or check['warm_parent_hidden_mse']!=0 or check['text_top1_matches']!=check['frames']:
                    raise ValueError('Self-comparison is not exact: '+json.dumps(check))
                parent_loss=float(teacher_loss(pz[0,0],target,mask)[0]) if target is not None else None
                install(candidate);ch,cz=forward(inputs);metrics=differences(ph,pz,ch,cz,mask)
                row=dict(group=group,index=index,teacher_id=record['id'],kind=record['kind'],
                    self_comparison=check,**metrics)
                if target is not None:
                    row.update(approved_teacher_balanced_ce_parent=parent_loss,
                        approved_teacher_balanced_ce_child=float(teacher_loss(cz[0,0],target,mask)[0]),
                        nonpad=differences(ph,pz,ch,cz,mask & (target!=3)),
                        pad3=differences(ph,pz,ch,cz,mask & (target==3)))
                else:
                    base_h,prob=references[index]
                    row.update(legacy_unadapted_kl_parent=float(F.kl_div(pz[0,0,mask].float().log_softmax(-1),prob,reduction='batchmean')),
                        legacy_unadapted_kl_child=float(F.kl_div(cz[0,0,mask].float().log_softmax(-1),prob,reduction='batchmean')),
                        legacy_unadapted_hidden_mse_parent=float((ph[:,mask].float()-base_h.float()).square().mean()),
                        legacy_unadapted_hidden_mse_child=float((ch[:,mask].float()-base_h.float()).square().mean()))
                rows.append(row);print(json.dumps(row),flush=True)
    finally:
        with torch.no_grad():install(saved)
        for p in learned.values():p.grad=None
    if tensor_hashes(learned.items())!=before:
        raise RuntimeError('Diagnostic did not restore exact parent adapter')
    frozen=tensor_hashes((name,p) for name,p in lm.named_parameters() if name not in learned)
    if frozen!=metadata['frozen_hashes']:raise RuntimeError('Diagnostic changed frozen base parameters')
    assert len(rows)==53
    report=dict(schema='dl-native-retention-drift-diagnostic-v1',metadata=metadata,
        parent=str(args.warm_start_proof),candidate=str(proof),candidate_identity=identity,
        parent_checkpoint_sha256=sha256(args.warm_start_proof/'adapters.safetensors'),
        source_code_sha256=sha256(Path(__file__)),complete_teacher_inputs=49,complete_legacy_references=4,
        identity_mean_reproduction=rounding,
        cases=rows,optimizer_steps=0,new_checkpoint_written=False,parent_parameters_restored_exact=True,
        frozen_base_parameters_unchanged=True,legacy_reference_targets_and_weights_unchanged=True,
        evaluation_outputs_used_for_training=False,training_approved=False,candidate_adopted=False,
        native_behavior_evidence=False,diagnostic_only=True,
        interpretation='Pointwise drift on complete approved histories; does not establish loss-component causality or justify extension')
    (args.out/'native-v2-retention-drift-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete_inputs=53,optimizer_steps=0,parent_parameters_restored_exact=True)),flush=True)
