"""Additional native520 ordinary-word anchor; every legacy loss stays intact."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256

IDS=tuple('teacher-recovery-train-%03d'%i for i in range(8,12))+tuple(
    'teacher-native2-train-%03d'%i for i in range(8,14))
DRIFT_SHA256='6819554f26727800dc6f680b86ff60de8e6940a6f9a76efcbf0aeb0c2e50b50b'


def select(old,new):
    if len(old)!=25 or len(new)!=24:raise ValueError('All49 approved teachers must remain present')
    rows={b[3]['id']:b for b in old+new}
    if len(rows)!=49 or not set(IDS)<=set(rows):raise ValueError('Missing/duplicate approved ordinary TRAIN history')
    result=[rows[i] for i in IDS]
    if any(b[3]['kind']!='simple' or b[3]['source_split']!='train' for b in result):
        raise ValueError('Warm anchor may use only its ten predeclared ordinary TRAIN histories')
    return result


def loss(hidden,logits,mask,reference):
    logp=logits[0,0,mask].float().log_softmax(-1)
    target_h,target_logp=reference
    kl=(target_logp.exp()*(target_logp-logp)).sum(-1).mean()
    mse=(hidden[:,mask].float()-target_h).square().mean()
    return kl+10.*mse,kl,mse


def prepare(lm,learned,old,new,args,forward,metadata):
    import torch
    from safetensors.torch import load_file
    from evals.moshi_training_smoke import tensor_hashes
    selected=select(old,new)
    drift_path=args.native_v2_warm_ordinary_approval;drift=json.loads(drift_path.read_text())
    if (sha256(drift_path)!=DRIFT_SHA256 or drift['schema']!='dl-native-retention-drift-diagnostic-v1' or drift['optimizer_steps']!=0
            or drift['complete_teacher_inputs']!=49 or drift['complete_legacy_references']!=4
            or len(drift['cases'])!=53 or not drift['parent_parameters_restored_exact']
            or not drift['frozen_base_parameters_unchanged'] or not drift['diagnostic_only']
            or drift['parent_checkpoint_sha256']!=sha256(args.warm_start_proof/'adapters.safetensors')
            or drift['metadata']['data_sha256']!=metadata['data_sha256']
            or drift['metadata']['voice_sha256']!=metadata['voice_sha256']
            or drift['metadata']['assets_manifest_sha256']!=metadata['assets_manifest_sha256']
            or drift['metadata']['frozen_hashes']!=metadata['frozen_hashes']
            or metadata['recipe']['retention_kl_weight']!=1. or metadata['recipe']['retention_hidden_mse_weight']!=10.
            or drift['metadata']['recipe']['retention_teacher_text']!=metadata['recipe']['retention_teacher_text']
            or drift['metadata']['recipe']['additional_native_v2']!=metadata['recipe']['additional_native_v2']):
        raise ValueError('Complete53 unchanged approved-history drift evidence required for additional anchor')
    parent=load_file(args.warm_start_proof/'adapters.safetensors',device='cuda')
    if set(parent)!=set(learned):raise ValueError('Incomplete native520 reference parameters')
    saved={n:p.detach().clone() for n,p in learned.items()};before=tensor_hashes(learned.items());refs=[];stored={};records=[]
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            for n,p in learned.items():p.copy_(parent[n])
            for inputs,mask,targets,record in selected:
                active=mask & (targets!=3)
                if not active.any():raise ValueError('Ordinary anchor has no assistant word targets')
                h,z=forward(inputs);reference=(h[:,active].float().detach().clone(),z[0,0,active].float().log_softmax(-1).detach().clone())
                check,kl,mse=loss(h,z,active,reference)
                if float(check)!=0 or float(kl)!=0 or float(mse)!=0:raise ValueError('Warm reference self check failed')
                refs.append((inputs,active,reference,record['id']))
                stored[record['id']+'.hidden']=reference[0].cpu();stored[record['id']+'.logp']=reference[1].cpu()
                records.append(dict(id=record['id'],kind=record['kind'],source_split=record['source_split'],
                    approved_codes_sha256=record['codes_sha256'],approved_targets_sha256=record['targets_sha256'],
                    approved_mask_sha256=record['mask_sha256'],nonpad_frames=int(active.sum())))
    finally:
        with torch.no_grad():
            for n,p in learned.items():p.copy_(saved[n])
    if tensor_hashes(learned.items())!=before:raise RuntimeError('Warm-reference construction changed model parameters')
    policy=dict(mode='native520-approved-ordinary-nonpad-v1',weight=args.native_v2_warm_ordinary_weight,
        internal_kl_weight=1.,internal_hidden_mse_weight=10.,parent_checkpoint_sha256=sha256(args.warm_start_proof/'adapters.safetensors'),
        complete53_diagnostic_sha256=sha256(drift_path),source_code_sha256=sha256(Path(__file__)),
        examples=records,reference_tensor_hashes=tensor_hashes(stored.items()),
        sampling='Mean over all10 anchors on every update; legacy28/25/new24 losses unchanged',
        source_words_audio_targets_and_legacy_losses_unchanged=True,evaluation_outputs_used_for_training=False)
    audit=args.out/'warm-ordinary-reference-audit.json';cache=args.out/'warm-ordinary-references.pt'
    if args.reload:
        prior=json.loads(audit.read_text())
        if prior['policy']!=policy or sha256(cache)!=prior['cache_sha256']:
            raise ValueError('Independent warm-reference reconstruction differs from saved training targets')
    else:
        if cache.exists() or audit.exists():raise ValueError('Preserve existing warm-reference artifacts')
        torch.save(stored,cache)
        audit.write_text(json.dumps(dict(schema='dl-native-warm-ordinary-reference-audit-v1',policy=policy,
            cache_sha256=sha256(cache),exact_self_checks=10,parameters_restored_exact=True,
            reference_targets_verified=True,optimizer_steps_during_construction=0,model_adoption_approved=False),indent=2)+'\n')
    return policy,refs
