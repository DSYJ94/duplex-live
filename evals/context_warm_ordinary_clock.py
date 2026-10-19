"""Separate ordinary clock/native-audio anchor beside unchanged word/legacy losses."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.context_warm_ordinary_retention import IDS,select as teacher_select,loss

ACTUAL=(
 ('7505207','recovery-train-008','5f1b5da415c2836d42115877dd133dd573e438aca5342c0bef0b56fb852e373f'),
 ('7505214','recovery-train-009','f14a42a75653487f5a493551711c3c4579c2661bc542b11083848dc730bbae6f'),
 ('7505216','recovery-train-010','6a4fa2721c32186c633a3594daa4472a7d9dd2e1fd8c0c1be8c6cd6d5842acb5'),
 ('7505217','recovery-train-011','056091bf2777498a4d576efb5027db38393cbd80b450e0fb74ab1a6ab3d61f9a'))
CLOCK_IDS=tuple('clock-'+i for i in IDS)+tuple('clock-actual-'+i[0] for i in ACTUAL)
DATA_SHA256='de92099057ded15a1af0e387c96b457cdf4c1731c26594dcdca211d4c442fc1e'


def select(old,new,batches,audits):
    if len(batches)!=28 or len(audits)!=28:raise ValueError('Clock retention must preserve all28 actual components')
    rows=[(i,m,t,dict(r,id='clock-'+r['id'])) for i,m,t,r in teacher_select(old,new)]
    ordinary=[(i,a) for i,a in enumerate(audits) if a['target_kind']=='ordinary_answer']
    if len(ordinary)!=4:raise ValueError('All four predeclared approved actual ordinary histories required')
    for (index,a),(job,scenario,manifest) in zip(ordinary,ACTUAL):
        if (a['component']!='/projects/u6si/duplex-live/runs/context-training-proof-'+job
                or a['scenario']!=scenario or a['manifest_sha256']!=manifest
                or not a['passed'] or a['past_loss_targets']!=0 or a['audio_loss_targets']!=0):
            raise ValueError('Changed actual ordinary TRAIN history; no evaluation-output training permitted')
        inputs,mask,targets=batches[index]
        rows.append((inputs,mask,targets,dict(id='clock-actual-'+job,kind='simple',source_split='train',
            component=a['component'],scenario=scenario,approved_capture_manifest_sha256=manifest)))
    if [r[3]['id'] for r in rows]!=list(CLOCK_IDS):raise ValueError('Incomplete ordinary clock anchor selection')
    return rows


def prepare(lm,learned,old,new,batches,audits,args,forward,word_policy):
    import torch
    from safetensors.torch import load_file
    from evals.moshi_training_smoke import tensor_hashes
    selected=select(old,new,batches,audits)
    if (sha256(args.out/'data.json')!=DATA_SHA256 or word_policy['weight']!=64
            or word_policy['mode']!='native520-approved-ordinary-nonpad-v1'
            or not word_policy['source_words_audio_targets_and_legacy_losses_unchanged']):
        raise ValueError('Clock anchor needs exact all28 TRAIN data and unchanged ten-word anchor')
    parent=load_file(args.warm_start_proof/'adapters.safetensors',device='cuda')
    if set(parent)!=set(learned):raise ValueError('Incomplete accepted parent')
    before=tensor_hashes(learned.items());saved={n:p.detach().clone() for n,p in learned.items()}
    stored={};refs=[];records=[]
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            for n,p in learned.items():p.copy_(parent[n])
            for inputs,mask,targets,record in selected:
                if not mask.any():raise ValueError('Clock anchor has no approved assistant frames')
                h,z=forward(inputs);reference=(h[:,mask].float().detach().clone(),z[0,0,mask].float().log_softmax(-1).detach().clone())
                value,kl,mse=loss(h,z,mask,reference)
                if float(value)!=0 or float(kl)!=0 or float(mse)!=0:raise ValueError('Clock/native-audio reference self check failed')
                refs.append((inputs,mask,reference,record['id']))
                stored[record['id']+'.hidden']=reference[0].cpu();stored[record['id']+'.logp']=reference[1].cpu()
                records.append(dict(source=record,frames=int(mask.sum()),pad3_frames=int((mask & (targets==3)).sum()),
                    nonpad_frames=int((mask & (targets!=3)).sum()),
                    approved_tensor_hashes=tensor_hashes([('inputs',inputs),('mask',mask),('targets',targets)])))
    finally:
        with torch.no_grad():
            for n,p in learned.items():p.copy_(saved[n])
    if tensor_hashes(learned.items())!=before:raise RuntimeError('Clock reference construction changed parameters')
    policy=dict(mode='native520-approved-ordinary-clock-and-actual-v1',weight=64.,internal_kl_weight=1.,internal_hidden_mse_weight=10.,
        parent_checkpoint_sha256=sha256(args.warm_start_proof/'adapters.safetensors'),data_sha256=DATA_SHA256,
        source_code_sha256=sha256(Path(__file__)),ids=list(CLOCK_IDS),examples=records,
        reference_tensor_hashes=tensor_hashes(stored.items()),
        sampling='Mean all14 every update at original approved assistant/future mask positions, including PAD3',
        word_anchor_and_all_legacy_losses_unchanged=True,source_masks_words_audio_and_targets_unchanged=True,
        evaluation_outputs_used_for_training=False,user_past_audio_loss_targets=0)
    audit=args.out/'warm-ordinary-clock-reference-audit.json';cache=args.out/'warm-ordinary-clock-references.pt'
    if args.reload:
        prior=json.loads(audit.read_text())
        if prior['policy']!=policy or sha256(cache)!=prior['cache_sha256']:
            raise ValueError('Independent clock/native-audio reference reconstruction changed')
    else:
        if cache.exists() or audit.exists():raise ValueError('Preserve prior clock artifacts')
        torch.save(stored,cache)
        audit.write_text(json.dumps(dict(schema='dl-native-warm-ordinary-clock-reference-audit-v1',policy=policy,
            cache_sha256=sha256(cache),exact_self_checks=14,parameters_restored_exact=True,
            reference_targets_verified=True,optimizer_steps_during_construction=0,adopted=False),indent=2)+'\n')
    return policy,refs
