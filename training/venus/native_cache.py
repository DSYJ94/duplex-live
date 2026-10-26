"""Teacher replay through the RELEASED streaming path, with exact causal labels.

No optimizer runs here. Forced choices construct training histories; they are
never counted as free-running native behavior or successful student delegation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoModel, set_seed


def load_native(path,language_only=False):
    from portable.venus.native_app import ensure_guard
    ensure_guard()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one GPU; no Mac CPU native inference')
    set_seed(42)
    model=AutoModel.from_pretrained(str(path), trust_remote_code=True, local_files_only=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16, init_vision=False,
        init_audio=not language_only, init_tts=not language_only).eval().cuda()
    model.requires_grad_(False)
    return model, None if language_only else model.as_duplex(generate_audio=False)


def schedule(duplex, timeline):
    scheduled={}
    for action in timeline['actions']:
        if action['kind']=='end_turn':
            chunks=[[duplex.speak_token_id,duplex.turn_eos_token_id,duplex.chunk_eos_token_id]]
        else:
            text=action['text']
            if action['kind']=='delegate':text='<delegate>'+text+'</delegate>'
            ids=duplex.tokenizer.encode(text,add_special_tokens=False)
            chunks=[]
            for start in range(0,len(ids),16):
                last=start+16>=len(ids)
                chunks.append([duplex.speak_token_id]+ids[start:start+16]+
                    ([duplex.turn_eos_token_id] if last and action['final'] else [])+
                    [duplex.chunk_eos_token_id])
        for offset,ids in enumerate(chunks):
            frame=action['frame']+offset
            if frame in scheduled or frame>=timeline['frames'] or len(ids)>19:
                raise ValueError('Teacher speech/action does not fit original causal timeline')
            scheduled[frame]=ids
    return scheduled


def capture(duplex, timeline, wav, retain_segments=False):
    import soundfile as sf
    data,sr=sf.read(wav,dtype='float32');assert sr==16000 and data.ndim==1
    assert len(data)==timeline['frames']*16000
    schedule_ids=schedule(duplex,timeline)
    original_feed,original_decode=duplex.decoder.feed,duplex.decoder.decode
    embeddings=[];segments=[];positions=[];targets=[];decision=[];logit_proofs=[];cursor=0;queue=[]
    special={duplex.listen_token_id,duplex.speak_token_id,duplex.turn_eos_token_id,
             duplex.chunk_eos_token_id,duplex.chunk_tts_eos_token_id}
    def feed(x,return_logits=False):
        nonlocal cursor
        embeddings.append(x.detach().cpu().clone());segments.append(dict(start=cursor,length=x.shape[0],return_logits=return_logits))
        cursor+=x.shape[0]
        return original_feed(x,return_logits=return_logits)
    def decode(logits,**kwargs):
        if not queue:raise ValueError('Native decoder requested an unplanned teacher token')
        ident=queue.pop(0);positions.append(cursor-1);targets.append(ident);decision.append(ident in special)
        if retain_segments and len(logit_proofs)<32:
            logit_proofs.append(dict(position=cursor-1,logits=logits.detach().cpu().clone()))
        return torch.tensor([ident],device=logits.device,dtype=torch.long)
    duplex.decoder.feed,duplex.decoder.decode=feed,decode
    try:
        with torch.no_grad():
            duplex.prepare(prefix_system_prompt='Streaming Omni Conversation.')
            for frame in range(timeline['frames']):
                from portable.venus.native_app import ensure_guard
                ensure_guard()
                queue=list(schedule_ids.get(frame,[duplex.listen_token_id]))
                if frame<duplex.force_listen_count and queue!=[duplex.listen_token_id]:
                    raise ValueError('Teacher would bypass released initial listening rule')
                if not duplex.current_turn_ended and queue==[duplex.listen_token_id]:
                    raise ValueError('Teacher attempts illegal listen during unfinished native turn')
                feedback=[f['text'] for f in timeline['backend_feedback'] if f['frame']==frame]
                result=duplex.streaming_prefill(audio_waveform=data[frame*16000:(frame+1)*16000],text_list=feedback or None)
                if not result['success']:raise ValueError(result)
                duplex.streaming_generate(max_new_speak_tokens_per_chunk=20,decode_mode='greedy')
                if frame<duplex.force_listen_count:
                    # The native initial forced listen does not call decode; no fabricated label.
                    queue=[]
                if queue:raise ValueError('Native decoder did not consume exact teacher sequence')
    finally:
        duplex.decoder.feed,duplex.decoder.decode=original_feed,original_decode
    example=dict(id=timeline['scene_id'],fold=timeline['fold'],family=timeline['family'],
        embeddings=torch.cat(embeddings),positions=torch.tensor(positions),targets=torch.tensor(targets),
        decision_mask=torch.tensor(decision),segments=segments,logit_proofs=logit_proofs,forward_path='native-segmented-bptt-v1',
        native_teacher_forced=True,student_behavior_claimed=False,
        timeline_sha256=hashlib.sha256(json.dumps(timeline,sort_keys=True).encode()).hexdigest())
    if not positions or min(positions)<0 or max(positions)>=example['embeddings'].shape[0]-1:
        raise ValueError('Invalid next-token causal alignment')
    return example


def replay_proof(duplex,example):
    """Same native feed segmentation, fresh KV state, exact logits at captured choices."""
    duplex.decoder.reset();by_position={r['position']:r['logits'] for r in example['logit_proofs']};errors=[]
    with torch.no_grad():
        for segment in example['segments']:
            start,n=segment['start'],segment['length'];last=start+n-1
            result=duplex.decoder.feed(example['embeddings'][start:start+n].to(duplex.device),return_logits=segment['return_logits'])
            if last in by_position:
                if result is None:raise ValueError('Captured native label lacks actual preceding logits')
                errors.append(float((result[0].cpu()-by_position[last]).abs().max()))
    if not errors or max(errors)!=0:raise ValueError('Exact native segmented replay failed: '+str(errors))
    return dict(compared=len(errors),maximum_logit_error=max(errors),passed=True)


def training_input_proof(model,example):
    """Check the differentiable training input against the captured native choices.

    Same-segmentation replay above is exact. Full versus cached BF16 attention
    uses different reduction shapes; declare separate tolerances before use.
    """
    from .lora import selected_forward
    limits=dict(minimum_cosine=.9999,maximum_relative_l2=.01,
                maximum_target_ce_delta=.03,minimum_argmax_agreement=.95)
    device=next(model.llm.parameters()).device
    with torch.no_grad():logits,_=selected_forward(model.llm,example,device)
    indices={int(p):i for i,p in enumerate(example['positions'])}
    proof=example['logit_proofs'];assert proof
    selected=torch.stack([logits[indices[r['position']]] for r in proof]).float().cpu()
    reference=torch.cat([r['logits'] for r in proof]).float()
    targets=torch.tensor([int(example['targets'][indices[r['position']]]) for r in proof])
    cosine=float(torch.nn.functional.cosine_similarity(selected,reference,dim=-1).min())
    relative=float((selected-reference).norm()/reference.norm().clamp_min(1e-12))
    ce_delta=float((torch.nn.functional.cross_entropy(selected,targets,reduction='none')-torch.nn.functional.cross_entropy(reference,targets,reduction='none')).abs().max())
    agreement=float((selected.argmax(-1)==reference.argmax(-1)).float().mean())
    report=dict(forward_path=example.get('forward_path','whole-sequence'),compared=len(proof),minimum_cosine=cosine,relative_l2=relative,maximum_target_ce_delta=ce_delta,argmax_agreement=agreement,maximum_absolute_logit_error=float((selected-reference).abs().max()),limits=limits)
    report['passed']=cosine>=limits['minimum_cosine'] and relative<=limits['maximum_relative_l2'] and ce_delta<=limits['maximum_target_ce_delta'] and agreement>=limits['minimum_argmax_agreement']
    return report


def main(a):
    root=Path(a.timelines);manifest=json.loads((root/'manifest.json').read_text())
    if a.approval:
        approval=json.loads(Path(a.approval).read_text())
        assert approval['approved'] and approval['manifest_sha256']==hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
    elif not a.proof_only:raise ValueError('Independent full data approval required before cache collection')
    model,duplex=load_native(a.model);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[]
    scenes=manifest['scenes']
    if a.proof_only:scenes=[next(s for s in scenes if s['fold']=='TRAIN' and s['family']=='ordinary')]
    else:scenes=[s for s in scenes if s['fold'] in ('TRAIN','DEV')]
    start=time.monotonic()
    for i,row in enumerate(scenes):
        folder=root/row['id'];assert hashlib.sha256((folder/'timeline.json').read_bytes()).hexdigest()==row['timeline_sha256']
        assert hashlib.sha256((folder/'user.wav').read_bytes()).hexdigest()==row['user_wav_sha256']
        timeline=json.loads((folder/'timeline.json').read_text());ex=capture(duplex,timeline,folder/'user.wav',retain_segments=(i==0))
        if i==0:
            (out/'native-replay-proof.json').write_text(json.dumps(replay_proof(duplex,ex),indent=2)+'\n')
            parity=training_input_proof(model,ex);(out/'native-training-input-proof.json').write_text(json.dumps(parity,indent=2)+'\n')
            if not parity['passed']:raise ValueError('Native/training BF16 sequence parity failed: '+str(parity))
        path=out/(row['id']+'.pt');torch.save(ex,path);rows.append(dict(id=row['id'],fold=row['fold'],family=row['family'],file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tokens=ex['embeddings'].shape[0],labels=len(ex['targets'])))
        print(json.dumps(dict(event='native_cache_scene',completed=i+1,total=len(scenes),elapsed=time.monotonic()-start,**rows[-1])),flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),model_revision='4681004131ea1f1fb9aca9f0db76be322657cc79',scenes=rows,proof_only=a.proof_only,optimizer_updates=0),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--timelines',required=True);p.add_argument('--output',required=True);p.add_argument('--approval');p.add_argument('--proof-only',action='store_true');main(p.parse_args())
