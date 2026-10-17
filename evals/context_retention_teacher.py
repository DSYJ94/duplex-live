"""Audit an explicit teacher-text term on the existing four retention histories.

This is supervised assistant text, not new labels on student self-history.
Prefix, supplied context, user and audio targets remain excluded.
"""
import hashlib
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def validate_weight(weight):
    if not math.isfinite(weight) or not 0.<=weight<=2.:
        raise ValueError('Retention teacher weight must be finite and between zero and two')


def scope(example):
    import torch
    r=example['record'];codes=example['codes'];clean=example['clean_codes'];mask=example['text_mask']
    if (r['source_model']!='gpt-live-1' or r['source_split']!='train'
            or r.get('request_policy')!='native-prelead-epad'
            or r['kind'] not in ('private','simple','clarify','cancel')):
        raise RuntimeError('Retention text requires an approved independent TRAIN teacher')
    if (codes.dtype!=torch.int64 or codes.shape!=(17,r['frames'])
            or clean.dtype!=torch.int64 or clean.shape!=(17,r['real_frames'])
            or mask.dtype!=torch.bool or mask.shape!=(r['frames'],)):
        raise RuntimeError('Retention teacher sequence or mask changed')
    mapping=r['real_to_virtual'];prefix=r['prefix_frames']
    if (len(mapping)!=r['real_frames'] or not mapping or mapping[0]!=prefix
            or any(not isinstance(i,int) for i in mapping)
            or any(a>=b for a,b in zip(mapping,mapping[1:]))
            or mapping[-1]>=r['frames'] or prefix<1):
        raise RuntimeError('Retention teacher lost its complete real-to-virtual mapping')
    expected=torch.zeros_like(mask);expected[mapping]=True
    if not torch.equal(mask,expected) or not torch.equal(codes[0,mapping],clean[0]):
        raise RuntimeError('Prefix/span tokens acquired loss or measured teacher text changed')
    labels=codes[0].clone();selected=labels[mask]
    if (selected<0).any() or (selected>=32000).any() or not (selected>=5).any():
        raise RuntimeError('Retention teacher has invalid or empty spoken targets')
    evidence={k:r[k] for k in ('id','kind','source_model','source_split','cache','cache_sha256',
        'codes_sha256','wav_sha256','sidecar_sha256','ingestion_sha256','request_policy')}
    evidence.update(frames=r['frames'],real_frames=r['real_frames'],prefix_frames=prefix,
        text_targets=int(mask.sum()),spoken_targets=int((selected>=5).sum()),
        request_targets=int((selected==4).sum()),
        targets_sha256=hashlib.sha256(selected.contiguous().numpy().tobytes()).hexdigest(),
        mask_sha256=hashlib.sha256(mask.contiguous().numpy().tobytes()).hexdigest(),
        prefix_loss_targets=0,span_loss_targets=0,user_loss_targets=0,audio_loss_targets=0)
    return labels,mask,evidence


def loss(logits,labels,mask):
    import torch
    from torch.nn import functional as F
    if (logits.ndim!=2 or labels.dtype!=torch.int64 or mask.dtype!=torch.bool
            or labels.shape!=mask.shape or labels.shape!=(logits.shape[0],) or not mask.any()):
        raise RuntimeError('Invalid retention teacher loss coverage')
    z=logits[mask].float();y=labels[mask]
    if not torch.isfinite(z).all() or (y<0).any() or (y>=z.shape[1]).any():
        raise RuntimeError('Invalid eligible retention teacher logits or labels')
    weights=torch.where(y==3,.5,1.)
    return (F.cross_entropy(z,y,reduction='none')*weights).sum()/weights.sum()


def policy(weight,teachers,audit_file,assets,voice_file):
    validate_weight(weight)
    if not weight:return None
    r=json.loads(audit_file.read_text())
    if (r['schema']!='dl-context-retention-teacher-audit-v1' or r['passed'] is not True
            or r['optimizer_steps']!=0 or r['teachers']!=teachers
            or r['assets_manifest_sha256']!=sha256(assets)
            or r['voice_sha256']!=sha256(voice_file)
            or r['code_sha256']!=sha256(Path(__file__)) or r['user_loss_targets']!=0
            or r['prefix_loss_targets']!=0 or r['span_loss_targets']!=0
            or r['audio_loss_targets']!=0 or len(teachers)!=4
            or {t['kind'] for t in teachers}!={'private','simple','clarify','cancel'}):
        raise RuntimeError('Retention teacher loss lacks its matching independent CPU scope audit')
    return dict(weight=weight,teachers=teachers,audit_sha256=sha256(audit_file),
        code_sha256=r['code_sha256'],scope='existing four approved full TRAIN teacher assistant-text histories',
        prefix_loss_targets=0,span_loss_targets=0,user_loss_targets=0,audio_loss_targets=0)


def main():
    import argparse
    import sys
    import torch
    import sentencepiece
    from evals.context_training_data import load_training
    from evals.personaplex_train import MODEL_HASHES
    from evals.lychee_context import BASE_INSTRUCTIONS
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('assets','tokenizer','voice','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True);args=ap.parse_args()
    if sha256(args.tokenizer)!=MODEL_HASHES['tokenizer_spm_32k_3.model']:raise RuntimeError('Changed teacher tokenizer')
    assets=json.loads(args.assets.read_text());sys.path.insert(0,assets['source'])
    from contextspan.model.sequence_convention import persona_prefix
    spm=sentencepiece.SentencePieceProcessor(str(args.tokenizer));voice=torch.load(args.voice,map_location='cpu',weights_only=True)['codes']
    examples,_=load_training(args.cache,spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),request_policy='native-prelead-epad')
    teachers=[]
    for kind in ('private','simple','clarify','cancel'):
        e=max((e for e in examples if e['record']['kind']==kind),key=lambda e:e['codes'].shape[-1])
        _,_,audit=scope(e);teachers.append(audit)
    result=dict(schema='dl-context-retention-teacher-audit-v1',passed=True,optimizer_steps=0,
        assets_manifest_sha256=sha256(args.assets),voice_sha256=sha256(args.voice),
        code_sha256=sha256(Path(__file__)),teachers=teachers,prefix_loss_targets=0,span_loss_targets=0,
        user_loss_targets=0,audio_loss_targets=0,native_behavior_qualified=False)
    args.out.mkdir(exist_ok=False);(args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
