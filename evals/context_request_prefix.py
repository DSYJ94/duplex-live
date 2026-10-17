"""Revalidate a teacher-rolled-in lookup prefix, excluding every later receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from evals.moshi_model_download import sha256


def scope(manifest,labels,mask):
    teacher=manifest['teacher'];requests=teacher.get('derived_requests',[])
    if (teacher['kind']!='private' or teacher['source_split']!='train'
            or teacher['source_model']!='gpt-live-1' or teacher.get('request_policy')!='native-prelead-epad'
            or len(requests)!=1 or requests[0]['operation']!='lookup'
            or manifest['actual_prefix_has_no_reply_or_request_tokens'] is not True
            or manifest['all_teacher_reply_and_request_tokens_retained'] is not True
            or manifest['source_start_frame']!=manifest['future_first_real_frame']):
        raise RuntimeError('Lookup prefix requires a complete independently approved private TRAIN teacher')
    frame=requests[0]['derived_request_frame'];relative=frame-manifest['future_first_real_frame']
    if not 1<=relative<manifest['future_real_frames']:
        raise RuntimeError('First approved lookup is outside the captured future')
    count=relative+1;stop=manifest['real_to_future_virtual'][relative]+1
    calls=[c for c in manifest['calls'] if c['start']<stop]
    if (stop!=count or manifest['real_to_future_virtual'][:count]!=list(range(count))
            or len(calls)!=count or any(c!={'phase':'live','route':'graph','start':i,'frames':1}
                for i,c in enumerate(calls))
            or any(s['real_frame']<=frame for s in manifest['future_recorded_receipts'])):
        raise RuntimeError('Lookup prefix must end before every receipt and virtual span')
    selected=labels[:stop]
    if (len(selected)!=count or selected[-2:]!=[0,4]
            or any(t not in (0,3) for t in selected[:-1]) or not all(mask[:stop])):
        raise RuntimeError('Lookup prefix cannot contain spoken words, missing targets or another request')
    return count,stop


def derive(parent,data,assets,examples,spm,voice):
    import torch
    from evals.context_early_validate import validate as validate_parent
    approval,full=validate_parent(parent,data,assets,examples,spm,voice)
    manifest=json.loads((parent/'early-future.json').read_text())
    count,stop=scope(manifest,full['text_targets'].tolist(),full['text_mask'].tolist())
    tensors={'delayed_inputs':full['delayed_inputs'][:,:,:stop].clone(),
        'hidden':full['hidden'][:,:stop].clone(),'text_targets':full['text_targets'][:stop].clone(),
        'text_mask':torch.arange(stop)==stop-1}
    result={'schema':'dl-context-request-prefix-plan-v1','target_kind':'lookup_request',
        'parent_component':str(parent),'parent_manifest_sha256':sha256(parent/'early-future.json'),
        'parent_tensor_sha256':manifest['sha256'],'parent_restart_gate_sha256':approval['restart_gate_sha256'],
        'actual_history':manifest['actual_history'],'source_history_sha256':approval['source_history_sha256'],
        'source_execution':manifest['source_execution'],'teacher':manifest['teacher'],'world':manifest['world'],
        'voice_codes_sha256':manifest['voice_codes_sha256'],
        'future_first_real_frame':manifest['future_first_real_frame'],'future_frames':count,
        'future_virtual_frames':stop,'request_real_frame':manifest['future_first_real_frame']+count-1,
        'request_prediction_real_frame':manifest['future_first_real_frame']+count-1,
        'expected_request_readout_real_frame':manifest['future_first_real_frame']+count,
        'returned_text_delay_frames':1,'prior_rollin_conditioning_loss_targets':0,
        'teacher_rollin_conditioning_verified':True,'future_receipts':0,'spoken_word_targets':0,
        'private_fact_targets':0,'past_loss_targets':0,'span_loss_targets':0,'audio_loss_targets':0,
        'user_loss_targets':0,'new_backend_calls':False,'training_approved':False,
        'scope':'approved initial lookup controls only; all later words and backend receipts excluded'}
    return result,tensors


def validate(folder,data,assets,examples,spm,voice):
    import torch
    m=json.loads((folder/'request-prefix.json').read_text())
    expected,tensors=derive(Path(m['parent_component']),data,assets,examples,spm,voice)
    if (m['schema']!='dl-context-request-prefix-capture-v1' or m['file']!='request-prefix.pt'
            or any(m.get(k)!=v for k,v in expected.items() if k!='schema')
            or sha256(folder/m['file'])!=m['sha256'] or not re.fullmatch('[a-f0-9]{40}',m['builder_commit'])):
        raise RuntimeError('Changed or unsupported lookup-prefix component')
    code=subprocess.check_output(['git','show',m['builder_commit']+':evals/context_request_prefix.py'],
        cwd=Path(__file__).resolve().parents[1])
    if hashlib.sha256(code).hexdigest()!=m['builder_code_sha256']:
        raise RuntimeError('Changed lookup-prefix builder provenance')
    saved=torch.load(folder/m['file'],map_location='cpu',weights_only=True)
    if (set(saved)!=set(tensors) or any(not torch.equal(saved[k],v) for k,v in tensors.items())
            or not torch.equal(voice,torch.load(folder/'voice.pt',map_location='cpu',weights_only=True)['codes'])):
        raise RuntimeError('Lookup prefix changed teacher conditioning, labels or voice')
    report={'schema':'dl-context-request-prefix-audit-v1','component':str(folder),'passed':True,
        'target_kind':'lookup_request','diagnostic_kind':'backend_request',
        'diagnostic_frame':m['future_frames']-1,'diagnostic_target':4,
        'manifest_sha256':sha256(folder/'request-prefix.json'),'tensor_sha256':m['sha256'],
        'source_history_sha256':m['source_history_sha256'],'source_execution':m['source_execution'],
        'voice_codes_sha256':m['voice_codes_sha256'],'builder_commit':m['builder_commit'],
        'scenario':json.loads((Path(m['actual_history'])/'student-history.json').read_text())['source']['scenario'],
        'teacher':m['teacher'],'world':m['world'],'frames':m['future_frames'],
        'text_targets':int(saved['text_mask'].sum()),'request_targets':1,
        'past_loss_targets':0,'span_loss_targets':0,'audio_loss_targets':0,'user_loss_targets':0,
        'spoken_word_targets':0,'private_fact_targets':0,'future_receipts':0,
        'teacher_rollin_conditioning_verified':True,'native_action_component_approved':True,
        'whole_model_quality_or_delegation_qualified':False}
    return report,saved


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('parent','data','assets','tokenizer','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True);args=ap.parse_args()
    import torch
    import sentencepiece
    from evals.personaplex_train import MODEL_HASHES
    from evals.context_training_data import load_training
    from evals.lychee_context import BASE_INSTRUCTIONS
    if sha256(args.tokenizer)!=MODEL_HASHES['tokenizer_spm_32k_3.model']:raise RuntimeError('Changed native tokenizer')
    assets=json.loads(args.assets.read_text());sys.path.insert(0,assets['source'])
    from contextspan.model.sequence_convention import persona_prefix
    spm=sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    voice=torch.load(args.parent/'voice.pt',map_location='cpu',weights_only=True)['codes']
    examples,_=load_training(args.cache,spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),request_policy='native-prelead-epad')
    result,tensors=derive(args.parent,args.data,args.assets,examples,spm,voice);args.out.mkdir(exist_ok=False)
    torch.save(tensors,args.out/'request-prefix.pt');(args.out/'voice.pt').write_bytes((args.parent/'voice.pt').read_bytes())
    result.update(schema='dl-context-request-prefix-capture-v1',file='request-prefix.pt',
        sha256=sha256(args.out/'request-prefix.pt'),builder_code_sha256=sha256(Path(__file__)),
        builder_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    (args.out/'request-prefix.json').write_text(json.dumps(result,indent=2)+'\n')
    audit,_=validate(args.out,args.data,args.assets,examples,spm,voice)
    (args.out/'cpu-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
