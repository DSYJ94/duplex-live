"""Build and independently revalidate action-only labels; never qualify a task."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
from evals.moshi_model_download import sha256


def validate(folder,data,assets,examples,spm,voice):
    import torch
    from evals.context_action_targets import plan
    manifest=json.loads((folder/'action-future.json').read_text())
    expected,component=plan(Path(manifest['actual_history']),data,assets,examples,spm,voice,manifest['teacher']['id'])
    if (any(manifest.get(k)!=v for k,v in expected.items() if k!='schema')
            or manifest['schema']!='dl-context-action-capture-v1' or manifest['file']!='action-future.pt'
            or manifest['sha256']!=sha256(folder/'action-future.pt')):
        raise RuntimeError('Changed or unapproved action component')
    names={'context_action_targets.py','context_action_component.py'}
    if (set(manifest['builder_code_sha256'])!=names
            or not re.fullmatch('[a-f0-9]{40}',manifest['builder_commit'])):
        raise RuntimeError('Incomplete action builder provenance')
    for name,digest in manifest['builder_code_sha256'].items():
        code=subprocess.check_output(['git','show',manifest['builder_commit']+':evals/'+name],
            cwd=Path(__file__).resolve().parents[1])
        if hashlib.sha256(code).hexdigest()!=digest:raise RuntimeError('Changed action builder')
    saved=torch.load(folder/'action-future.pt',map_location='cpu',weights_only=True)
    if set(saved)!=set(component) or any(not torch.equal(saved[k],v) for k,v in component.items()):
        raise RuntimeError('Action conditioning or independently approved labels changed')
    if not torch.equal(voice,torch.load(folder/'voice.pt',map_location='cpu',weights_only=True)['codes']):
        raise RuntimeError('Action voice changed')
    audit={'schema':'dl-context-action-audit-v1','component':str(folder),'passed':True,
        'manifest_sha256':sha256(folder/'action-future.json'),'tensor_sha256':manifest['sha256'],
        'source_history_sha256':expected['source_history_sha256'],
        'voice_codes_sha256':expected['voice_codes_sha256'],'source_execution':expected['source_execution'],
        'builder_commit':manifest['builder_commit'],'scenario':expected['scenario'],
        'target_kind':'cancel_request','diagnostic_kind':'backend_request',
        'diagnostic_frame':2,'diagnostic_target':4,'frames':3,'text_targets':3,
        'past_loss_targets':0,'span_loss_targets':0,'audio_loss_targets':0,'user_loss_targets':0,
        'spoken_acknowledgement_targets':0,'private_fact_targets':0,
        'observed_conditioning_exact':True,'native_action_component_approved':True,
        'whole_model_quality_or_delegation_qualified':False}
    return audit,saved


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('history','data','assets','tokenizer','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--teacher-id',required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    args=ap.parse_args()
    import torch
    import sentencepiece
    from evals.personaplex_train import MODEL_HASHES
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_training_data import load_training
    from evals.context_action_targets import plan
    if sha256(args.tokenizer)!=MODEL_HASHES['tokenizer_spm_32k_3.model']:
        raise RuntimeError('Changed native tokenizer')
    assets=json.loads(args.assets.read_text());sys.path.insert(0,assets['source'])
    from contextspan.model.sequence_convention import persona_prefix
    spm=sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    voice=torch.load(args.history/'student-voice.pt',map_location='cpu',weights_only=True)['codes']
    examples,_=load_training(args.cache,spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),request_policy='native-prelead-epad')
    manifest,tensors=plan(args.history,args.data,args.assets,examples,spm,voice,args.teacher_id)
    args.out.mkdir(exist_ok=False)
    torch.save(tensors,args.out/'action-future.pt')
    (args.out/'voice.pt').write_bytes((args.history/'student-voice.pt').read_bytes())
    result={**manifest,'schema':'dl-context-action-capture-v1','file':'action-future.pt',
        'sha256':sha256(args.out/'action-future.pt'),
        'builder_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'builder_code_sha256':{n:sha256(Path(__file__).with_name(n))
            for n in ('context_action_targets.py','context_action_component.py')}}
    (args.out/'action-future.json').write_text(json.dumps(result,indent=2)+'\n')
    audit,_=validate(args.out,args.data,args.assets,examples,spm,voice)
    (args.out/'cpu-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
