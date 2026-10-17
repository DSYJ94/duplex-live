"""Independently check native teacher traces with explicitly streamed user inputs.

Only user conditioning may differ from the offline cache. Approved assistant
words, acoustic conditioning, timing, prefix and loss masks must remain fixed.
This is data approval, not optimizer or whole-model qualification.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from evals.moshi_model_download import sha256


def prepared(example,prefix,user,spm):
    import torch
    from contextspan.moshi.models.lm import _delay_sequence
    from contextspan.model.sequence_convention import assemble_training_sequence
    delays=[0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]
    clean=example['clean_codes'].clone()
    if user.shape!=(1,8,clean.shape[1]) or user.dtype!=torch.int64 or (user<0).any() or (user>=2048).any():
        raise RuntimeError('Native teacher user conditioning has invalid coverage or codec IDs')
    clean[9:17]=user[0]
    assembled,mask,_=assemble_training_sequence(clean,torch.ones(clean.shape[1],dtype=torch.bool),
        example['record']['spans'],spm)
    assembled=torch.cat([prefix,assembled],dim=1)
    mask=torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),mask])
    if not torch.equal(assembled[:9],example['codes'][:9]) or not torch.equal(mask,example['text_mask']):
        raise RuntimeError('Streaming user conversion changed assistant targets, prefix or masks')
    initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
    return torch.cat([initial,_delay_sequence(delays,assembled[None],initial)],dim=2)[:,:,:-1]


def validate(folder,examples,prefix,spm,scope_file,assets,voice_file):
    import torch
    from evals.context_retention_teacher import scope
    from evals.context_teacher_engine_inputs import compare
    report_file=folder/'teacher-engine-input-audit.json';r=json.loads(report_file.read_text())
    approved=json.loads(scope_file.read_text())
    if (r['schema']!='dl-context-teacher-engine-input-audit-v1' or r['optimizer_steps']!=0
            or r['training_approved'] is not False or not r['teacher_text_and_audio_forced']
            or not r['native_user_encoder_and_agent_decoder_executed']
            or not r['real_to_virtual_exact'] or not r['all_real_targets_provided']
            or approved['schema']!='dl-context-all-teacher-scope-audit-v1' or not approved['passed']
            or approved['code_sha256']!=sha256(Path(__file__).with_name('context_retention_teacher.py'))
            or approved['voice_sha256']!=sha256(voice_file)
            or approved['assets_manifest_sha256']!=sha256(assets)
            or r['teacher_scope_sha256']!=sha256(scope_file)
            or r['assets_manifest_sha256']!=sha256(assets)
            or r['voice_sha256']!=sha256(voice_file) or sha256(folder/'voice.pt')!=sha256(voice_file)
            or r['file']!='teacher-engine-inputs.pt' or sha256(folder/r['file'])!=r['sha256']
            or r['audio_file']!='teacher-forced-native.wav' or sha256(folder/r['audio_file'])!=r['audio_sha256']):
        raise RuntimeError('Native teacher trace lacks complete matching scope, voice or artifacts')
    commit=r['source_commit'];names={'context_teacher_engine_inputs.py','context_native_layout.py','context_training_data.py'}
    if not re.fullmatch('[a-f0-9]{40}',commit) or set(r['code_sha256'])!=names:
        raise RuntimeError('Native teacher trace lacks builder provenance')
    for name,digest in r['code_sha256'].items():
        content=subprocess.check_output(['git','show',commit+':evals/'+name],cwd=Path(__file__).resolve().parents[1])
        if hashlib.sha256(content).hexdigest()!=digest:raise RuntimeError('Changed native teacher builder')
    example=next(e for e in examples if e['record']['id']==r['teacher']['id'])
    labels,mask,checked=scope(example)
    if checked!=r['teacher'] or checked not in approved['teachers']:
        raise RuntimeError('Native teacher labels differ from independent approval')
    saved=torch.load(folder/r['file'],map_location='cpu',weights_only=True)
    count=example['clean_codes'].shape[1];total=example['codes'].shape[1]
    shapes={'delayed_inputs':(1,17,total),'expected_prepared_inputs':(1,17,total),
        'text_targets':(total,),'text_mask':(total,),'live_native_targets':(1,17,count),
        'live_provided':(1,17,count),'encoded_user_codes':(1,8,count)}
    if set(saved)!=set(shapes) or any(tuple(saved[k].shape)!=s for k,s in shapes.items()):
        raise RuntimeError('Native teacher trace lost complete tensor coverage')
    for key,value in saved.items():
        dtype=torch.bool if key in ('text_mask','live_provided') else torch.int64
        if value.dtype!=dtype or not torch.isfinite(value).all():raise RuntimeError('Invalid native teacher tensor')
    original=prepared(example,prefix,example['clean_codes'][None,9:17],spm)
    streamed=prepared(example,prefix,saved['encoded_user_codes'],spm)
    if (not torch.equal(original,saved['expected_prepared_inputs'])
            or not torch.equal(labels,saved['text_targets']) or not torch.equal(mask,saved['text_mask'])
            or not saved['live_provided'].all()):
        raise RuntimeError('Native teacher changed original input oracle, labels or ownership')
    old=compare(saved['delayed_inputs'],original)
    codec=compare(saved['encoded_user_codes'],example['clean_codes'][None,9:17])
    if old!=r['input_comparison'] or codec!=r['streaming_user_codec_comparison']:
        raise RuntimeError('Native teacher diagnostic disagrees with its actual tensors')
    current=saved['live_native_targets'];clean=example['clean_codes'];user=saved['encoded_user_codes'][0]
    target=torch.cat([clean[:2],torch.cat([prefix[2:9,-1:],clean[2:9,:-1]],dim=1),
        user[:1],torch.cat([prefix[10:17,-1:],user[1:,:-1]],dim=1)],dim=0)[None]
    if not torch.equal(current,target):raise RuntimeError('Native teacher current targets have the wrong stream or delay')
    cursor=0;indices=[];spans=[];n=0
    for call in r['calls']:
        if call['start']!=cursor or call['frames']<1:raise RuntimeError('Noncontiguous native teacher ledger')
        if call['phase']=='live':
            if call['route']!='graph' or call['frames']!=1:raise RuntimeError('Invalid native real-frame route')
            indices.append(cursor);n+=1
        elif call['phase']=='span' and call['route']=='direct':spans.append((n,cursor,call['frames']))
        elif call['phase']!='prefix' or call['route']!='direct':raise RuntimeError('Invalid native teacher call phase')
        cursor+=call['frames']
    if cursor!=total or n!=count or indices!=example['record']['real_to_virtual']:
        raise RuntimeError('Native teacher ledger altered real/virtual time')
    expected=[]
    from contextspan.model.sequence_convention import context_span_ids
    for span in example['record']['spans']:
        size=len(context_span_ids(span['reference'],spm));f=span['inject_frame']
        start=prefix.shape[1]+f+sum(s[2] for s in expected)
        expected.append((f,start,size))
    if spans!=expected or [(s['real_frame'],s['virtual_start'],s['frames']) for s in r['applied_spans']]!=expected:
        raise RuntimeError('Native teacher spans differ from independently approved arrivals')
    if [s['reference'] for s in r['applied_spans']]!=[s['reference'] for s in example['record']['spans']]:
        raise RuntimeError('Native teacher changed provided reference text')
    result=compare(saved['delayed_inputs'],streamed)
    audit=dict(schema='dl-context-teacher-native-input-approval-v1',passed=result['exact'],teacher=checked,
        capture=str(folder),capture_report_sha256=sha256(report_file),capture_tensor_sha256=r['sha256'],
        scope_sha256=sha256(scope_file),voice_sha256=sha256(voice_file),assets_manifest_sha256=sha256(assets),
        native_input_comparison=result,legacy_input_comparison=old,user_codec_comparison=codec,
        labels_and_mask_unchanged=True,only_user_conditioning_reencoded=True,
        prefix_loss_targets=0,span_loss_targets=0,user_loss_targets=0,audio_loss_targets=0,
        optimizer_steps=0,training_approved=False,whole_model_quality_qualified=False)
    return audit,dict(delayed_inputs=saved['delayed_inputs'],text_targets=labels,text_mask=mask)


def main():
    import argparse,sys,torch,sentencepiece
    from evals.context_training_data import load_training
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_spanning_download import SOURCE_COMMIT
    from evals.personaplex_train import MODEL_HASHES
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('assets','tokenizer','voice','scope','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    ap.add_argument('--capture',type=Path,action='append',required=True);args=ap.parse_args()
    asset=json.loads(args.assets.read_text());source=Path(asset['source'])
    if (asset['source_commit']!=SOURCE_COMMIT or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD']).decode().strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain']).strip()
            or sha256(args.tokenizer)!=MODEL_HASHES['tokenizer_spm_32k_3.model']):
        raise RuntimeError('Changed pinned teacher source/tokenizer')
    sys.path.insert(0,str(source))
    from contextspan.model.sequence_convention import persona_prefix
    spm=sentencepiece.SentencePieceProcessor(str(args.tokenizer));voice=torch.load(args.voice,map_location='cpu',weights_only=True)['codes']
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
    examples,_=load_training(args.cache,spm,prefix,request_policy='native-prelead-epad')
    args.out.mkdir(exist_ok=False);results=[]
    for folder in args.capture:
        audit,_=validate(folder,examples,prefix,spm,args.scope,args.assets,args.voice)
        results.append(audit);print(json.dumps(audit),flush=True)
    if len({a['teacher']['id'] for a in results})!=len(results):raise RuntimeError('Duplicate approved teacher trace')
    result=dict(schema='dl-context-native-teacher-approval-set-v1',passed=all(a['passed'] for a in results),
        cases=results,optimizer_steps=0,training_approved=False,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise RuntimeError('Native teacher has differences beyond declared streaming user conditioning')


if __name__=='__main__':main()
