"""Complete32 actual native Engine trace with the original delivered text spans.

Force approved teacher text/agent acoustics only for this engineering audit.
Independently rebuild every delayed input and loss mask with streamed user codes.
No behavior, latency, numerical, gradient or optimizer approval is granted.
"""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from evals import native_reasoning_cache as cache_module
from evals import native_control_engine as old
from evals.native_reasoning_source import sha

SCHEMA='dl-native-reasoning32-engine-inputs-v1'


def span_plan(arrivals,frames):
    if (type(frames)is not int or frames<1 or any(type(a['ready_at_s'])not in (int,float)
            or not math.isfinite(a['ready_at_s'])or type(a['text'])is not str for a in arrivals)):
        raise ValueError('Finite original delivery clocks and native frame count required')
    spans=[dict(inject_frame=math.ceil(a['ready_at_s']/.08),reference=a['text'])for a in arrivals]
    if (any(not 1<=s['inject_frame']<frames or not s['reference'].strip()for s in spans)
            or len({s['inject_frame']for s in spans})!=len(spans)
            or [s['inject_frame']for s in spans]!=sorted(s['inject_frame']for s in spans)):
        raise ValueError('Every actual delivery needs an ordered complete live-frame insertion')
    return spans


def assemble(clean,prefix,spans,spm,user=None):
    import torch
    from contextspan.model.sequence_convention import assemble_training_sequence,context_span_ids,SILENCE_TOKENS,SINE_TOKENS
    if (clean.dtype!=torch.int64 or clean.ndim!=2 or clean.shape[0]!=17
            or prefix.dtype!=torch.int64 or prefix.ndim!=2 or prefix.shape[0]!=17
            or (clean[0]<0).any()or(clean[0]>=32000).any()
            or (clean[1:]<0).any()or(clean[1:]>=2048).any()):
        raise ValueError('Complete valid native lanes and original prefix required')
    actual=clean.clone()
    if user is not None:
        if user.dtype!=torch.int64 or tuple(user.shape)!=(1,8,clean.shape[1])or(user<0).any()or(user>=2048).any():
            raise ValueError('Complete actual streamed-user conditioning required')
        actual[9:17]=user[0]
    codes,tm,am=assemble_training_sequence(actual,torch.ones(clean.shape[1],dtype=torch.bool),spans,spm)
    codes=torch.cat([prefix,codes],dim=1)
    tm=torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),tm])
    am=torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),am])
    mapping=[prefix.shape[1]+f+sum(len(context_span_ids(s['reference'],spm))for s in spans if s['inject_frame']<=f)
        for f in range(clean.shape[1])]
    expected=torch.zeros(codes.shape[1],dtype=torch.bool);expected[mapping]=True
    expected_audio=expected.clone();real=actual.clone();ac=[*range(2,9),*range(10,17)]
    for span in spans:
        f=span['inject_frame'];expected_audio[mapping[f-1]]=False
        real[2:9,f-1]=torch.tensor(SILENCE_TOKENS[1:]);real[10:17,f-1]=torch.tensor(SINE_TOKENS[1:])
        if not torch.equal(codes[ac,mapping[f]-1],actual[ac,f-1]):
            raise ValueError('Pinned upstream span must relocate, never lose, provided pending acoustics')
    if (codes.shape[1]>1100 or not torch.equal(tm,expected)or not torch.equal(am,expected_audio)
            or not torch.equal(codes[:,mapping],real)):
        raise ValueError('Exact upstream real/text/audio/span ownership required; no extra loss exclusions')
    return codes,tm,am,mapping


def trace(engine,voice,clean,pcm,spans,label):
    import numpy as np
    import torch
    from evals.lychee_context import BASE_INSTRUCTIONS
    lm=engine.lm;engine.reset();state=engine.lm_gen._streaming_state;gen=engine.lm_gen
    direct,graph,step,prepare=lm.forward_codes,state.graphed_main,gen.step,gen.prepare_step_input
    chunks,calls,user,provided,targets,waves,applied=[],[],[],[],[],[],[]
    phase='prefix';frame=-1;cursor=0;began=time.monotonic()
    def capture(original,route):
        def wrapped(value):
            nonlocal cursor
            if phase=='live'and(route!='graph'or value.shape[-1]!=1):
                raise ValueError('Actual live native calls must remain one-frame graph calls')
            chunks.append(value.detach().cpu().clone())
            calls.append(dict(phase=phase,route=route,start=cursor,frames=value.shape[-1]));cursor+=value.shape[-1]
            return original(value)
        return wrapped
    def forced(*args,**kwargs):
        if phase=='live':
            if args or kwargs.get('moshi_tokens')is not None or kwargs.get('text_token')is not None:
                raise ValueError('Unexpected preexisting teacher forcing')
            user.append(kwargs['input_tokens'].detach().cpu().clone())
            kwargs.update(text_token=clean[0,frame:frame+1].to(engine.device),
                moshi_tokens=clean[None,1:9,frame:frame+1].to(engine.device))
        return step(*args,**kwargs)
    def prepared(*args,**kwargs):
        r=prepare(*args,**kwargs)
        if phase=='live':
            if r is None:raise ValueError('Original real frame was lost during bootstrap')
            _,flags,current,_,_=r;provided.append(flags.detach().cpu().clone());targets.append(current.detach().cpu().clone())
        return r
    lm.forward_codes,state.graphed_main=capture(direct,'direct'),capture(graph,'graph')
    gen.step,gen.prepare_step_input=forced,prepared
    try:
        with torch.no_grad():
            engine.set_persona(BASE_INSTRUCTIONS,voice)
            for frame in range(clean.shape[1]):
                for span in (s for s in spans if s['inject_frame']==frame):
                    phase='span';begin=cursor;n=engine.inject_context_span(span['reference'])
                    applied.append(dict(real_frame=frame,virtual_start=begin,frames=n,reference=span['reference']))
                phase='live';r=engine.step(pcm[frame*1920:(frame+1)*1920]);wave=np.asarray(r['agent_pcm'],dtype=np.float32)
                if wave.shape!=(1920,)or not np.isfinite(wave).all():
                    raise ValueError('Every original real frame requires a finite decoded native packet')
                waves.append(wave.copy())
                if(frame+1)%256==0:print(json.dumps(dict(id=label,frames=frame+1,total=clean.shape[1])),flush=True)
    finally:
        lm.forward_codes,state.graphed_main=direct,graph;gen.step,gen.prepare_step_input=step,prepare
    if graph._graph is None:raise ValueError('Published native CUDA main graph did not execute')
    return dict(inputs=torch.cat(chunks,dim=2),streamed_user_codes=torch.cat(user,dim=2),
        provided=torch.cat(provided,dim=2),current_targets=torch.cat(targets,dim=2)),calls,np.concatenate(waves),applied,dict(
        graph_policy=old.GRAPH_POLICY,cuda_main_graph_captured=True,elapsed_seconds=time.monotonic()-began)


def validate_trace(clean,prefix,spans,spm,saved,calls,applied,initial):
    import torch
    from contextspan.moshi.models.lm import _delay_sequence
    from contextspan.model.sequence_convention import context_span_ids
    from evals.context_native_bf16_stream_diagnostic import call_partition
    codes,tm,am,mapping=assemble(clean,prefix,spans,spm,saved['streamed_user_codes'])
    offline,_,_,_=assemble(clean,prefix,spans,spm)
    required={'inputs','streamed_user_codes','provided','current_targets','codes','mask','audio_mask','targets','offline_inputs'}
    if (set(saved)!=required or not torch.equal(saved['inputs'],old.delayed(codes,_delay_sequence,initial))
            or not torch.equal(saved['offline_inputs'],old.delayed(offline,_delay_sequence,initial))
            or not torch.equal(saved['codes'],codes)or not torch.equal(saved['mask'],tm)
            or not torch.equal(saved['audio_mask'],am)or not torch.equal(saved['targets'],codes[0])
            or saved['provided'].dtype!=torch.bool or tuple(saved['provided'].shape)!=(1,17,clean.shape[1])
            or not saved['provided'].all()or saved['current_targets'].dtype!=torch.int64
            or tuple(saved['current_targets'].shape)!=(1,17,clean.shape[1])
            or not torch.equal(saved['current_targets'][0,0],clean[0])
            or not torch.equal(saved['inputs'][:,:9],saved['offline_inputs'][:,:9])):
        raise ValueError('Complete native inputs, masks, supplied targets or assistant ownership changed')
    user=saved['streamed_user_codes'][0]
    current=torch.cat([clean[:2],torch.cat([prefix[2:9,-1:],clean[2:9,:-1]],dim=1),
        user[:1],torch.cat([prefix[10:17,-1:],user[1:,:-1]],dim=1)],dim=0)[None]
    if not torch.equal(current,saved['current_targets']):raise ValueError('Current native delay targets differ from approved stream ownership')
    call_partition(calls,codes.shape[1]);live=[];seen=[];n=0
    for call in calls:
        if call['phase']=='live':
            if call['route']!='graph'or call['frames']!=1:raise ValueError('One actual native graph call per real frame required')
            live.append(call['start']);n+=1
        elif call['phase']=='span'and call['route']=='direct':seen.append((n,call['start'],call['frames']))
        elif call['phase']!='prefix'or call['route']!='direct':raise ValueError('Unexpected native call phase')
    expected=[]
    for span in spans:
        expected.append((span['inject_frame'],prefix.shape[1]+span['inject_frame']+sum(x[2]for x in expected),
            len(context_span_ids(span['reference'],spm))))
    if (live!=mapping or seen!=expected or [(s['real_frame'],s['virtual_start'],s['frames'])for s in applied]!=expected
            or [s['reference']for s in applied]!=[s['reference']for s in spans]):
        raise ValueError('Actual native span ledger or every real-to-virtual frame mapping changed')
    return dict(full_real_frames=clean.shape[1],prefix_frames=prefix.shape[1],full_virtual_frames=codes.shape[1],
        text_targets=int(tm.sum()),assistant_acoustic_source_frames_per_head=int(am.sum()),
        spoken_targets=int((codes[0,tm]>=5).sum()),request_targets=int((codes[0,tm]==4).sum()),
        actual_delivered_contexts=len(spans),all17_delayed_input_channels_exact=True,
        assistant_targets_unchanged=True,only_streamed_user_conditioning_changed=True,
        pre_span_acoustic_loss_exclusions=len(spans),all_source_acoustics_preserved=True,
        acoustic_mask_policy='Unchanged upstream: preceding real frame acoustic loss excluded when pending delay-one acoustics relocate into last masked span column',
        prefix_loss_targets=0,supplied_span_loss_targets=0,user_audio_loss_targets=0,training_approved=False)


def cache_args(a):return SimpleNamespace(**{**vars(a),'out':a.cache})


def source(a):
    from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256
    from evals.personaplex_train import MODEL_HASHES
    manifest=json.loads(a.assets.read_text());root=Path(manifest['source']);weights=Path(manifest['path'])
    if (manifest['source_commit']!=SOURCE_COMMIT or manifest['revision']!=MODEL_REVISION
            or subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)
            or sha(weights/'context_spanning_7b.pt')!=WEIGHT_SHA256
            or any(sha(a.model/name)!=h for name,h in MODEL_HASHES.items())):
        raise ValueError('Pinned clean native source, complete weights and original model assets required')
    sys.path.insert(0,str(root));return manifest,weights


def capture(a):
    report=cache_module.verify(cache_args(a));_,weights=source(a)
    from evals.context_rollout_conditioning import verify_checkpoint
    identity,parent=verify_checkpoint(a.proof)
    if a.proof.name!='context-training-proof-7510331'or identity['optimizer_steps']!=520:
        raise ValueError('Exact original native520 data-audit comparator required')
    a.out.mkdir(exist_ok=False)
    import shutil
    shutil.copyfile(a.proof/'voice.pt',a.out/'voice.pt')
    os.environ.update(CS_BASE_DIR=str(a.model),CS_WEIGHTS_DIR=str(weights),CS_VOICES_DIR=str(weights/'voices'),HF_HUB_OFFLINE='1')
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-m','evals.cuda_linker_path','--out',str(a.out/'driver')],text=True).strip()
    from portable.context.native_audition import configure_cuda_toolkit
    configure_cuda_toolkit()
    import numpy as np,soundfile as sf,torch,sentencepiece
    import contextspan.model.engine as native
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_adapter_load import before_engine
    from evals.context_spanning_probe import audited_loads
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_execution import math_attention,snapshot
    from evals.context_native_audio49_data import digest
    voice=torch.load(a.out/'voice.pt',weights_only=True,map_location='cpu')['codes']
    tokenizer=sentencepiece.SentencePieceProcessor(str(a.model/'tokenizer_spm_32k_3.model'))
    prefix_check=persona_prefix(BASE_INSTRUCTIONS,voice,tokenizer)
    for entry in report['examples']:
        payload=torch.load(a.cache/entry['file'],weights_only=True,map_location='cpu');clean=payload['codes'][0]
        assemble(clean,prefix_check,span_plan(payload['arrivals'],clean.shape[1]),tokenizer)
    print(json.dumps(dict(complete32_prepared_ownership_checked=True,model_loaded=False,training_approved=False)),flush=True)
    math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    native.seed_all(42424242);loads=[]
    with before_engine(native,a.proof)as adapter,audited_loads(loads):
        engine=native.Engine(checkpoint=str(weights/'context_spanning_7b.pt'),device='cuda',temp=.8,temp_text=.7)
    final=[r for r in loads if r['module_identity']==id(engine.lm)]
    if not final or not final[-1]['strict']or final[-1]['missing']or final[-1]['unexpected']:
        raise ValueError('Strict complete native model load required')
    if not getattr(engine,'_dl_checkpoint_native_protocol',None):raise ValueError('Original native bootstrap/provided-acoustic span protocol missing')
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,engine.spm);initial=engine.lm._get_initial_token().detach().cpu()
    if not torch.equal(prefix,prefix_check):raise ValueError('Actual Engine vocabulary/prefix differs from complete32 preparation')
    before=tensor_hashes(engine.lm.named_parameters());converted=json.loads((a.data/'ingestion.json').read_text())
    result=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256=sha(Path(__file__)),cache_sha256=sha(a.cache/'cache.json'),assets_sha256=sha(a.assets),parent=identity,adapter=adapter,
        voice_sha256=sha(a.out/'voice.pt'),prefix_sha256=digest(prefix),initial_token_sha256=digest(initial),
        cases=[],complete32=False,passed=False,optimizer_updates=0,training_approved=False,behavior_qualified=False,
        teacher_text_and_agent_audio_forced=True,native_graph_policy=old.GRAPH_POLICY,execution=snapshot(engine.lm),
        numerical_gradient_and_behavior_approval_required=True)
    for entry,row in zip(report['examples'],converted['accepted']):
        folder=a.out/entry['id'];folder.mkdir();payload=torch.load(a.cache/entry['file'],weights_only=True,map_location='cpu')
        clean=payload['codes'][0];spans=span_plan(payload['arrivals'],clean.shape[1])
        audio,rate=sf.read(a.data/row['path'],dtype='float32',always_2d=True)
        if rate!=24000 or audio.shape!=(clean.shape[1]*1920,2):raise ValueError('Every original stereo sample must be consumed')
        saved,calls,wave,applied,execution=trace(engine,voice,clean,audio[:,1],spans,entry['id'])
        codes,tm,am,_=assemble(clean,prefix,spans,engine.spm,saved['streamed_user_codes'])
        offline,_,_,_=assemble(clean,prefix,spans,engine.spm)
        saved.update(codes=codes,mask=tm,audio_mask=am,targets=codes[0],offline_inputs=old.delayed(offline,_delay_sequence,initial))
        ownership=validate_trace(clean,prefix,spans,engine.spm,saved,calls,applied,initial)
        torch.save(saved,folder/'inputs.pt');sf.write(folder/'teacher-forced-native.wav',wave,24000,subtype='FLOAT')
        result['cases'].append(dict(id=entry['id'],family=entry['family'],calls=calls,applied_spans=applied,ownership=ownership,
            file='inputs.pt',sha256=sha(folder/'inputs.pt'),source_cache_sha256=entry['sha256'],
            audio_file='teacher-forced-native.wav',audio_sha256=sha(folder/'teacher-forced-native.wav'),
            audio_peak=float(np.abs(wave).max()),native_execution=execution,passed=True))
        (a.out/'engine-inputs.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(id=entry['id'],complete=len(result['cases']),expected=32)),flush=True)
    unchanged=before==tensor_hashes(engine.lm.named_parameters())
    if not unchanged or len(result['cases'])!=32:raise ValueError('Complete unchanged32 native data audit required')
    result.update(complete32=True,passed=True,all_model_parameters_unchanged=True)
    (a.out/'engine-inputs.json').write_text(json.dumps(result,indent=2)+'\n');return result


def verify(a):
    import numpy as np,soundfile as sf,torch,sentencepiece
    report=cache_module.verify(cache_args(a));source(a)
    from evals.context_rollout_conditioning import verify_checkpoint
    from contextspan.model.sequence_convention import persona_prefix
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_native_audio49_data import digest
    r=json.loads((a.out/'engine-inputs.json').read_text());identity,parent=verify_checkpoint(a.proof)
    if (r['schema']!=SCHEMA or not r['complete32']or not r['passed']or not r['all_model_parameters_unchanged']
            or r['code_sha256']!=sha(Path(__file__))or r['cache_sha256']!=sha(a.cache/'cache.json')
            or r['assets_sha256']!=sha(a.assets)or r['parent']!=identity or identity['optimizer_steps']!=520
            or r['voice_sha256']!=sha(a.out/'voice.pt')or r['voice_sha256']!=parent['voice_sha256']
            or r['optimizer_updates']!=0 or r['training_approved']is not False or r['behavior_qualified']is not False
            or r['teacher_text_and_agent_audio_forced']is not True or r['native_graph_policy']!=old.GRAPH_POLICY
            or r['numerical_gradient_and_behavior_approval_required']is not True
            or [x['id']for x in r['cases']]!=[x['id']for x in report['examples']]):
        raise ValueError('Exact complete32 native input trace and immutable provenance required')
    spm=sentencepiece.SentencePieceProcessor(str(a.model/'tokenizer_spm_32k_3.model'))
    voice=torch.load(a.out/'voice.pt',weights_only=True,map_location='cpu')['codes'];prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
    initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
    if r['prefix_sha256']!=digest(prefix)or r['initial_token_sha256']!=digest(initial):raise ValueError('Original prefix/bootstrap changed')
    for entry,row in zip(report['examples'],r['cases']):
        folder=a.out/entry['id'];payload=torch.load(a.cache/entry['file'],weights_only=True,map_location='cpu')
        if (row['file']!='inputs.pt'or row['audio_file']!='teacher-forced-native.wav'or not row['passed']
                or row['source_cache_sha256']!=entry['sha256']or row['family']!=entry['family']
                or row['sha256']!=sha(folder/row['file'])or row['audio_sha256']!=sha(folder/row['audio_file'])
                or row['native_execution']['graph_policy']!=old.GRAPH_POLICY
                or row['native_execution']['cuda_main_graph_captured']is not True):
            raise ValueError('Native source or trace artifact changed')
        saved=torch.load(folder/row['file'],weights_only=True,map_location='cpu');clean=payload['codes'][0]
        check=validate_trace(clean,prefix,span_plan(payload['arrivals'],clean.shape[1]),spm,saved,row['calls'],row['applied_spans'],initial)
        wave,rate=sf.read(folder/row['audio_file'],dtype='float32')
        if check!=row['ownership']or rate!=24000 or wave.shape!=(clean.shape[1]*1920,)or not np.isfinite(wave).all()or float(np.abs(wave).max())!=row['audio_peak']:
            raise ValueError('Native ownership or every decoded real PCM packet changed')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','version','semantic','data','model','codec-source','interleaver-source','cache','assets','proof','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    r=(verify if a.verify_only else capture)(a)
    print(json.dumps(dict(complete32=r['complete32'],passed=r['passed'],optimizer_updates=0,training_approved=False)))
