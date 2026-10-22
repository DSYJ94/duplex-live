"""Whole40 native Engine input audit; force teachers, never optimize or qualify behavior.

The offline user codec remains a diagnostic. Only actual streamed-user conditioning
may differ; assistant text/audio, every frame and the native persona remain fixed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from data.synthetic.native_control_render import sha256
from evals.native_control_cache import verify as verify_cache

SCHEMA = 'dl-native-control40-engine-inputs-v1'
DELAYS = [0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1]


def assemble(clean,prefix,user=None):
    import torch
    if (clean.dtype!=torch.int64 or clean.ndim!=2 or clean.shape[0]!=17 or clean.shape[1]<1
            or prefix.dtype!=torch.int64 or prefix.ndim!=2 or prefix.shape[0]!=17 or prefix.shape[1]<1
            or (clean[0]<0).any() or (clean[0]>=32000).any() or (clean[0]==4).any()
            or (clean[1:]<0).any() or (clean[1:]>=2048).any()):
        raise ValueError('Complete native control source and nonempty persona prefix required')
    actual=clean.clone()
    if user is not None:
        if user.dtype!=torch.int64 or user.shape!=(1,8,clean.shape[1]) or (user<0).any() or (user>=2048).any():
            raise ValueError('Complete actual streamed user codebooks required')
        actual[9:17]=user[0]
    codes=torch.cat([prefix,actual],dim=1)
    mask=torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),torch.ones(clean.shape[1],dtype=torch.bool)])
    if not (codes[0,mask]>=5).any():raise ValueError('Approved spoken targets must accompany complete quiet/clock targets')
    return codes,mask


def delayed(codes,delay_sequence,initial):
    import torch
    if initial.shape!=(1,17,1) or initial.dtype!=torch.int64:
        raise ValueError('Pinned native initial-token geometry required')
    return torch.cat([initial,delay_sequence(DELAYS,codes[None],initial)],dim=2)[:,:,:-1]


def validate_trace(clean,prefix,saved,calls,delay_sequence,initial):
    import torch
    from evals.context_native_bf16_stream_diagnostic import call_partition
    codes,mask=assemble(clean,prefix,saved['streamed_user_codes'])
    original,_=assemble(clean,prefix)
    expected=delayed(codes,delay_sequence,initial)
    offline=delayed(original,delay_sequence,initial)
    frames=clean.shape[1];p=prefix.shape[1]
    call_partition(calls,codes.shape[1])
    live=[c for c in calls if c['phase']=='live']
    dtypes={k:torch.int64 for k in ('inputs','offline_inputs','codes','targets','streamed_user_codes','current_targets')}
    dtypes.update(mask=torch.bool,provided=torch.bool)
    if (set(saved)!={'inputs','offline_inputs','codes','mask','targets','streamed_user_codes','provided','current_targets'}
            or any(not isinstance(saved[k],torch.Tensor) or saved[k].dtype!=dtype for k,dtype in dtypes.items())
            or not torch.equal(saved['inputs'],expected) or not torch.equal(saved['offline_inputs'],offline)
            or not torch.equal(saved['codes'],codes) or not torch.equal(saved['mask'],mask)
            or not torch.equal(saved['targets'],codes[0])
            or saved['provided'].dtype!=torch.bool or saved['provided'].shape!=(1,17,frames) or not saved['provided'].all()
            or saved['current_targets'].dtype!=torch.int64 or saved['current_targets'].shape!=(1,17,frames)
            or not torch.equal(saved['current_targets'][0,0],clean[0])
            or [c['start'] for c in live]!=list(range(p,p+frames))
            or any(c['frames']!=1 or c['route']!='graph' for c in live)
            or any(c['phase'] not in ('prefix','live') for c in calls)
            or not torch.equal(saved['inputs'][:,:9],offline[:,:9])):
        raise ValueError('Actual Engine inputs, all-frame ownership or unchanged teacher targets differ')
    return dict(full_real_frames=frames,prefix_frames=p,text_targets=int(mask.sum()),
                spoken_targets=int((codes[0,mask]>=5).sum()),clock_targets=int((codes[0,mask]<5).sum()),
                all17_native_input_channels_exact=True,assistant_targets_unchanged=True,
                streamed_user_conditioning_only=True,prefix_text_loss_targets=0,
                prefix_audio_loss_targets=0,user_audio_loss_targets=0,spurious_request_targets=0,
                assistant_acoustic_source_frames_per_head=frames,native_depth_target_mask_approval_required=True,training_approved=False)


def trace(engine,voice,clean,pcm):
    import numpy as np
    import torch
    from contextspan.moshi.utils.compile import no_compile,no_cuda_graph
    from evals.lychee_context import BASE_INSTRUCTIONS
    lm=engine.lm;engine.reset()
    state=engine.lm_gen._streaming_state;gen=engine.lm_gen
    direct,graph,step,prepare=lm.forward_codes,state.graphed_main,gen.step,gen.prepare_step_input
    chunks,calls,user,provided,targets,waves=[],[],[],[],[],[]
    phase='prefix';frame=-1;cursor=0
    def capture(original,route):
        def wrapped(value):
            nonlocal cursor
            if phase=='live' and (route!='graph' or value.shape[-1]!=1):
                raise ValueError('Unexpected native live input call')
            chunks.append(value.detach().cpu().clone())
            calls.append(dict(phase=phase,route=route,start=cursor,frames=value.shape[-1]))
            cursor+=value.shape[-1];return original(value)
        return wrapped
    def forced(*args,**kwargs):
        if phase=='live':
            if args or kwargs.get('moshi_tokens') is not None or kwargs.get('text_token') is not None:
                raise ValueError('Native live Engine already supplied unexpected teacher tokens')
            user.append(kwargs['input_tokens'].detach().cpu().clone())
            kwargs.update(text_token=clean[0,frame:frame+1].to(engine.device),
                          moshi_tokens=clean[None,1:9,frame:frame+1].to(engine.device))
        return step(*args,**kwargs)
    def prepared(*args,**kwargs):
        result=prepare(*args,**kwargs)
        if phase=='live':
            if result is None:raise ValueError('Native real frame unexpectedly bootstrapped')
            _,flags,target,_,_=result
            provided.append(flags.detach().cpu().clone());targets.append(target.detach().cpu().clone())
        return result
    lm.forward_codes,state.graphed_main=capture(direct,'direct'),capture(graph,'graph')
    gen.step,gen.prepare_step_input=forced,prepared
    try:
        with torch.no_grad(),no_compile(),no_cuda_graph():
            engine.set_persona(BASE_INSTRUCTIONS,voice)
            for frame in range(clean.shape[1]):
                phase='live';result=engine.step(pcm[frame*1920:(frame+1)*1920])
                packet=np.asarray(result['agent_pcm'],dtype=np.float32)
                if packet.shape!=(1920,) or not np.isfinite(packet).all():
                    raise ValueError('Native teacher decoder lost a complete real PCM frame')
                waves.append(packet.copy())
    finally:
        lm.forward_codes,state.graphed_main=direct,graph
        gen.step,gen.prepare_step_input=step,prepare
    return dict(inputs=torch.cat(chunks,dim=2),streamed_user_codes=torch.cat(user,dim=2),
                provided=torch.cat(provided,dim=2),current_targets=torch.cat(targets,dim=2)),calls,np.concatenate(waves)


def run(engine,voice,cache,report,out):
    import numpy as np
    import soundfile as sf
    import torch
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_execution import snapshot
    from evals.context_native_audio49_data import digest
    conversion=json.loads((Path(report['conversion'])/'ingestion.json').read_text())
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,engine.spm)
    initial=engine.lm._get_initial_token().detach().cpu()
    if list(engine.lm.delays)!=DELAYS or engine.frame_size!=1920 or engine.frame_rate!=12.5:
        raise ValueError('Pinned native token-delay/stream clock changed')
    result=dict(schema=SCHEMA,cache=str(cache),cache_sha256=sha256(cache/'cache.json'),
                created_utc=datetime.now(timezone.utc).isoformat(),code_sha256=sha256(Path(__file__)),
                prefix_sha256=digest(prefix),voice_sha256=sha256(out/'voice.pt'),execution=snapshot(engine.lm),
                cases=[],complete40=False,passed=False,optimizer_updates=0,training_approved=False,
                numerical_gradient_and_behavior_approval_required=True,teacher_text_and_agent_audio_forced=True,
                latency_qualified=False,original_source_pcm_or_cache_changed=False)
    for entry,source in zip(report['examples'],conversion['accepted']):
        folder=out/entry['id'];folder.mkdir()
        clean=torch.load(cache/entry['file'],weights_only=True,map_location='cpu')['codes'][0]
        audio,rate=sf.read(source['path'],dtype='float32',always_2d=True)
        if rate!=24000 or audio.shape!=(clean.shape[1]*1920,2):
            raise ValueError('Every original stereo PCM frame must be consumed')
        saved,calls,wave=trace(engine,voice,clean,audio[:,1])
        codes,mask=assemble(clean,prefix,saved['streamed_user_codes'])
        original,_=assemble(clean,prefix)
        saved.update(codes=codes,mask=mask,targets=codes[0],offline_inputs=delayed(original,_delay_sequence,initial))
        torch.save(saved,folder/'inputs.pt')
        sf.write(folder/'teacher-forced-native.wav',wave,24000,subtype='FLOAT')
        ownership=validate_trace(clean,prefix,saved,calls,_delay_sequence,initial)
        record=dict(id=entry['id'],family=entry['family'],split='train',calls=calls,ownership=ownership,
                    file='inputs.pt',sha256=sha256(folder/'inputs.pt'),source_cache_sha256=entry['sha256'],
                    audio_file='teacher-forced-native.wav',audio_sha256=sha256(folder/'teacher-forced-native.wav'),
                    audio_peak=float(np.abs(wave).max()),passed=True)
        result['cases'].append(record)
        (out/'engine-inputs.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(id=entry['id'],complete=len(result['cases']),ownership_passed=True)),flush=True)
    if len(result['cases'])!=40 or len({r['id'] for r in result['cases']})!=40:
        raise ValueError('Native input approval requires complete40, never a subset')
    result.update(complete40=True,passed=True)
    (out/'engine-inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def verify(cache,out):
    """Independently reconstruct complete40 inputs/targets from original cache and voice."""
    import numpy as np
    import soundfile as sf
    import torch
    import sentencepiece
    report=verify_cache(cache)
    result=json.loads((out/'engine-inputs.json').read_text())
    assets=Path(result['assets']);manifest=json.loads(assets.read_text())
    from evals.context_spanning_download import SOURCE_COMMIT
    source=Path(manifest['source'])
    if (result['schema']!=SCHEMA or result['complete40'] is not True or result['passed'] is not True
            or result['code_sha256']!=sha256(Path(__file__)) or result['cache_sha256']!=sha256(cache/'cache.json')
            or result['assets_sha256']!=sha256(assets) or result['voice_sha256']!=sha256(out/'voice.pt')
            or result['optimizer_updates']!=0 or result['training_approved'] is not False
            or result['all_model_parameters_unchanged'] is not True
            or result['teacher_text_and_agent_audio_forced'] is not True
            or result['numerical_gradient_and_behavior_approval_required'] is not True
            or result['latency_qualified'] is not False or result['original_source_pcm_or_cache_changed'] is not False
            or manifest['source_commit']!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or [r['id'] for r in result['cases']]!=[r['id'] for r in report['examples']]):
        raise ValueError('Complete immutable zero-update Engine trace required')
    sys.path.insert(0,str(source))
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.context_native_audio49_data import digest
    identity,parent=verify_checkpoint(Path(result['parent']['proof']))
    if identity!=result['parent'] or identity['optimizer_steps']!=520 or parent['voice_sha256']!=sha256(out/'voice.pt'):
        raise ValueError('Original native520 and its exact voice required')
    spm=sentencepiece.SentencePieceProcessor(str(Path(report['model'])/'tokenizer_spm_32k_3.model'))
    voice=torch.load(out/'voice.pt',weights_only=True,map_location='cpu')['codes']
    prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
    initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
    if result['prefix_sha256']!=digest(prefix) or result['initial_token_sha256']!=digest(initial):
        raise ValueError('Native persona or bootstrap initial token changed')
    for entry,row in zip(report['examples'],result['cases']):
        folder=out/entry['id']
        if (row['file']!='inputs.pt' or row['audio_file']!='teacher-forced-native.wav'
                or row['family']!=entry['family'] or row['split']!='train'
                or row['source_cache_sha256']!=entry['sha256'] or row['passed'] is not True
                or sha256(folder/row['file'])!=row['sha256'] or sha256(folder/row['audio_file'])!=row['audio_sha256']):
            raise ValueError('Native control trace artifact or source identity changed')
        clean=torch.load(cache/entry['file'],weights_only=True,map_location='cpu')['codes'][0]
        saved=torch.load(folder/row['file'],weights_only=True,map_location='cpu')
        actual=validate_trace(clean,prefix,saved,row['calls'],_delay_sequence,initial)
        wave,rate=sf.read(folder/row['audio_file'],dtype='float32')
        if (actual!=row['ownership'] or rate!=24000 or wave.shape!=(clean.shape[1]*1920,)
                or not np.isfinite(wave).all() or float(np.abs(wave).max())!=row['audio_peak']):
            raise ValueError('Native teacher ownership or complete decoded waveform changed')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('cache','out'):ap.add_argument('--'+n,type=Path,required=True)
    for n in ('assets','base','proof'):ap.add_argument('--'+n,type=Path)
    ap.add_argument('--verify-only',action='store_true')
    a=ap.parse_args()
    if a.verify_only:
        result=verify(a.cache,a.out)
        print(json.dumps(dict(complete40=result['complete40'],optimizer_updates=0,training_approved=False)));return
    if not all((a.assets,a.base,a.proof)):ap.error('Capture requires assets, base and proof')
    report=verify_cache(a.cache)  # Full source/quality/semantic chain before Engine/CUDA.
    from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.personaplex_train import MODEL_HASHES
    assets=json.loads(a.assets.read_text());source=Path(assets['source']);weights=Path(assets['path'])
    identity,_=verify_checkpoint(a.proof)
    if (a.proof.name!='context-training-proof-7510331' or identity['optimizer_steps']!=520 or assets['source_commit']!=SOURCE_COMMIT or assets['revision']!=MODEL_REVISION
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
            or sha256(weights/'context_spanning_7b.pt')!=WEIGHT_SHA256):
        raise ValueError('Frozen native520 and pinned published Engine/weights required')
    if any(sha256(a.base/name)!=digest for name,digest in MODEL_HASHES.items()):
        raise ValueError('Pinned native base/codec/tokenizer assets changed')
    a.out.mkdir(parents=True,exist_ok=False)
    import shutil
    shutil.copyfile(a.proof/'voice.pt',a.out/'voice.pt')
    sys.path.insert(0,str(source))
    os.environ.update(CS_BASE_DIR=str(a.base),CS_WEIGHTS_DIR=str(weights),CS_VOICES_DIR=str(weights/'voices'),HF_HUB_OFFLINE='1')
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-m','evals.cuda_linker_path','--out',str(a.out/'driver')],text=True).strip()
    from portable.context.native_audition import configure_cuda_toolkit
    configure_cuda_toolkit()
    import torch
    import contextspan.model.engine as native
    from contextspan.moshi.utils.compile import no_compile,no_cuda_graph
    from evals.context_adapter_load import before_engine
    from evals.context_spanning_probe import audited_loads
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_execution import math_attention
    from evals.context_native_audio49_data import digest
    math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    native.seed_all(42424242);loads=[]
    with before_engine(native,a.proof) as adapter,audited_loads(loads),no_compile(),no_cuda_graph():
        engine=native.Engine(checkpoint=str(weights/'context_spanning_7b.pt'),device='cuda',temp=.8,temp_text=.7)
    final=[r for r in loads if r['module_identity']==id(engine.lm)]
    if not final or not final[-1]['strict'] or final[-1]['missing'] or final[-1]['unexpected']:
        raise ValueError('Strict complete native model load required')
    if not getattr(engine,'_dl_checkpoint_native_protocol',None):raise ValueError('Approved bootstrap layout absent')
    voice=torch.load(a.out/'voice.pt',weights_only=True,map_location='cpu')['codes']
    before=tensor_hashes(engine.lm.named_parameters())
    result=run(engine,voice,a.cache,report,a.out)
    if before!=tensor_hashes(engine.lm.named_parameters()):raise ValueError('Read-only Engine audit changed model parameters')
    result.update(all_model_parameters_unchanged=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  assets=str(a.assets),assets_sha256=sha256(a.assets),parent=identity,adapter=adapter,
                  initial_token_sha256=digest(engine.lm._get_initial_token()),independent_revalidation_required=True)
    (a.out/'engine-inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    verify(a.cache,a.out)
    print(json.dumps(dict(complete40=True,optimizer_updates=0,training_approved=False)))


if __name__=='__main__':main()
