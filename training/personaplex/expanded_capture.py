"""Capture all1072 native streams with the selected British reference.

Teacher forcing records inputs; it does not qualify generated speech or learn.
Every source frame, fold and prefix-free loss mask is independently reconstructed.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
from evals.moshi_model_download import sha256 as sha
from .expanded_codec import inputs
from .expanded_guard import check

BASE=Path('/projects/u6ada/duplex-live')
PP=BASE/'public-models/personaplex-fdaf-v1'
VENDOR=BASE/'vendor/context-spanning-79df-v1'
WEIGHTS=BASE/'public-models/context-spanning-e62a-v1'
REFERENCE=BASE/'artifacts/native-v75/recovered-data-v1/venus-british-native-v85/references/rp-ben-16k.wav'
REFERENCE_SHA='552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0'
INSTRUCTIONS=('You are a helpful English-speaking voice assistant. Answer ordinary and straightforward questions directly. '
    'For difficult work or outside information, ask the background assistant to help while you keep listening. '
    'Do not invent results. Respect requests to remain silent, pause, resume or cancel pending work.')


def preflight(a):
    _,rows=inputs(a.data);cache=json.loads((a.cache/'cache.json').read_text())
    from . import expanded_codec
    if not cache['complete1072'] or cache['TRAIN']!=1008 or cache['DEV']!=64 or cache['optimizer_updates']!=0 or cache['source_manifest_sha256']!=sha(a.data/'manifest.json') or cache['code_sha256']!=sha(Path(expanded_codec.__file__)) or [r['id']for r in cache['rows']]!=[r['id']for r in rows]:raise ValueError('Complete actual independently verified native1072 cache required')
    if sha(REFERENCE)!=REFERENCE_SHA or any(p['sha256']==REFERENCE_SHA for r in rows for p in r['parents']):raise ValueError('Exact disjoint selected RPBen reference required')
    for entry,row in zip(cache['rows'],rows):
        path=a.cache/('shard-%02d'%entry['shard'])/entry['file']
        if path.name!=row['id']+'.pt' or sha(path)!=entry['sha256'] or entry['fold']!=row['fold']:raise ValueError('Native source artifact/fold changed')
    return rows,cache


def capture(a):
    import numpy as np,soundfile as sf,torch
    from evals.native_control_engine import assemble,delayed,trace,validate_trace,DELAYS,GRAPH_POLICY
    from evals.context_native_audio49_data import digest
    from evals.context_execution import math_attention
    from evals.context_native_layout import install
    from evals.context_spanning_probe import audited_loads
    from evals.context_spanning_download import SOURCE_COMMIT,WEIGHT_SHA256
    from evals.personaplex_train import MODEL_HASHES
    from evals.moshi_training_smoke import tensor_hashes
    from data.personaplex.expanded_source_export import resample_native
    check(a.lease,240);rows,cache=preflight(a)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source or not 0<=a.shard<a.shards<=16:raise ValueError('Exact committed distinct capture shard required')
    if subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT or subprocess.check_output(['git','-C',str(VENDOR),'status','--porcelain'],text=True).strip() or sha(WEIGHTS/'context_spanning_7b.pt')!=WEIGHT_SHA256 or any(sha(PP/n)!=v for n,v in MODEL_HASHES.items()):raise ValueError('Exact unmodified published native model/codec/tokenizer required')
    out=a.root/('shard-%02d'%a.shard);out.mkdir(parents=True,exist_ok=False)
    os.environ.update(CS_BASE_DIR=str(PP),CS_WEIGHTS_DIR=str(WEIGHTS),CS_VOICES_DIR=str(WEIGHTS/'voices'),HF_HUB_OFFLINE='1');sys.path.insert(0,str(VENDOR))
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-B','-m','evals.cuda_linker_path','--out',str(out/'driver')],text=True).strip()
    from contextspan.model.engine import Engine,seed_all
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;seed_all(42424242);loads=[]
    with audited_loads(loads):engine=Engine(checkpoint=str(WEIGHTS/'context_spanning_7b.pt'),device='cuda',temp=.8,temp_text=.7)
    final=[r for r in loads if r['module_identity']==id(engine.lm)]
    if not final or not final[-1]['strict'] or final[-1]['missing'] or final[-1]['unexpected'] or engine.lm.dep_q!=16 or list(engine.lm.delays)!=DELAYS or engine.frame_size!=1920 or engine.frame_rate!=12.5:raise ValueError('Full strict native load and original stream/delay clock required')
    install(engine);engine.lm.requires_grad_(False);before=tensor_hashes(engine.lm.named_parameters())
    raw,rate=sf.read(REFERENCE,dtype='float32');reference=resample_native(raw,rate);padding=(-len(reference))%1920;reference=np.pad(reference,(0,padding));sf.write(out/'reference24k.wav',reference,24000,subtype='FLOAT');engine.reset()
    with torch.no_grad():voice=engine.mimi.encode(torch.as_tensor(reference[None,None],device='cuda')).detach().cpu()[0]
    if voice.shape!=(8,len(reference)//1920) or voice.dtype!=torch.int64:raise ValueError('Complete selected reference codebooks required')
    torch.save(dict(codes=voice),out/'voice.pt');prefix=persona_prefix(INSTRUCTIONS,voice,engine.spm);initial=engine.lm._get_initial_token().detach().cpu()
    report=dict(source=a.source,code_sha256=sha(Path(__file__)),cache_sha256=sha(a.cache/'cache.json'),shard=a.shard,shards=a.shards,pid=os.getpid(),reference_source_sha256=REFERENCE_SHA,reference_role='REFERENCE_ONLY',reference_voice_id='yf0u5Jay5VVCpmAhFyo4',reference_resampling='full16kto24k-polyphase3/2; no gain/trim',reference_tail_padding=padding,reference24k_sha256=sha(out/'reference24k.wav'),voice_sha256=sha(out/'voice.pt'),prefix_sha256=digest(prefix),initial_token_sha256=digest(initial),instructions=INSTRUCTIONS,native_graph_policy=GRAPH_POLICY,cases=[],complete_shard=False,optimizer_updates=0,native_training_approved=False)
    for source,entry in list(zip(rows,cache['rows']))[a.shard::a.shards]:
        check(a.lease,120);folder=out/entry['id'];folder.mkdir();clean=torch.load(a.cache/('shard-%02d'%entry['shard'])/entry['file'],weights_only=True,map_location='cpu')['codes'][0];wave,rate=sf.read(a.data/source['path'],dtype='float32',always_2d=True)
        if rate!=24000 or wave.shape!=(clean.shape[1]*1920,2):raise ValueError('Whole original stereo source required')
        saved,calls,decoded,execution=trace(engine,voice,clean,wave[:,1],entry['id']);codes,mask=assemble(clean,prefix,saved['streamed_user_codes']);offline,_=assemble(clean,prefix)
        saved.update(codes=codes,mask=mask,targets=codes[0],offline_inputs=delayed(offline,_delay_sequence,initial));ownership=validate_trace(clean,prefix,saved,calls,_delay_sequence,initial)
        torch.save(saved,folder/'inputs.pt');sf.write(folder/'teacher-forced-native.wav',decoded,24000,subtype='FLOAT')
        report['cases'].append(dict(id=entry['id'],fold=entry['fold'],calls=calls,ownership=ownership,inputs_sha256=sha(folder/'inputs.pt'),source_cache_sha256=entry['sha256'],teacher_forced_wav_sha256=sha(folder/'teacher-forced-native.wav'),native_execution=execution))
        (out/'capture.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(shard=a.shard,completed=len(report['cases']),required=len(rows[a.shard::a.shards]))),flush=True)
    if before!=tensor_hashes(engine.lm.named_parameters()):raise ValueError('Capture changed model parameters')
    report.update(complete_shard=True,all_model_parameters_unchanged=True);(out/'capture.json').write_text(json.dumps(report,indent=2)+'\n')


def verify(a):
    import torch,sentencepiece,numpy as np,soundfile as sf
    from evals.native_control_engine import assemble,validate_trace
    from evals.context_native_audio49_data import digest
    rows,cache=preflight(a);sys.path.insert(0,str(VENDOR));os.environ['CS_BASE_DIR']=str(PP)
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import _delay_sequence
    spm=sentencepiece.SentencePieceProcessor(str(PP/'tokenizer_spm_32k_3.model'));initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000;result=[];all_rows={r['id']:r for r in rows}
    for rank in range(a.shards):
        out=a.root/('shard-%02d'%rank);report=json.loads((out/'capture.json').read_text())
        if not report['complete_shard'] or not report['all_model_parameters_unchanged'] or report['optimizer_updates']!=0 or report['source']!=a.source or report['shard']!=rank or report['shards']!=a.shards or report['code_sha256']!=sha(Path(__file__)) or report['cache_sha256']!=sha(a.cache/'cache.json') or report['pid']==os.getpid() or report['reference_source_sha256']!=REFERENCE_SHA or report['instructions']!=INSTRUCTIONS or sha(out/'voice.pt')!=report['voice_sha256'] or sha(out/'reference24k.wav')!=report['reference24k_sha256']:raise ValueError('Exact complete distinct-process capture required')
        voice=torch.load(out/'voice.pt',weights_only=True,map_location='cpu')['codes'];prefix=persona_prefix(INSTRUCTIONS,voice,spm)
        if digest(prefix)!=report['prefix_sha256'] or digest(initial)!=report['initial_token_sha256']:raise ValueError('Original prefix/bootstrap changed')
        if [r['id']for r in report['cases']]!=[r['id']for r in rows[rank::a.shards]]:raise ValueError('Whole unchanged shard membership required')
        for case,entry in zip(report['cases'],cache['rows'][rank::a.shards]):
            path=out/case['id']/'inputs.pt';clean=torch.load(a.cache/('shard-%02d'%entry['shard'])/entry['file'],weights_only=True,map_location='cpu')['codes'][0];saved=torch.load(path,weights_only=True,map_location='cpu')
            if sha(path)!=case['inputs_sha256'] or case['source_cache_sha256']!=entry['sha256'] or case['fold']!=all_rows[case['id']]['fold'] or validate_trace(clean,prefix,saved,case['calls'],_delay_sequence,initial)!=case['ownership'] or not case['native_execution']['cuda_main_graph_captured']:raise ValueError('Independent all-frame native ownership reconstruction failed')
            wave=out/case['id']/'teacher-forced-native.wav'
            if sha(wave)!=case['teacher_forced_wav_sha256']:raise ValueError('Native teacher waveform changed')
            pcm,rate=sf.read(wave,dtype='float32')
            if rate!=24000 or pcm.shape!=(clean.shape[1]*1920,) or not np.isfinite(pcm).all():raise ValueError('Full finite native teacher PCM required')
            codes,mask=assemble(clean,prefix,saved['streamed_user_codes'])
            if not torch.equal(codes,saved['codes']) or not torch.equal(mask,saved['mask']):raise ValueError('Native teacher targets changed')
            result.append(dict(case,shard=rank,inputs_file=str(path),native_qualification=True))
    if len(result)!=1072 or len({r['id']for r in result})!=1072:raise ValueError('All1072 independent native records required')
    report=dict(source=a.source,complete1072=True,TRAIN=sum(r['fold']=='TRAIN'for r in result),DEV=sum(r['fold']=='DEV'for r in result),rows=result,verification_pid=os.getpid(),cache_sha256=sha(a.cache/'cache.json'),optimizer_updates=0,native_gradient_restore_and_behavior_still_required=True)
    with (a.root/'inputs.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['capture','verify'])
    for n in ('data','cache','root','lease'):p.add_argument('--'+n,type=Path,required=n!='lease')
    p.add_argument('--source',required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--shard',type=int,default=0);a=p.parse_args();(capture if a.stage=='capture'else verify)(a)
