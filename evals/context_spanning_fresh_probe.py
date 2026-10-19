"""Fixed native fresh20 inference, isolated from hash-pinned TRAIN history probes.

Only approved new user waveform and native actual backend receipts enter the
model. Scoring references never enter inference; no history/targets are recorded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evals.moshi_model_download import sha256
from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256,VOICE_SHA256


from evals.context_spanning_probe import audited_loads


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','base','data','out','native-fresh-quality','adapter-proof','voice-reference','voice-provenance'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--scenario',required=True)
    ap.add_argument('--native-fresh-behavior-review',type=Path)
    ap.add_argument('--native-fresh-candidate-proof',type=Path)
    ap.add_argument('--expected-kind',choices=('private','simple','clarify','cancel'),required=True)
    ap.add_argument('--backend-mode',choices=('native-asr','observe-only'),required=True)
    ap.add_argument('--asr-python',type=Path)
    ap.add_argument('--lookup-delivery-delay-s',type=float,default=0.)
    ap.set_defaults(voice='accepted-synthetic',private_value=None,train_history=False,
        ret_logit_bias=0.,warm_audio=False,pending_status=False,adapter_disabled=False,adapter_zero=False,
        layout_audit=None,event_log_mode='immediate',backend_reference_profile='answer-only',
        persona_profile='recorded-records',math_attention=False)
    args=ap.parse_args()
    from evals.native_fresh_probe import validate_options
    validate_options(args)
    if not 0<=args.lookup_delivery_delay_s<=30:ap.error('Bounded predeclared lookup delay required')
    if args.backend_mode=='native-asr':
        devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        if len(devices)!=2 or len(set(devices))!=2 or not args.asr_python:
            ap.error('Native-ASR consultation requires two distinct assigned GPUs and its isolated Python')
        os.environ['CUDA_VISIBLE_DEVICES']=devices[0];args.asr_gpu=devices[1]
    import numpy as np
    import soundfile as sf
    import torch
    from evals.personaplex_train import MODEL_HASHES
    from evals.recovery_worker import RecordWorld
    from evals.lychee_context import BASE_INSTRUCTIONS
    assets=json.loads(args.assets.read_text());source=Path(assets['source']);weights=Path(assets['path'])
    if (assets['source_commit']!=SOURCE_COMMIT or assets['revision']!=MODEL_REVISION
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Context Spanning source/revision changed')
    for name,record in assets['files'].items():
        if sha256(weights/name)!=record['sha256']:raise RuntimeError('Context Spanning asset changed')
    if (assets['files']['context_spanning_7b.pt']['sha256']!=WEIGHT_SHA256
            or assets['files']['voices/main.pt']['sha256']!=VOICE_SHA256):
        raise RuntimeError('Assets differ from pinned upstream LFS digests')
    for name,digest in MODEL_HASHES.items():
        if sha256(args.base/name)!=digest:raise RuntimeError('PersonaPlex base asset changed')
    from evals.native_fresh_probe import load as load_fresh
    fresh,row=load_fresh(args)
    audio,rate=sf.read(row['path'],dtype='float32')
    if rate!=24000 or audio.ndim!=1 or not np.isfinite(audio).all():raise RuntimeError('Invalid fresh native mono source')
    from evals.native_fresh_probe import user_stream
    user=user_stream(audio,rate,fresh);args.source_split='fresh-evaluation'
    args.out.mkdir(exist_ok=False)
    sf.write(args.out/'user_in.wav',user,rate)
    os.environ.update(CS_BASE_DIR=str(args.base),CS_WEIGHTS_DIR=str(weights),CS_VOICES_DIR=str(weights/'voices'))
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-m','evals.cuda_linker_path',
        '--out',str(args.out/'driver')],text=True).strip()
    checkpoint=weights/'context_spanning_7b.pt'
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    if unsafe:raise RuntimeError('Checkpoint requires unsupported pickle globals: '+repr(unsafe))
    sys.path.insert(0,str(source))
    from contextspan.model.engine import Engine,seed_all
    import contextspan.model.engine as engine_module
    from evals.context_adapter_load import before_engine
    from contextspan.model.weights import load_voice
    import inspect
    if Path(inspect.getfile(Engine)).resolve()!=(source/'contextspan/model/engine.py').resolve():
        raise RuntimeError('Wrong Context Spanning engine module imported')
    records=[];started=time.monotonic()
    with before_engine(engine_module,args.adapter_proof,args.adapter_disabled,args.adapter_zero) as adapter:
        with audited_loads(records):
            engine=Engine(checkpoint=str(checkpoint),device='cuda',temp=.8,temp_text=.7)
            voice=load_voice('main')
    lm_loads=[r for r in records if r['module_identity']==id(engine.lm)]
    if (type(engine.lm).load_state_dict is not torch.nn.Module.load_state_dict
            or not lm_loads or not lm_loads[-1]['strict'] or lm_loads[-1]['missing'] or lm_loads[-1]['unexpected']):
        raise RuntimeError('Final Context Spanning checkpoint did not strictly cover native LM')
    if engine.frame_size!=1920 or engine.frame_rate!=12.5:raise RuntimeError('Native frame clock changed')
    layout=None
    if getattr(engine,'_dl_checkpoint_native_protocol',None):
        layout={'native_layout':'bootstrap prefix and preserve provided acoustics',
            'checkpoint_required':True,'native_protocol':engine._dl_checkpoint_native_protocol,
            'quality_qualified':False}
        from evals.context_span_trace import attach
        layout['actual_free_running_span_checks']=attach(engine)
    if args.voice=='accepted-synthetic':
        if not args.voice_reference or not args.voice_provenance:raise RuntimeError('Synthetic voice provenance required')
        receipt=json.loads(args.voice_provenance.read_text())
        if (receipt['schema']!='dl-voice-codec-probe-v1' or receipt['source_model']!='gpt-live-1'
                or receipt['reference_sha256']!=sha256(args.voice_reference)):
            raise RuntimeError('Accepted synthetic voice provenance changed')
        reference,voice_rate=sf.read(args.voice_reference,dtype='float32')
        if reference.ndim!=1 or not np.isfinite(reference).all():raise RuntimeError('Invalid synthetic reference')
        if args.adapter_proof:
            if adapter['voice_reference_sha256']!=sha256(args.voice_reference):
                raise RuntimeError('Adapter training voice differs from audition voice')
            voice=torch.load(args.adapter_proof/'voice.pt',map_location='cpu',weights_only=True)['codes']
        else:
            voice=engine.clone_voice(reference,voice_rate)
    elif args.adapter_proof:
        raise RuntimeError('Adapter comparisons require the exact accepted synthetic voice codes')
    warm=None
    seed_all(42424242);engine.reset()
    persona=BASE_INSTRUCTIONS
    engine.set_persona(persona,voice)
    torch.cuda.synchronize()
    loading={'loads':records,'final_lm_strict':True,'dep_q':engine.lm.dep_q,
        'scenario':args.scenario,'scenario_kind':row['kind'],
        'model_load_and_prefix_s':time.monotonic()-started,'pickle_globals':unsafe,
        'assets_manifest_sha256':sha256(args.assets),'base_hashes':MODEL_HASHES,
        'voice':args.voice,'voice_codes_shape':list(voice.shape),'input_warmup':warm,'adapter':adapter,'layout':layout,
        'persona_profile':args.persona_profile,'persona_text':persona,
        'persona_text_sha256':hashlib.sha256(persona.encode()).hexdigest(),
        'lookup_delivery_minimum_delay_s':args.lookup_delivery_delay_s,
        'torch':torch.__version__,'torch_file':torch.__file__,'cuda':torch.version.cuda}
    loading['fresh_evaluation_input']=fresh
    (args.out/'loading.json').write_text(json.dumps(loading,indent=2)+'\n')
    world=RecordWorld(fresh['worker_only_world'])
    if args.backend_mode=='observe-only':
        from evals.native_fresh_probe import observe,finish
        observe(engine,user,args,loading);finish(args,fresh)
        return
    if args.backend_mode=='native-asr':
        from evals.context_spanning_native import run
        run(engine,user,world,args,loading)
        from evals.native_fresh_probe import finish
        finish(args,fresh)
        return


if __name__=='__main__':main()
