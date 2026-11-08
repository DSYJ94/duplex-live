"""Isolate the published Context Spanning overlay from the PersonaPlex weights.

Both conditions use the same released Context Spanning engine, voice, persona,
sampler and complete frozen32 inputs. No worker, forced answer or weight update.
The earlier official PersonaPlex32 outputs remain an independent interface control.
"""
import argparse,inspect,json,os,subprocess,sys,time
from pathlib import Path
from training.venus.foundation_native import cases,word_records,audit_conditions,PP
from training.venus.isambard_native import BASE,guarded
from training.venus.large_inputs import sha
from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256,VOICE_SHA256

VENDOR=BASE/'vendor/context-spanning-79df-v1'
WEIGHTS=BASE/'public-models/context-spanning-e62a-v1'
VARIANTS=('personaplex-context-interface','context-spanning-release')
PERSONA='You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.'


def raw_model(weights,device='meta'):
    """Identical vendor geometry, with strict loading instead of its fill policy."""
    import torch
    kwargs=dict(weights.loaders._lm_kwargs)
    kwargs['dep_q']=weights.loaders._infer_dep_q(str(PP/weights.loaders.MOSHI_NAME),default=16)
    model=weights.loaders.LMModel(device='meta',dtype=torch.bfloat16,**kwargs)
    if device=='meta':return model
    state=weights.loaders.load_file(str(PP/weights.loaders.MOSHI_NAME),device=torch.device(device).type)
    model.load_state_dict(state,strict=True,assign=True)
    return model.to(device=device,dtype=torch.bfloat16).eval()


def check_raw_shapes(model,shapes):
    actual={k:tuple(v.shape) for k,v in model.state_dict().items()}
    if actual!={k:tuple(v) for k,v in shapes.items()}:raise ValueError('Exact raw checkpoint keys and every native shape required')
    return dict(full_key_and_shape_coverage=True,tensors=len(actual),native_depth_heads=model.dep_q)


def preflight(work):
    if subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT or subprocess.check_output(['git','-C',str(VENDOR),'status','--porcelain'],text=True).strip():raise ValueError('Exact clean Context Spanning source required')
    manifest=json.loads((WEIGHTS/'original-manifest.json').read_text())
    if manifest['revision']!=MODEL_REVISION or manifest['source_commit']!=SOURCE_COMMIT:raise ValueError('Original released Context Spanning identity required')
    for name,digest in (('context_spanning_7b.pt',WEIGHT_SHA256),('voices/main.pt',VOICE_SHA256)):
        if manifest['files'][name]['sha256']!=digest:raise ValueError('Wrong original released asset digest')
    for name,record in manifest['files'].items():
        if sha(WEIGHTS/name)!=record['sha256'] or (WEIGHTS/name).stat().st_size!=record['bytes']:raise ValueError('Changed released asset')
    base=json.loads((BASE/'native-foundation-native-v1/personaplex-model-manifest.json').read_text())
    for r in base['files']:
        if sha(PP/r['path'])!=r['sha256'] or (PP/r['path']).stat().st_size!=r['bytes']:raise ValueError('Changed original PersonaPlex asset')
    rows=cases(work)
    if len(rows)!=32:raise ValueError('Complete original32 required')
    os.environ.update(CS_BASE_DIR=str(PP),CS_WEIGHTS_DIR=str(WEIGHTS),CS_VOICES_DIR=str(WEIGHTS/'voices'))
    sys.path.insert(0,str(VENDOR))
    from contextspan.model.engine import Engine
    from contextspan.model import weights
    from safetensors import safe_open
    if Path(inspect.getfile(Engine)).resolve()!=(VENDOR/'contextspan/model/engine.py').resolve():raise ValueError('Wrong released engine imported')
    with safe_open(str(PP/weights.loaders.MOSHI_NAME),framework='pt',device='cpu') as f:
        shapes={k:f.get_slice(k).get_shape() for k in f.keys()}
    check_raw_shapes(raw_model(weights),shapes)
    return rows


def collect(a,rows):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from portable.venus.native_app import ensure_guard
    from contextspan.model import engine as engine_module,weights
    from contextspan.model.engine import seed_all
    from evals.context_spanning_probe import audited_loads
    from training.venus.behavior_review import cue_frame
    root=Path(a.root)/a.variant;root.mkdir(exist_ok=False)
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-B','-m','evals.cuda_linker_path','--out',str(root/'driver')],text=True).strip()
    records=[];original=engine_module.load_model
    def base_only(checkpoint=None,device='cuda',cpu_offload=False):
        if cpu_offload:raise ValueError('No offload in this matched comparison')
        return raw_model(weights,device),weights.load_mimi(device),weights.load_tokenizer()
    if a.variant=='personaplex-context-interface':engine_module.load_model=base_only
    seed_all(42)
    try:
        with audited_loads(records):engine=engine_module.Engine(checkpoint=str(WEIGHTS/'context_spanning_7b.pt'),device='cuda',temp=.8,temp_text=.7)
    finally:engine_module.load_model=original
    loads=[r for r in records if r['module_identity']==id(engine.lm)]
    if not loads or not loads[-1]['strict'] or loads[-1]['missing'] or loads[-1]['unexpected'] or engine.lm.dep_q!=16 or engine.frame_size!=1920 or engine.frame_rate!=12.5:raise ValueError('Strict complete native model and original clock required')
    voice=torch.load(WEIGHTS/'voices/main.pt',map_location='cpu',weights_only=True)['codes'].long()
    if voice.ndim!=2 or voice.shape[0]!=8:raise ValueError('Released native voice code shape differs')
    profile=dict(source=a.source,condition=a.variant,upstream=SOURCE_COMMIT,overlay_revision=MODEL_REVISION if a.variant=='context-spanning-release' else None,overlay_sha256=WEIGHT_SHA256 if a.variant=='context-spanning-release' else None,voice_sha256=VOICE_SHA256,persona=PERSONA,seed=42,temp_audio=.8,temp_text=.7,loads=loads,torch=torch.__version__,native_worker_interface=False,voice_accent_not_certified=True,optimizer_updates=0,adopted=False)
    (root/'profile.json').write_text(json.dumps(profile,indent=2)+'\n');word_quality=word_records();results=[]
    with torch.inference_mode():
        for case in rows:
            ensure_guard();guarded(a.lease,180);seed_all(42);engine.reset();engine.set_persona(PERSONA,voice)
            folder=root/case['id'];folder.mkdir();audio,sr=sf.read(Path(case['folder'])/'user.wav',dtype='float32')
            if sr!=16000 or audio.ndim!=1 or sha(Path(case['folder'])/'user.wav')!=case['input_sha256']:raise ValueError('Original complete input wave required')
            user=resample_poly(audio,3,2).astype(np.float32);padding=(-len(user))%1920;user=np.pad(user,(0,padding));sf.write(folder/'derived-user-24k.wav',user,24000,subtype='FLOAT')
            trace=[];tokens=[];started=time.monotonic()
            for i,offset in enumerate(range(0,len(user),1920)):
                ensure_guard();before=time.monotonic();out=engine.step(user[offset:offset+1920]);elapsed=time.monotonic()-before
                pcm=np.asarray(out['agent_pcm'] if out['agent_pcm'] is not None else [],dtype=np.float32)
                if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Finite original native audio required')
                name='chunk-%04d.wav'%i;sf.write(folder/name,pcm,24000,subtype='FLOAT');tokens.append(out['text_token'])
                trace.append(dict(sequence=i,chunk_file=name,audio_samples=len(pcm),peak=float(np.abs(pcm).max()) if len(pcm) else 0.,elapsed=elapsed,wall_since_start=time.monotonic()-started,text_token=out['text_token'],is_ret=out['is_ret']))
                time.sleep(max(0,started+(i+1)*.08-time.monotonic()))
            (folder/'trace.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in trace))
            r=dict(id=case['id'],family=case['family'],input_sha256=case['input_sha256'],input_frames=case['frames'],condition=a.variant,complete=True,native_worker_interface=False,actual_worker_events=[],native_ret_tokens=sum(t['is_ret'] for t in trace),generated_text=engine.decode_text(tokens),wall_seconds=time.monotonic()-started,derived_tail_padding_samples=padding,qualified=False)
            voiced=lambda t:t['audio_samples']>0 and t['peak']>.001
            if case['family']=='quiet':
                _,end=cue_frame(case['user_turns'][-1],word_quality[case['user_turns'][-1]['sha256']]);eligible=__import__('math').ceil(end/.08)-1
                r.update(cue_word_end_seconds=end,voiced_frames_before_cue_word=sum(voiced(t) for t in trace if t['sequence']<eligible),voiced_frames_after_cue_word=sum(voiced(t) for t in trace if t['sequence']>=eligible))
            if case['family']=='pause':
                turns=case['user_turns'];start=turns[1]['start_seconds'];end=start+turns[1]['seconds'];resume=turns[-1]['start_seconds']+turns[-1]['seconds']
                r.update(voiced_before_pause=any(voiced(t) and t['sequence']*.08<start for t in trace),voiced_frames_while_paused=sum(voiced(t) for t in trace if end<=t['sequence']*.08<resume))
            (folder/'result.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r)
            (root/'results.json').write_text(json.dumps(dict(cases=results,required_cases=32,qualified=False),indent=2)+'\n');print(json.dumps(dict(condition=a.variant,completed=len(results),required=32)),flush=True)


def main(a):
    if a.preflight_only:
        rows=preflight(Path.cwd());print(json.dumps(dict(complete32=len(rows)==32,source=a.source,actual_CPU_preflight=True,optimizer_updates=0)));return
    from portable.venus.native_app import ensure_guard
    ensure_guard();guarded(a.lease,180)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source or json.loads(Path(a.lease).read_text())['source']!=a.source:raise ValueError('Exact actual execution source required')
    if a.variant=='audit':
        summaries=audit_conditions(a,VARIANTS)
        (Path(a.root)/'comparison.json').write_text(json.dumps(dict(complete_all2_times32=True,source=a.source,conditions=summaries,worker_execution_tested=False,voice_accent_not_certified=True,optimizer_updates=0,external_API_calls=0,adopted=False),indent=2)+'\n');print(json.dumps(summaries),flush=True);return
    collect(a,preflight(Path.cwd()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','source'):p.add_argument('--'+n,required=True)
    p.add_argument('--lease');p.add_argument('--variant',choices=VARIANTS+('audit',));p.add_argument('--preflight-only',action='store_true');main(p.parse_args())
