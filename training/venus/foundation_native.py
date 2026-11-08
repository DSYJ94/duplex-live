"""Matched native foundations, without changing weights or accepting a subset.

Venus uses its unpatched generation methods and the existing synthetic worker
harness. PersonaPlex uses official offline inference; no worker is attached to
that condition, so its speech cannot count as successful backend execution.
"""
import argparse,functools,gc,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from .large_inputs import sha
from .isambard_native import BASE,ART,guarded

UPSTREAM='16419a2ef7b98611f58c9df07477d00e5ad61d5f'
PP=BASE/'public-models/personaplex-fdaf-v1'
VENDOR=BASE/'vendor/personaplex-16419a-v1'
DATA=ART/'recovered-data-v1/venus-training-v1'


def word_records():
    q=json.loads((DATA/'paced-word-quality-v22.json').read_text())
    if not q['all_passed'] or q['passed']!=q['requested'] or q['requested']!=218:raise ValueError('Original whole218 source-word approval required')
    return {r['source_sha256']:r for r in q['rows']}


def cases(work):
    from .behavior import preflight
    fresh=preflight(work/'evals/fixtures/venus-control-repair-source-v2-20261027.json',DATA/'paced-timelines-v22',DATA/'paced-data-approval-v22.json')
    old=[r for r in preflight(work/'evals/fixtures/venus-control-corpus-v4-20261026.json',DATA/'timelines-v7-repaired',DATA/'data-approval-v7.json') if r['family']=='ordinary']
    reference=BASE/'native-posttrain646-v1/native/main/results.json'
    pin=json.loads((work/'docs/NATIVE-RELEASE-CURRENT32-INPUTS-2026-11-07.json').read_text())
    if sha(reference)!=pin['reference_collection_sha256']:raise ValueError('Original same32 reference changed')
    prior=json.loads(reference.read_text())['cases'];by={r['id']:r for r in fresh+old}
    if len(fresh)!=24 or len(old)!=8 or len(prior)!=32 or len(by)!=32:raise ValueError('Every original32 input required')
    for r in prior:
        if any(by[r['id']][k]!=r[{'frames':'input_frames'}.get(k,k)] for k in ('family','frames','input_sha256')):raise ValueError('Original native input changed')
    return [by[r['id']] for r in prior]


def timed_assembly(frames,arrays,rate,*,frame_seconds=1.):
    import numpy as np
    starts=[round(t['sequence']*frame_seconds*rate) for t in frames]
    out=np.zeros(max([s+len(a) for s,a in zip(starts,arrays)]+[1]),dtype=np.float32)
    for start,a in zip(starts,arrays):out[start:start+len(a)]+=a
    return out


async def collect_venus(a,rows):
    import torch
    from transformers import set_seed
    from portable.venus.native_app import NativeEngine,ensure_guard
    from .behavior import run_case
    from .behavior_review import cue_frame
    voice=Path(a.model)/'assets/HT_ref_audio.wav' if a.variant=='venus-released-reference' else ART/'recovered-data-v1/venus-british-native-v85/references/rp-ben-16k.wav'
    expected=sha(voice)
    engine=NativeEngine(a.model,voice_reference=voice,voice_reference_sha256=expected,nonspoken_actions=False,speech_completion=False,tts_attention='released')
    if engine.action_channel is not None or engine.speech_completion is not None or any(type(m).__name__=='LoRALinear' for m in engine.model.modules()):raise ValueError('Untouched generation path required')
    profile=dict(engine.profile,source=a.source,condition=a.variant,worker_interface='existing synthetic LookupWorker; no model API',native_sampler_patches=False,voice_accent_not_certified=True,adopted=False)
    root=Path(a.root)/a.variant;root.mkdir(exist_ok=False);(root/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    words=word_records();stop=json.loads(Path(a.lease).read_text())['stop_unix'];results=[]
    for case in rows:
        ensure_guard();set_seed(42)
        r=await run_case(engine,case,root,stop)
        if case['family']=='quiet':
            eligible,end=cue_frame(case['user_turns'][-1],words[case['user_turns'][-1]['sha256']])
            trace=[json.loads(s) for s in (root/case['id']/'trace.jsonl').read_text().splitlines()]
            r.update(cue_word_end_seconds=end,voiced_frames_before_cue_word=sum(t['audio_samples']>0 and t['peak']>.001 for t in trace if t['sequence']<eligible),voiced_frames_after_cue_word=sum(t['audio_samples']>0 and t['peak']>.001 for t in trace if t['sequence']>=eligible))
        r['condition']=a.variant;results.append(r)
        (root/'results.json').write_text(json.dumps(dict(cases=results,required_cases=32,qualified=False),indent=2)+'\n')
        print(json.dumps(dict(condition=a.variant,completed=len(results),required=32)),flush=True)
    del engine;gc.collect();torch.cuda.empty_cache()


def collect_personaplex(a,rows):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from moshi import offline
    from portable.venus.native_app import ensure_guard
    if subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM or subprocess.check_output(['git','-C',str(VENDOR),'status','--porcelain'],text=True).strip():raise ValueError('Pinned clean official PersonaPlex code required')
    manifest=json.loads((Path(a.root)/'personaplex-model-manifest.json').read_text())
    for item in manifest['files']+[manifest['voice']]:
        if sha(PP/item['path'])!=item['sha256'] or (PP/item['path']).stat().st_size!=item['bytes']:raise ValueError('Exact preserved PersonaPlex assets required')
    root=Path(a.root)/a.variant;root.mkdir(exist_ok=False)
    # The official helper counts downloads with a config lookup even when all
    # model paths are explicit. Bind that lookup to this pinned local cache.
    cache=root/'local-hub';snapshot=cache/'models--nvidia--personaplex-7b-v1/snapshots'/manifest['model_revision'];snapshot.mkdir(parents=True)
    shutil.copyfile(PP/'config.json',snapshot/'config.json')
    original_download=offline.hf_hub_download
    download=functools.partial(original_download,cache_dir=str(cache),revision=manifest['model_revision'],local_files_only=True)
    if sha(download('nvidia/personaplex-7b-v1','config.json'))!=sha(PP/'config.json'):raise ValueError('Pinned local config resolution differs')
    driver=subprocess.check_output([sys.executable,'-B','-m','evals.cuda_linker_path','--out',str(root/'driver')],text=True).strip()
    os.environ['TRITON_LIBCUDA_PATH']=driver
    (root/'profile.json').write_text(json.dumps(dict(source=a.source,model='nvidia/personaplex-7b-v1',model_revision=manifest['model_revision'],upstream=UPSTREAM,torch=torch.__version__,seed=42,voice=manifest['voice'],native_worker_interface=False,voice_accent_not_certified=True,paced_audio_replay=True,live_microphone_latency=False,optimizer_updates=0,adopted=False),indent=2)+'\n')
    # Preserve raw decoder output before the official WAV export can quantize it.
    original_decode=offline.decode_tokens_to_pcm;original_iterate=offline.lm_iterate_audio
    results=[];word_quality=word_records()
    for case in rows:
        ensure_guard();guarded(a.lease,180);folder=root/case['id'];folder.mkdir();trace=[];holder={}
        audio,sr=sf.read(Path(case['folder'])/'user.wav',dtype='float32');assert sr==16000
        converted=resample_poly(audio,3,2).astype(np.float32);sf.write(folder/'derived-user-24k.wav',converted,24000,subtype='FLOAT')
        def iterate(*args,**kw):
            for i,chunk in enumerate(original_iterate(*args,**kw)):
                if i==0:holder['stream_started']=time.monotonic()
                holder['frame']=i;holder['frame_started']=time.monotonic();yield chunk
                # Same causal paced replay discipline as the Venus collector.
                time.sleep(max(0,holder['stream_started']+(i+1)*.08-time.monotonic()))
        def decode(*args,**kw):
            ensure_guard();decoder_started=time.monotonic();pcm=original_decode(*args,**kw);computed=time.monotonic()
            if not np.isfinite(pcm).all():raise ValueError('Nonfinite raw native PersonaPlex audio')
            i=holder['frame'];name='chunk-%04d.wav'%i;sf.write(folder/name,pcm,24000,subtype='FLOAT')
            trace.append(dict(sequence=i,chunk_file=name,audio_samples=len(pcm),peak=float(np.abs(pcm).max()) if len(pcm) else 0.,elapsed=computed-holder['frame_started'],decoder_seconds=computed-decoder_started,wall_since_start=computed-holder['stream_started'],token_ids=args[3].detach().cpu().flatten().tolist()))
            return pcm
        offline.lm_iterate_audio=iterate;offline.decode_tokens_to_pcm=decode;offline.hf_hub_download=download;started=time.monotonic()
        try:
            with torch.inference_mode():
                offline.run_inference(input_wav=str(folder/'derived-user-24k.wav'),output_wav=str(folder/'official-export.wav'),output_text=str(folder/'official-text-tokens.json'),text_prompt='You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.',voice_prompt_path=str(PP/'NATF2.pt'),tokenizer_path=str(PP/'tokenizer_spm_32k_3.model'),moshi_weight=str(PP/'model.safetensors'),mimi_weight=str(PP/'tokenizer-e351c8d8-checkpoint125.safetensors'),hf_repo='nvidia/personaplex-7b-v1',device='cuda',seed=42,temp_audio=.8,temp_text=.7,topk_audio=250,topk_text=25,greedy=False,save_voice_prompt_embeddings=False,cpu_offload=False)
        finally:offline.lm_iterate_audio=original_iterate;offline.decode_tokens_to_pcm=original_decode;offline.hf_hub_download=original_download
        if not trace:raise ValueError('No actual raw native PersonaPlex output')
        (folder/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
        tokens=json.loads((folder/'official-text-tokens.json').read_text());text=''.join(t for t in tokens if t not in ('EPAD','BOS','EOS','PAD','<s>','</s>')).strip()
        r=dict(id=case['id'],family=case['family'],input_sha256=case['input_sha256'],input_frames=case['frames'],condition=a.variant,complete=True,native_worker_interface=False,actual_worker_events=[],generated_text=text,wall_seconds=time.monotonic()-started,qualified=False)
        voiced=lambda t:t['audio_samples']>0 and t['peak']>.001
        if case['family']=='quiet':
            from .behavior_review import cue_frame
            _,end=cue_frame(case['user_turns'][-1],word_quality[case['user_turns'][-1]['sha256']]);eligible=math.ceil(end/.08)-1
            r.update(cue_word_end_seconds=end,voiced_frames_before_cue_word=sum(voiced(t) for t in trace if t['sequence']<eligible),voiced_frames_after_cue_word=sum(voiced(t) for t in trace if t['sequence']>=eligible))
        if case['family']=='pause':
            turns=case['user_turns'];start=turns[1]['start_seconds'];end=turns[1]['start_seconds']+turns[1]['seconds'];resume=turns[-1]['start_seconds']+turns[-1]['seconds']
            r.update(voiced_before_pause=any(voiced(t) and t['sequence']*.08<start for t in trace),voiced_frames_while_paused=sum(voiced(t) for t in trace if end<=t['sequence']*.08<resume))
        (folder/'result.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r)
        (root/'results.json').write_text(json.dumps(dict(cases=results,required_cases=32,qualified=False),indent=2)+'\n')
        print(json.dumps(dict(condition=a.variant,completed=len(results),required=32)),flush=True);gc.collect();torch.cuda.empty_cache()


def audit_conditions(a, variants):
    import numpy as np,soundfile as sf
    from scipy.signal import resample_poly
    from .voice_compare import recognizer
    from .spoken_audit import references
    from data.venus.pacing import words
    from data.venus.repair import wer
    pipe=recognizer();conditions={};expected={r['id']:r['input_sha256'] for r in cases(Path.cwd())}
    for variant in variants:
        root=Path(a.root)/variant;collected=json.loads((root/'results.json').read_text())['cases']
        if len(collected)!=32 or {c['id']:c['input_sha256'] for c in collected}!=expected:raise ValueError('All32 exact native inputs per condition required')
        audits=[]
        for case in collected:
            guarded(a.lease,90);folder=root/case['id'];traces=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
            arrays=[sf.read(folder/t['chunk_file'],dtype='float32')[0] for t in traces];raw=np.concatenate(arrays)
            if not np.isfinite(raw).all():raise ValueError('Nonfinite original audio')
            wav=folder/'joined-emitted-raw.wav';sf.write(wav,raw,24000,subtype='FLOAT')
            moshi=variant in ('personaplex-release','context-spanning-release','personaplex-context-interface')
            timed=timed_assembly(traces,arrays,24000,frame_seconds=.08 if moshi else 1.)
            sf.write(folder/'timed-emitted-raw.wav',timed,24000,subtype='FLOAT')
            ref=dict(literal_generated=case['generated_text'],intended_user_facing=case['generated_text']) if moshi else references(traces)
            recognized=pipe(dict(array=resample_poly(raw,2,3).astype(np.float32),sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if len(raw) and np.any(raw) else ''
            error=wer(' '.join(words(ref['intended_user_facing'])),' '.join(words(recognized)));peak=float(np.abs(raw).max()) if len(raw) else 0.
            audits.append(dict(id=case['id'],family=case['family'],input_sha256=case['input_sha256'],references=ref,asr_text=recognized,word_error_rate=error,word_pass=error<=.05,raw_peak=peak,raw_peak_pass=peak<=.995,has_voiced_output=bool(np.any(np.abs(raw)>.001)),timed_raw_peak=float(np.abs(timed).max()),raw_wav_sha256=sha(wav),timed_raw_wav_sha256=sha(folder/'timed-emitted-raw.wav'),source_chunks=[dict(file=t['chunk_file'],sha256=sha(folder/t['chunk_file'])) for t in traces],native_worker_interface=not moshi,semantic_or_naturalness_pass_claimed=False))
            (root/'audit-progress.json').write_text(json.dumps(dict(required=32,cases=audits),indent=2)+'\n')
        quiet=[r for r in collected if r['family']=='quiet'];pause=[r for r in collected if r['family']=='pause']
        summary=dict(cases=32,word_pass=sum(r['word_pass'] for r in audits),raw_peak_pass=sum(r['raw_peak_pass'] for r in audits),joint_pass=sum(r['word_pass'] and r['raw_peak_pass'] for r in audits),without_voiced_output=sum(not r['has_voiced_output'] for r in audits),necessary_silence_and_resume=sum(r['voiced_frames_before_cue_word']==0 and r['voiced_frames_after_cue_word']>0 for r in quiet),silent_while_paused=sum(r['voiced_frames_while_paused']==0 for r in pause))
        report=dict(complete=True,source=a.source,summary=summary,cases=audits,word_limit=.05,peak_limit=.995,adopted=False)
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');conditions[variant]=summary
    return conditions


def audit(a):
    conditions=audit_conditions(a,('venus-released-reference','venus-rp-reference','personaplex-release'))
    (Path(a.root)/'comparison.json').write_text(json.dumps(dict(complete_all3_times32=True,source=a.source,conditions=conditions,optimizer_updates=0,external_API_calls=0,PersonaPlex_backend_execution_not_tested=True,naturalness_and_semantics_require_review=True,adopted=False),indent=2)+'\n')
    print(json.dumps(conditions),flush=True)


def main(a):
    from portable.venus.native_app import ensure_guard
    ensure_guard();guarded(a.lease,180)
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source or json.loads(Path(a.lease).read_text())['source']!=a.source:raise ValueError('Actual Git/Slurm source required')
    if a.variant=='audit':return audit(a)
    rows=cases(Path.cwd())
    if a.variant=='personaplex-release':return collect_personaplex(a,rows)
    import asyncio
    return asyncio.run(collect_venus(a,rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','source','lease','model'):p.add_argument('--'+n,required=True)
    p.add_argument('--variant',required=True,choices=('venus-released-reference','venus-rp-reference','personaplex-release','audit'));main(p.parse_args())
