"""Twelve sealed native speech observations; unchanged sampling and zero updates."""
import argparse,base64,gc,hashlib,json,time,wave
from contextlib import nullcontext
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    import numpy as np,soundfile as sf,torch
    from transformers import set_seed
    from scipy.signal import resample_poly
    from portable.venus.native_app import NativeEngine,ensure_guard
    from portable.venus.tts_observer import observe_duplex
    from .lora import disabled
    from .voice_compare import recognizer,REVISION
    from .spoken_audit import references
    from data.venus.repair import wer
    from data.venus.pacing import words
    ensure_guard();end=json.loads(Path(a.lease).read_text())['stop_unix']
    assert end-time.time()>getattr(a,'minimum_start_seconds',900)
    selected=json.loads(Path(a.selection).read_text());assert len(selected['scenes'])==6 and selected['required_native_cases']==12
    assert selected['conditions']==['untouched-release','candidate466'] and selected['terminal_budget_unchanged']==26
    assert sha(a.checkpoint)=='e2c16cb825dc654519e922379a220b8c962e9d8a78eec652a3b195922a69720f'
    root=Path(a.timelines)
    for s in selected['scenes']:
        assert s['family']=='ordinary' and sha(root/s['id']/'user.wav')==s['user_wav_sha256'] and sha(root/s['id']/'timeline.json')==s['timeline_sha256']
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    engine=NativeEngine(a.model,adapter_checkpoint=a.checkpoint,voice_reference=a.voice_reference,voice_reference_sha256=a.voice_reference_sha256,tts_attention='released',speech_completion=getattr(a,'speech_completion',False))
    engine.model.requires_grad_(False)
    rows=[]
    for condition in selected['conditions']:
        for s in selected['scenes']:
            ensure_guard();assert end-time.time()>450
            folder=out/condition/s['id'];folder.mkdir(parents=True)
            with wave.open(str(root/s['id']/'user.wav')) as wav:
                assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getnframes())==(16000,1,2,s['frames']*16000)
                pcm=wav.readframes(wav.getnframes())
            traces=[];audio=[]
            with disabled(engine.model.llm) if condition=='untouched-release' else nullcontext():
                set_seed(42);engine.prepare()
                for index in range(s['frames']):
                    ensure_guard()
                    with observe_duplex(engine.duplex) as observed:
                        frame=engine.step(pcm[index*32000:(index+1)*32000],None)
                    if any('observer_error' in r for r in observed['sampler']):raise RuntimeError('Actual sampler observation failed; no successful-subset report')
                    encoded=frame.pop('audio')
                    samples=np.zeros(frame['audio_samples'],dtype=np.float32) if frame['wire_silence_omitted'] else np.frombuffer(base64.b64decode(encoded),dtype='<f4')
                    assert len(samples)==frame['audio_samples'] and np.isfinite(samples).all()
                    frame.update(sequence=index,tts_stop_observations=observed)
                    traces.append(frame);audio.append(samples)
                    sf.write(folder/('chunk-%03d.wav'%index),samples,24000,subtype='FLOAT')
            (folder/'trace.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
            joined=folder/'emitted.wav';sf.write(joined,np.concatenate(audio),24000,subtype='FLOAT')
            row=dict(acoustic_pending_at_end=(traces[-1].get('speech_completion',{}).get('pending') if getattr(a,'speech_completion',False) else None),condition=condition,id=s['id'],fold=s['fold'],input_sha256=s['user_wav_sha256'],frames=len(traces),trace_sha256=sha(folder/'trace.jsonl'),joined_wav_sha256=sha(joined),references=references(traces),raw_peak=float(np.abs(np.concatenate(audio)).max()),sampler_calls=[r for t in traces for r in t['tts_stop_observations']['sampler']],qualified=False)
            rows.append(row);(out/'progress.json').write_text(json.dumps(dict(complete=False,rows=rows,required=12),indent=2)+'\n')
            print(json.dumps(dict(event='tts_stop_native_case',condition=condition,id=s['id'],completed=len(rows),required=12,sampler_calls=row['sampler_calls'])),flush=True)
    profile=dict(engine.profile)
    del engine;gc.collect();torch.cuda.empty_cache();pipe=recognizer()
    for row in rows:
        ensure_guard();assert end-time.time()>180
        wav=out/row['condition']/row['id']/'emitted.wav';data,sr=sf.read(wav,dtype='float32');assert sr==24000
        text=pipe(dict(array=resample_poly(data,2,3).astype(np.float32),sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if np.any(data) else ''
        row.update(asr_text=text,wer=wer(' '.join(words(row['references']['intended_user_facing'])),' '.join(words(text))))
    report=dict(complete=len(rows)==12,required_cases=12,rows=rows,source=a.source,selection_sha256=sha(a.selection),checkpoint_sha256=sha(a.checkpoint),asr_revision=REVISION,new_optimizer_updates=0,new_external_model_API_calls=0,native_sampling_changed=bool(getattr(a,'speech_completion',False)),language_decoding_parameters_changed=False,audio_budget_changed=bool(getattr(a,'speech_completion',False)),speech_completion=bool(getattr(a,'speech_completion',False)),engine_profile=profile,all_opt_in_tails_complete=(all(r['acoustic_pending_at_end'] is False for r in rows) if getattr(a,'speech_completion',False) else None),adopted=False,qualified=False,diagnostic_not_full_behavioral_approval=True,profiler_overhead_precludes_latency_benchmark=True)
    assert report['complete']
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--minimum-start-seconds',type=int,choices=[600,900],default=900)
    p.add_argument('--speech-completion',action='store_true')
    for k in ('model','checkpoint','selection','timelines','voice-reference','voice-reference-sha256','lease','source','output'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
