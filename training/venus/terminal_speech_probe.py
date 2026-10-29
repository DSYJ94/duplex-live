"""Fixed TRAIN/DEV first-utterance rendering A/B; ZERO updates or adoption."""
import argparse,base64,gc,hashlib,json,time,wave
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from transformers import set_seed
    from portable.venus.native_app import NativeEngine,ensure_guard
    from portable.venus.terminal_speech import install
    from .voice_compare import recognizer,REVISION
    from .spoken_audit import references
    from data.venus.pacing import words
    from data.venus.repair import wer
    ensure_guard();stop=json.loads(Path(a.lease).read_text())['stop_unix'];assert stop-time.time()>600
    config=json.loads(Path(a.selection).read_text());timeline_root=Path(a.timelines);manifest=json.loads((timeline_root/'manifest.json').read_text());approval=json.loads(Path(a.approval).read_text())
    assert approval['approved'] and approval['manifest_sha256']==sha(timeline_root/'manifest.json')
    assert manifest['source_sha256']==sha(a.plan) and sha(a.checkpoint)==config['checkpoint_sha256']
    selected=[]
    for ident in config['scene_ids']:
        row=next(x for x in manifest['scenes'] if x['id']==ident)
        assert row['fold'] in ('TRAIN','DEV') and row['family']=='ordinary'
        assert sha(timeline_root/ident/'user.wav')==row['user_wav_sha256'];selected.append(row)
    assert len(selected)==2 and {x['fold'] for x in selected}=={'TRAIN','DEV'}
    out=Path(a.output);out.mkdir(exist_ok=False);rows=[]
    engine=NativeEngine(a.model,adapter_checkpoint=a.checkpoint,voice_reference=a.voice_reference,voice_reference_sha256=a.voice_reference_sha256,tts_attention='released')
    original=engine.model.tts.generate_chunk
    for maximum in (26,128):
        for scene in selected:
            ensure_guard();assert stop-time.time()>360
            engine.model.tts.generate_chunk=original
            observed=[]
            if maximum==128:observed=install(engine.duplex,128)
            else:
                def capture(*args,**kwargs):
                    tokens,cache=original(*args,**kwargs)
                    observed.append(dict(terminal=bool(engine.duplex.current_turn_ended),maximum=kwargs['max_new_token'],returned_tokens=int(tokens.shape[1])))
                    return tokens,cache
                engine.model.tts.generate_chunk=capture
            set_seed(42);engine.prepare();folder=out/('budget'+str(maximum))/scene['id'];folder.mkdir(parents=True)
            with wave.open(str(timeline_root/scene['id']/'user.wav')) as w:pcm=w.readframes(w.getnframes())
            traces=[];audio=[];terminal=False;began=time.monotonic()
            for index in range(scene['frames']):
                ensure_guard();result=engine.step(pcm[index*32000:(index+1)*32000]);encoded=result.pop('audio')
                samples=np.zeros(result['audio_samples'],dtype=np.float32) if result.get('wire_silence_omitted') else np.frombuffer(base64.b64decode(encoded),dtype='<f4')
                assert np.isfinite(samples).all() and len(samples)==result['audio_samples']
                sf.write(folder/f'chunk-{index:03d}.wav',samples,24000,subtype='FLOAT');audio.append(samples);traces.append(dict(result,sequence=index))
                if result['end_of_turn'] and not result['is_listen']:terminal=True;break
            (folder/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in traces))
            joined=folder/'emitted.wav';sf.write(joined,np.concatenate(audio),24000,subtype='FLOAT')
            row=dict(id=scene['id'],fold=scene['fold'],maximum=maximum,terminal_observed=terminal,frames=len(traces),source_wav_sha256=scene['user_wav_sha256'],references=references(traces),tts_calls=observed,wall_seconds=time.monotonic()-began,joined_wav=str(joined),joined_sha256=sha(joined),raw_peak=float(np.abs(np.concatenate(audio)).max()),qualified=False)
            rows.append(row);(folder/'generation.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(event='terminal_render_case',id=row['id'],maximum=maximum,terminal=terminal,tts_calls=observed)),flush=True)
    engine.model.tts.generate_chunk=original
    # Bound methods/closures retain the model unless explicitly released.
    del engine,original,capture;gc.collect();torch.cuda.empty_cache();pipe=recognizer()
    for row in rows:
        ensure_guard();assert stop-time.time()>180
        audio,sr=sf.read(row['joined_wav'],dtype='float32');assert sr==24000
        spoken=pipe(dict(array=resample_poly(audio,2,3).astype(np.float32),sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if np.any(audio) else ''
        row.update(asr_text=spoken,word_error_rate=wer(' '.join(words(row['references']['intended_user_facing'])),' '.join(words(spoken))),asr_revision=REVISION)
    pairs=[]
    for scene in selected:
        before=next(x for x in rows if x['id']==scene['id'] and x['maximum']==26);after=next(x for x in rows if x['id']==scene['id'] and x['maximum']==128)
        pairs.append(dict(id=scene['id'],same_native_generated_text=before['references']==after['references'],both_terminal=before['terminal_observed'] and after['terminal_observed'],wer26=before['word_error_rate'],wer128=after['word_error_rate'],peak26=before['raw_peak'],peak128=after['raw_peak'],adopted=False))
    report=dict(complete=len(rows)==4,source=a.source,checkpoint_sha256=config['checkpoint_sha256'],optimizer_updates=0,new_external_api_calls=0,selected_before_outputs=True,selection=config,rows=rows,pairs=pairs,qualified=False,adopted=False,not_a_full_behavior_or_realtime_benchmark=True)
    (out/'probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(event='terminal_render_complete',pairs=pairs)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('model','checkpoint','selection','plan','timelines','approval','voice-reference','voice-reference-sha256','lease','source','output'):p.add_argument('--'+key,required=True)
    main(p.parse_args())
