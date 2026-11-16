"""Frozen32 native generation and independent speech audit after full training.

No training, external API, backend result injection or adoption. All conditions
use the released Engine/persona/voice/sampler and preserve raw float PCM.
"""
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
from . import matched_guard as g
from .matched_job import write
from .matched_generation import completed_states,make_engine

CAPACITIES=('baseline','narrow','broad')


def complete_collection(value,rows,required=32):
    if type(required)is not int or required not in (32,64):raise ValueError('Only declared complete32/64 cohorts admitted')
    actual=value.get('cases',[])
    expected={r['id']:r['input_sha256'] for r in rows}
    if (value.get('required_cases')!=required or len(rows)!=required or len(expected)!=required
            or len(actual)!=required or {r['id']:r['input_sha256'] for r in actual}!=expected
            or any(r.get('complete') is not True for r in actual)):
        raise ValueError('All32 exact complete inputs required; no passing subset'if required==32 else'All declared exact complete inputs required; no passing subset')
    return actual


def speech_presence(generated,recognized,voiced):
    from data.venus.pacing import words
    return bool(words(generated))and bool(words(recognized))and bool(voiced)


def complete_frames(trace,derived_samples):
    if (derived_samples<=0 or derived_samples%1920 or len(trace)!=derived_samples//1920
            or [t['sequence'] for t in trace]!=list(range(len(trace)))):
        raise ValueError('Every contiguous native frame from the complete input required')


def interaction(case,trace,word_quality):
    from training.venus.behavior_review import cue_frame
    voiced=lambda t:t['audio_samples']>0 and t['peak']>.001
    result={}
    if case['family']=='quiet':
        _,end=cue_frame(case['user_turns'][-1],word_quality[case['user_turns'][-1]['sha256']])
        eligible=math.ceil(end/.08)-1
        result.update(cue_word_end_seconds=end,
            voiced_frames_before_cue_word=sum(voiced(t) for t in trace if t['sequence']<eligible),
            voiced_frames_after_cue_word=sum(voiced(t) for t in trace if t['sequence']>=eligible))
    if case['family']=='pause':
        turns=case['user_turns'];start=turns[1]['start_seconds'];end=start+turns[1]['seconds']
        resume=turns[-1]['start_seconds']+turns[-1]['seconds']
        result.update(voiced_before_pause=any(voiced(t) and t['sequence']*.08<start for t in trace),
            voiced_frames_while_paused=sum(voiced(t) for t in trace if end<=t['sequence']*.08<resume))
    return result


def collect(a,rows,selection,*,factory=None,persona=None,voice_codes=None,voice_sha256=None,profile_extra=None,guard=None):
    if len(rows)not in (32,64)or len({r['id']for r in rows})!=len(rows):raise ValueError('Complete unique32/64 cohort required before loading a model')
    ensure=g.check if guard is None else guard
    import numpy as np,soundfile as sf,torch
    from scipy.signal import resample_poly
    from contextspan.model.engine import seed_all
    from evals.context_spanning_download import SOURCE_COMMIT,VOICE_SHA256
    from .foundation_context import PERSONA,WEIGHTS,word_records
    root=Path(a.root)/(a.base+'-'+a.capacity)
    reused=[]
    if getattr(a,'reuse_record',None):
        from .matched_evaluation_reuse import prepared, pin
        from .matched_interrupted import DURABLE
        record_path=Path(a.reuse_record)
        record_path.resolve().relative_to(DURABLE.resolve())
        if not a.reuse_record_sha or g.sha(record_path)!=a.reuse_record_sha:
            raise ValueError('Exact pinned reuse record required')
        record=json.loads(record_path.read_text())
        Path(record['parent']).resolve().relative_to(DURABLE.resolve())
        profile=record['profile']
        if (profile['base']!=a.base or profile['capacity']!=a.capacity
                or profile['trainer_source']!=a.trainer_source
                or profile['selection']!=selection or profile['optimizer_updates']!=0
                or profile['seed']!=42 or profile['temp_audio']!=.8 or profile['temp_text']!=.7
                or profile['upstream']!=SOURCE_COMMIT or profile['voice_sha256']!=VOICE_SHA256
                or profile['persona']!=PERSONA):
            raise ValueError('Reuse requires identical model, sampler, voice and persona')
        if pin(record['parent'],rows,profile)!=record:
            raise ValueError('Pinned original cases changed')
        reused=prepared(record,root)
        (root/'profile.json').rename(root/'original-profile.json')
        (root/'results.json').rename(root/'original-results.json')
    else:
        root.mkdir(exist_ok=False)
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-B','-m',
        'evals.cuda_linker_path','--out',str(root/'driver')],text=True).strip()
    ensure(a.lease,600);seed_all(42);engine=(factory or make_engine)(a,selection)
    voice=(torch.load(WEIGHTS/'voices/main.pt',map_location='cpu',weights_only=True)['codes']if voice_codes is None else voice_codes).long()
    persona=PERSONA if persona is None else persona
    voice_sha256=VOICE_SHA256 if voice_sha256 is None else voice_sha256
    if voice.ndim!=2 or voice.shape[0]!=8:raise ValueError('Original released voice geometry required')
    write(root/'profile.json',dict(source=a.source,trainer_source=a.trainer_source,
        upstream=SOURCE_COMMIT,base=a.base,capacity=a.capacity,selection=selection,
        persona=persona,voice_sha256=voice_sha256,seed=42,temp_audio=.8,temp_text=.7,
        loads=engine.evaluation_model_loads,torch=torch.__version__,optimizer_updates=0,
        native_worker_interface=False,voice_accent_not_certified=True,live_microphone_latency=False,adopted=False,
        evaluation_context={} if profile_extra is None else profile_extra,
        actual_expanded_inference=getattr(engine,'expanded_inference_proof',None)))
    results=list(reused);quality=word_records()
    reused_ids={r['id'] for r in reused}
    if reused:
        write(root/'results.json',dict(source=a.source,cases=results,required_cases=len(rows),qualified=False,
            reused_producer=record['original_source'],reuse_record_sha256=a.reuse_record_sha))
    with torch.inference_mode():
        for case in rows:
            if case['id'] in reused_ids:continue
            ensure(a.lease,180);seed_all(42);engine.reset();engine.set_persona(persona,voice)
            folder=root/case['id'];folder.mkdir();path=Path(case['folder'])/'user.wav'
            audio,sr=sf.read(path,dtype='float32')
            if sr!=case.get('input_rate',16000) or sr not in (16000,24000) or audio.ndim!=1 or g.sha(path)!=case['input_sha256']:
                raise ValueError('Original complete source waveform required')
            ensure(a.lease,180+math.ceil(len(audio)/sr*3))
            user=(resample_poly(audio,3,2)if sr==16000 else audio).astype(np.float32);padding=(-len(user))%1920;user=np.pad(user,(0,padding))
            sf.write(folder/'derived-user-24k.wav',user,24000,subtype='FLOAT')
            trace=[];tokens=[];started=time.monotonic()
            # Every original output frame is saved, without gain/trim/normalization.
            with (folder/'trace.jsonl').open('x') as log:
                for i,offset in enumerate(range(0,len(user),1920)):
                    ensure(a.lease);before=time.monotonic();out=engine.step(user[offset:offset+1920]);elapsed=time.monotonic()-before
                    pcm=np.asarray(out['agent_pcm'] if out['agent_pcm'] is not None else [],dtype=np.float32)
                    if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Finite unmodified native PCM required')
                    name='chunk-%04d.wav'%i;sf.write(folder/name,pcm,24000,subtype='FLOAT');tokens.append(out['text_token'])
                    frame=dict(sequence=i,chunk_file=name,audio_samples=len(pcm),peak=float(np.abs(pcm).max()) if len(pcm) else 0.,
                        elapsed=elapsed,wall_since_start=time.monotonic()-started,text_token=out['text_token'],is_ret=out['is_ret'])
                    trace.append(frame);log.write(json.dumps(frame)+'\n');log.flush()
                    time.sleep(max(0,started+(i+1)*.08-time.monotonic()))
            result=dict(id=case['id'],family=case['family'],input_sha256=case['input_sha256'],input_frames=case['frames'],
                condition=a.base+'-'+a.capacity,complete=True,native_worker_interface=False,actual_worker_events=[],
                native_ret_tokens=sum(t['is_ret'] for t in trace),generated_text=engine.decode_text(tokens),
                wall_seconds=time.monotonic()-started,derived_tail_padding_samples=padding,qualified=False,
                **interaction(case,trace,quality))
            write(folder/'result.json',result);results.append(result)
            write(root/'results.json',dict(source=a.source,cases=results,required_cases=len(rows),qualified=False))
            print(json.dumps(dict(event='native_case_complete',condition=result['condition'],completed=len(results),required=len(rows))),flush=True)
    complete_collection(json.loads((root/'results.json').read_text()),rows,required=len(rows))


def audit(a,rows,*,required=32,guard=None,require_spoken_content=False,recognizer_factory=None):
    ensure=g.check if guard is None else guard
    import numpy as np,soundfile as sf
    from scipy.signal import resample_poly
    from training.venus.voice_compare import recognizer
    from training.venus.spoken_audit import REVISION
    from training.venus.foundation_native import timed_assembly
    from data.venus.pacing import words
    from data.venus.repair import wer
    root=Path(a.root)/(a.base+'-'+a.capacity)
    collected=complete_collection(json.loads((root/'results.json').read_text()),rows,required=required)
    profile=json.loads((root/'profile.json').read_text())
    if profile['source']!=a.source or profile['trainer_source']!=a.trainer_source:
        raise ValueError('Actual evaluation/training sources must stay separate and exact')
    ensure(a.lease,600);pipe=(recognizer_factory or recognizer)();audits=[]
    for case in collected:
        ensure(a.lease,90);folder=root/case['id'];trace=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
        arrays=[]
        for t in trace:
            pcm,sr=sf.read(folder/t['chunk_file'],dtype='float32')
            if sr!=24000 or pcm.ndim!=1 or len(pcm)!=t['audio_samples'] or not np.isfinite(pcm).all():
                raise ValueError('Every original raw chunk required')
            arrays.append(pcm)
        info=sf.info(folder/'derived-user-24k.wav')
        if info.samplerate!=24000 or info.channels!=1:raise ValueError('Original derived frame clock required')
        complete_frames(trace,info.frames)
        raw=np.concatenate(arrays);timed=timed_assembly(trace,arrays,24000,frame_seconds=.08)
        wav=folder/'joined-emitted-raw.wav';clockwav=folder/'timed-emitted-raw.wav'
        sf.write(wav,raw,24000,subtype='FLOAT');sf.write(clockwav,timed,24000,subtype='FLOAT')
        recognized=pipe(dict(array=resample_poly(raw,2,3).astype(np.float32),sampling_rate=16000),
            generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if len(raw) and np.any(raw) else ''
        error=wer(' '.join(words(case['generated_text'])),' '.join(words(recognized)))
        peak=float(np.abs(raw).max()) if len(raw) else 0.
        present=speech_presence(case['generated_text'],recognized,np.any(np.abs(raw)>.001))
        audits.append(dict(id=case['id'],family=case['family'],input_sha256=case['input_sha256'],
            generated_text=case['generated_text'],asr_text=recognized,word_error_rate=error,word_pass=error<=.05 and (present or not require_spoken_content),speech_content_present=present,
            raw_peak=peak,raw_peak_pass=peak<=.995,has_voiced_output=bool(np.any(np.abs(raw)>.001)),
            raw_wav_sha256=g.sha(wav),timed_raw_wav_sha256=g.sha(clockwav),timed_raw_peak=float(np.abs(timed).max()),
            source_chunks=[dict(file=t['chunk_file'],sha256=g.sha(folder/t['chunk_file'])) for t in trace],
            semantic_or_naturalness_pass_claimed=False))
        write(root/'audit-progress.json',dict(required=required,cases=audits))
    quiet=[r for r in collected if r['family']=='quiet'];pause=[r for r in collected if r['family']=='pause']
    summary=dict(cases=required,word_pass=sum(r['word_pass'] for r in audits),raw_peak_pass=sum(r['raw_peak_pass'] for r in audits),
        joint_pass=sum(r['word_pass'] and r['raw_peak_pass'] for r in audits),
        necessary_silence_and_resume=sum(r['voiced_frames_before_cue_word']==0 and r['voiced_frames_after_cue_word']>0 for r in quiet),
        silent_while_paused=sum(r['voiced_frames_while_paused']==0 for r in pause))
    report=dict(complete=len(audits)==required,source=a.source,trainer_source=a.trainer_source,summary=summary,cases=audits,
        word_limit=.05,peak_limit=.995,asr_revision=REVISION,external_API_calls=0,optimizer_updates=0,
        connected_backend_tested=False,semantics_and_naturalness_require_review=True,adopted=False)
    write(root/'report.json',report);print(json.dumps(dict(condition=a.base+'-'+a.capacity,summary=summary)),flush=True)


def main(a):
    actual=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    lease=g.check(a.lease,180)
    if actual!=a.source or lease['source']!=a.source:raise ValueError('Actual evaluation source and live guard required')
    selected=completed_states(a.training_root,a.trainer_source)
    selection=next((s for s in selected if s['base']==a.base and s['capacity']==a.capacity),None)
    if a.capacity!='baseline' and selection is None:raise ValueError('Full trained candidate missing')
    from .foundation_context import preflight
    rows=preflight(Path.cwd())
    if a.stage=='collect':collect(a,rows,selection)
    else:audit(a,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','trainer-source','training-root','lease'):p.add_argument('--'+name,required=True)
    p.add_argument('--base',choices=('personaplex','context-spanning'),required=True)
    p.add_argument('--capacity',choices=CAPACITIES,required=True)
    p.add_argument('--stage',choices=('collect','audit'),required=True)
    p.add_argument('--reuse-record');p.add_argument('--reuse-record-sha')
    main(p.parse_args())
