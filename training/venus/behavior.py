"""Complete36 held-out native collection: no forced decisions or teacher feedback.

This records actual speech and asynchronous worker events. Completion is not
behavioral approval; independent speech/semantic review and paired retention
remain required. No optimizer or external model API is used.
"""
import argparse,asyncio,base64,hashlib,json,math,time,wave
from pathlib import Path
from data.venus.receipts import LookupWorker
from portable.venus.native_app import NativeEngine,DelegateParser


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(plan_path,timelines,approval_path):
    root=Path(timelines);manifest=json.loads((root/'manifest.json').read_text())
    approval=json.loads(Path(approval_path).read_text());assert approval['approved'] and approval['manifest_sha256']==sha(root/'manifest.json')
    assert manifest['source_sha256']==sha(plan_path)
    plan=json.loads(Path(plan_path).read_text());selected={s['id']:s for s in plan['scenes'] if s['fold']=='HOLDOUT'}
    rows=[r for r in manifest['scenes'] if r['fold']=='HOLDOUT'];assert len(rows)==36 and set(selected)=={r['id'] for r in rows}
    cases=[]
    for row in rows:
        folder=root/row['id'];assert sha(folder/'timeline.json')==row['timeline_sha256'] and sha(folder/'user.wav')==row['user_wav_sha256']
        timeline=json.loads((folder/'timeline.json').read_text())
        with wave.open(str(folder/'user.wav')) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())==(16000,1,2,row['frames']*16000)
        # ONLY user timing/audio and the worker's world/delay are used. No teacher
        # actions, expected answers or pre-recorded backend feedback enter model inputs.
        scene=selected[row['id']]
        cases.append(dict(id=row['id'],family=row['family'],folder=str(folder),frames=row['frames'],
            user_turns=timeline['user_turns'],world=scene['synthetic_private_world'] or {},
            worker_delay=scene['teacher_program'].get('worker_delay_seconds',8),input_sha256=row['user_wav_sha256']))
    return cases


async def run_case(engine,case,out,stop_unix,pace=True):
    import numpy as np
    import soundfile as sf
    folder=out/case['id'];folder.mkdir(exist_ok=False)
    with wave.open(str(Path(case['folder'])/'user.wav')) as w:pcm=w.readframes(w.getnframes())
    worker=LookupWorker(case['world'],case['worker_delay']);parser=DelegateParser();events=[];audio_frames=[];last=None;seen_started=set();pending_at_followup=None
    await asyncio.to_thread(engine.prepare);origin=time.monotonic()
    try:
        with (folder/'trace.jsonl').open('x') as trace:
            for frame in range(case['frames']):
                if time.time()>=stop_unix-180:raise TimeoutError('Preserve backup/STOP reserve; no partial-set approval')
                if len(case['user_turns'])>1 and pending_at_followup is None and frame>=math.floor(case['user_turns'][1]['start_seconds']):
                    pending_at_followup=any(k in seen_started and not t.done() for k,t in worker.tasks.items())
                feedback=worker.take_results()
                if feedback:worker.event('backend_injected',payload=feedback,sequence=frame)
                raw=pcm[frame*32000:(frame+1)*32000];assert len(raw)==32000
                last=await asyncio.to_thread(engine.step,raw,feedback)
                queries=parser.feed(last['raw_text'])
                for query,original in zip(queries,parser.last_raw_queries):worker.submit(query,raw_query=original)
                await asyncio.sleep(0)  # Let actual workers begin/complete; no oracle dispatch.
                new_events=list(worker.events);worker.events.clear();events.extend(new_events)
                seen_started.update(e['work_id'] for e in new_events if e['type']=='worker_started')
                audio=np.frombuffer(base64.b64decode(last.pop('audio')),dtype='<f4');assert np.isfinite(audio).all()
                sf.write(folder/f'chunk-{frame:03d}.wav',audio,24000,subtype='FLOAT')
                audio_frames.append(dict(frame=frame,samples=len(audio),peak=last['peak']))
                trace.write(json.dumps(dict(last,sequence=frame,events=new_events,input_sha256=hashlib.sha256(raw).hexdigest(),wall_since_start=time.monotonic()-origin,chunk_file=f'chunk-{frame:03d}.wav'))+'\n');trace.flush()
                if pace:await asyncio.sleep(max(0,origin+frame+1-time.monotonic()))
    finally:
        await worker.close()
        (folder/'cleanup-events.json').write_text(json.dumps(dict(cleanup_only=True,events=worker.events),indent=2)+'\n')
    turns=case['user_turns'];quiet_release=math.ceil(turns[-1]['start_seconds']+turns[-1]['seconds'])
    before=[r for r in audio_frames if r['frame']<quiet_release]
    result=dict(id=case['id'],family=case['family'],complete=True,input_frames=case['frames'],input_sha256=case['input_sha256'],
        actual_worker_events=events,genuine_pending_at_followup=pending_at_followup,
        teacher_actions_used=False,oracle_feedback_used=False,qualified=False,independent_speech_semantic_review_required=True,
        wall_seconds=time.monotonic()-origin,native_turn_still_open=not engine.duplex.current_turn_ended,
        voiced_output_frames=sum(r['samples']>0 and r['peak']>.001 for r in audio_frames))
    if case['family']=='quiet':result.update(nonempty_audio_frames_before_cue_end=sum(r['samples']>0 for r in before),voiced_frames_before_cue_end=sum(r['samples']>0 and r['peak']>.001 for r in before),voiced_frames_after_cue_end=sum(r['samples']>0 and r['peak']>.001 for r in audio_frames if r['frame']>=quiet_release))
    if case['family']=='pause':
        start=turns[1]['start_seconds'];end=math.ceil(start+turns[1]['seconds']);resume=math.ceil(turns[-1]['start_seconds']+turns[-1]['seconds'])
        result.update(voiced_before_pause=any(r['samples']>0 and r['peak']>.001 and r['frame']<start for r in audio_frames),voiced_frames_while_paused=sum(r['samples']>0 and r['peak']>.001 for r in audio_frames if end<=r['frame']<resume))
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


async def main(a):
    import torch
    from transformers import set_seed
    cases=preflight(a.plan,a.timelines,a.approval);assert sum(c['frames'] for c in cases)==753
    assert sha(a.checkpoint)==a.checkpoint_sha256
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==state['metadata']['recipe']['updates']==190 and state['metadata']['approval_sha256']==sha(a.approval)
    stop=json.loads(Path(a.lease).read_text())['stop_unix'];required=753*1.1+90+72+180
    if stop-time.time()<=required:raise TimeoutError('Complete36 native collection does not fit; no subset or duplicate run')
    out=Path(a.output);out.mkdir(exist_ok=False)
    engine=await asyncio.to_thread(NativeEngine,a.model,'Streaming Omni Conversation.',a.checkpoint)
    (out/'profile.json').write_text(json.dumps(dict(engine.profile,collection_source=a.source,all36_holdout=True,paired_baseline_pending=True),indent=2)+'\n')
    results=[]
    try:
        for case in cases:
            set_seed(42)  # Fixed before collection, including matched synthetic worlds.
            result=await run_case(engine,case,out,stop);results.append(result)
            (out/'results.json').write_text(json.dumps(dict(cases=results,all36_complete=len(results)==36,qualified=False,paired_baseline_pending=True),indent=2)+'\n')
            print(json.dumps(dict(event='native_holdout_case_complete',id=case['id'],completed=len(results),total=36,qualified=False)),flush=True)
    finally:
        (out/'collection-terminal.json').write_text(json.dumps(dict(complete=len(results)==36,completed_cases=len(results),required_cases=36,optimizer_updates=0,teacher_forcing=False,oracle_feedback=False,qualified=False),indent=2)+'\n')
    assert len(results)==36


if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('model','checkpoint','checkpoint-sha256','plan','timelines','approval','output','source','lease')];asyncio.run(main(p.parse_args()))
