"""Actual concurrent ASR/controller/streaming-TTS development replay.

Input is paced approved TRAIN user PCM, never reference text. The external
controller chooses tools and words. One process owns both frozen GPU services;
real input frames have priority over bounded single TTS steps. No mic, room
AEC, native-model reasoning or fresh holdout qualification is claimed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time
from evals.moshi_model_download import sha256
from evals.kyutai_stt_probe import StreamingListener,selected_inputs
from evals.duplex_endpoint import TurnEndpoint
from evals.duplex_speaker import StreamingSpeaker
from evals.recovery_worker import RecordWorld,run_worker
from evals.utterance_playout import render_utterances


class PlaybackClock:
    """Same causal packet clock used by render_utterances, tracked online."""
    def __init__(self,rate=24000,buffer_s=.16):
        self.rate=rate;self.buffer=math.ceil(buffer_s*rate);self.end=0;self.started=False

    def append(self,available_s,samples):
        start=max(self.end,math.ceil(available_s*self.rate)+(0 if self.started else self.buffer))
        self.end=start+samples;self.started=True
        return start/self.rate,self.end/self.rate

    def stop(self,at):self.end=min(self.end,math.ceil(at*self.rate))


def counterfactual_records(scenario):
    # Declared read-only episode world, not a transcript or answer supplied to ASR.
    if scenario=='recovery-train-019':return {'Elm North':{'locker label':'copper kite'}}
    if scenario=='recovery-train-020':return {'Birch South':{'badge phrase':'violet crescent'}}
    if scenario=='recovery-train-008':return {}
    raise ValueError('Unapproved bounded development case')


def run_case(listener,speaker,row,out,delay_s):
    import numpy as np
    import soundfile as sf
    from data.synthetic.filter import norm_words,wer
    source,rate=sf.read(row['path'],dtype='float32',always_2d=True)
    if rate!=24000 or source.shape[1]!=2 or not np.isfinite(source).all():raise RuntimeError('Invalid TRAIN PCM')
    user=source[:,1].copy();out.mkdir();sf.write(out/'input-user.wav',user,rate,subtype='FLOAT')
    world=RecordWorld(counterfactual_records(row['scenario_id']));context=[];policy=TurnEndpoint(listener.spm.decode)
    listener.reset();speaker.stop();queue=[];futures={};held=[];packets=[];stops={};events=[];steps=[];backlogs=[]
    active=None;clock=None;spoken=[];asr_tokens=[];asr_events=[];utterance_count=0;generation_count=0
    # Tail allows an actual delayed result, a resumed answer and its PCM to finish.
    real_frames=math.ceil(len(user)/1920);max_frames=real_frames+math.ceil(45/.08)
    padded=np.pad(user,(0,max_frames*1920-len(user)));frame=0;start=time.perf_counter()
    executor=ThreadPoolExecutor(max_workers=4)
    def now():return time.perf_counter()-start
    def log(kind,**kw):events.append(dict(kind=kind,at_s=now(),**kw))
    def valid(item):
        if item['result']['route']=='delegate':
            return all(world.deliverable(r) for r in item['result']['receipts'] if r['tool']=='lookup_record')
        return not (item['result']['route']=='clarify' and item['turn']<utterance_count)
    def interrupt():
        nonlocal active,clock
        if active is None:return
        at=now();identity=active['event_id'];stops[identity]=at;clock.stop(at);speaker.stop()
        log('speech_interrupted',event_id=identity,task_id=active['task_id'],actual_stop_s=at)
        # Conservative baseline: retain ordinary answers in full. This can repeat
        # a heard prefix; no word-level heard-position inference is claimed.
        if valid(active) and active['result']['route']!='clarify':
            queue.append({k:v for k,v in active.items() if k!='event_id'})
        active=None;clock=None
    def dispatch(text):
        nonlocal utterance_count
        utterance_count+=1;task='turn-'+str(utterance_count);context.append(dict(role='user',text=text))
        revision=world.reserve(task,[e.copy() for e in context]);submitted=now()
        future=executor.submit(run_worker,[e.copy() for e in context],world,task,expected_revision=revision)
        futures[future]=dict(task_id=task,revision=revision,turn=utterance_count,submitted_s=submitted)
        log('controller_dispatched',task_id=task,text=text,reference_text_supplied=False)
    try:
        while frame<max_frames:
            t=now();next_frame=(frame+1)*.08
            if t>=next_frame:
                pcm=padded[frame*1920:(frame+1)*1920];before=time.perf_counter()
                backlogs.append(max(0.,before-start-next_frame));result=listener.step(pcm);compute=time.perf_counter()-before
                steps.append(compute);token=result['token']
                if token is not None and token>=4:asr_tokens.append(token)
                score=result['heads'][2][0] if result['heads'] is not None else None
                boundary=policy.update(token,score,float(np.sqrt(np.mean(pcm**2))),next_frame)
                asr_events.append(dict(frame=frame,input_available_s=next_frame,available_wall_s=now(),
                    token=token,heads=result['heads'],compute_s=compute,arrival_backlog_s=backlogs[-1],
                    real_samples_seen=min((frame+1)*1920,len(user)),synthetic_tail=frame>=real_frames))
                if boundary['speech_started']:
                    log('user_speech_started',input_available_s=next_frame);interrupt()
                if boundary['utterance']:
                    log('user_utterance_complete',**boundary['utterance']);dispatch(boundary['utterance']['text'])
                frame+=1
                continue  # Any arrived input always precedes further GPU speech work.
            for future,meta in list(futures.items()):
                if not future.done():continue
                del futures[future]
                result=future.result();item=dict(meta,result=result)
                item['ready_s']=now();item['release_s']=max(now(),meta['submitted_s']+delay_s) if result['route']=='delegate' else now()
                (out/(meta['task_id']+'-backend.json')).write_text(json.dumps(item,indent=2)+'\n')
                log('controller_completed',task_id=meta['task_id'],route=result['route'],release_s=item['release_s'])
                if result['route']!='delegate':world.finish(meta['task_id'],meta['revision'])
                held.append(item)
            for item in held[:]:
                if now()<item['release_s']:continue
                held.remove(item)
                if not valid(item):log('stale_result_discarded',task_id=item['task_id']);continue
                if item['result']['route']=='delegate':
                    for receipt in item['result']['receipts']:
                        if receipt['tool']=='lookup_record':world.mark_ready(receipt)
                queue.append(item)
            # Acoustic quiet alone is insufficient: await delayed ASR words and
            # the newest semantic dispatch before releasing any queued result.
            barrier=policy.speech_active or bool(policy.tokens) or bool(futures)
            if active and not speaker.active and now()>=clock.end/rate:
                log('speech_fully_played',event_id=active['event_id'],task_id=active['task_id'])
                spoken.append(dict(event_id=active['event_id'],text=active['result']['content'],task_id=active['task_id']))
                context.append(dict(role='assistant',text=active['result']['content']))
                if active['result']['route']=='delegate' and valid(active):world.finish(active['task_id'],active['revision'])
                active=None;clock=None
            queue[:]=[item for item in queue if valid(item)]
            if active is None and queue and not barrier:
                # Address the newest ordinary/cancel reply before older private
                # work, then resume interrupted work. No semantic native routing.
                queue.sort(key=lambda item:(item['result']['route']=='delegate',-item['turn']))
                active=queue.pop(0);generation_count+=1
                active['event_id']='speech-'+str(generation_count);clock=PlaybackClock()
                speaker.start(active['result']['content'])
                log('speech_started',event_id=active['event_id'],task_id=active['task_id'],text=active['result']['content'])
            # Generate only bounded PCM lead. A step may overrun; never disguise
            # that as a fast clock: the next ASR backlog records the actual delay.
            if (active and speaker.active and clock.end/rate-now()<.24
                    and next_frame-now()>.037):
                rendered=speaker.step();at=now()
                for pcm in rendered['pcm']:
                    packets.append(dict(event_id=active['event_id'],available_s=at,pcm=pcm))
                    clock.append(at,len(pcm))
                log('tts_step',event_id=active['event_id'],compute_s=rendered['compute_s'],packets=len(rendered['pcm']))
                continue
            if frame>=real_frames+7 and not futures and not held and not queue and active is None and not policy.tokens:
                break
            time.sleep(min(.002,max(0.,start+next_frame-time.perf_counter())))
        if frame>=max_frames:raise RuntimeError('Concurrent replay exceeded explicit 45s tail; unresolved work retained')
    finally:
        executor.shutdown(wait=True,cancel_futures=True);speaker.stop()
        (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
        (out/'asr-events.json').write_text(json.dumps(asr_events,indent=2)+'\n')
        if packets:
            np.savez(out/'generated-packets.npz',**{str(i):p['pcm'] for i,p in enumerate(packets)})
            (out/'packets.json').write_text(json.dumps([{k:v for k,v in p.items() if k!='pcm'}|dict(samples=len(p['pcm'])) for p in packets],indent=2)+'\n')
    if not packets:raise RuntimeError('No actual concurrent speech PCM was produced')
    audio,playout=render_utterances(packets,rate,.16,stops)
    sf.write(out/'model.wav',audio,rate,subtype='FLOAT')
    # Quality-only streams keep generated packets including cancelled material;
    # they are never counted as heard or used for task success.
    for identity in dict.fromkeys(p['event_id'] for p in packets):
        sf.write(out/(identity+'-generated.wav'),np.concatenate([p['pcm'] for p in packets if p['event_id']==identity]),rate,subtype='FLOAT')
    ledger=[{k:v for k,v in p.items() if k!='pcm'}|dict(samples=len(p['pcm'])) for p in packets]
    (out/'packets.json').write_text(json.dumps(ledger,indent=2)+'\n')
    side=json.loads(Path(row['path']).with_suffix('.json').read_text());reference=' '.join(w['text'] for w in side['user_words'])
    recognized=listener.spm.decode(asr_tokens)
    report=dict(scenario=row['scenario_id'],source_split='train',source_wav_sha256=row['wav_sha256'],
        user_pcm_sha256=hashlib.sha256(user.tobytes()).hexdigest(),recognized_text=recognized,
        reference_user_text=reference,recognition_wer=wer(norm_words(reference),norm_words(recognized)),
        reference_read_after_generation=True,reference_transcript_supplied=False,assistant_audio_supplied=False,
        user_samples=len(user),real_user_samples_processed=min(frame*1920,len(user)),all_live_user_rows_exact_at_temporal_forward=True,
        frames=frame,synthetic_tail_frames=max(0,frame-real_frames),asr_compute_p95_s=float(np.percentile(steps,95)),
        maximum_input_backlog_s=max(backlogs),declared_counterfactual_world=world.records,
        controlled_lookup_delivery_minimum_s=delay_s,world_tasks=world.tasks,actual_tool_receipts=world.receipts,
        fully_played_responses=spoken,playout=playout,audio_sha256=sha256(out/'model.wav'),wall_s=now(),
        unended_text=listener.spm.decode(policy.tokens),native_model_qualified=False,hybrid_system_qualified=False,
        microphone_acceptance=False,acoustic_echo_cancellation=False,read_only_tools=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('stt-assets','tts-assets','mimi-load-approval','data','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--scenario',action='append',required=True);ap.add_argument('--lookup-delay-s',type=float,default=12.)
    args=ap.parse_args()
    if not 0<=args.lookup_delay_s<=30:raise ValueError('Invalid bounded controlled delivery delay')
    rows=selected_inputs(args.data,args.scenario);args.out.mkdir(parents=True,exist_ok=False)
    listener=StreamingListener(args.stt_assets,args.mimi_load_approval)
    torch=listener.torch
    with listener.session():
        torch.manual_seed(42424242);torch.cuda.manual_seed_all(42424242)
        speaker=StreamingSpeaker(args.tts_assets,args.mimi_load_approval);speaker.open()
        warmup=speaker.warmup()
        listener.reset()
        import numpy as np
        for _ in range(20):listener.step(np.zeros(1920,dtype=np.float32),verify_input=False)
        results=[]
        try:
            for row in rows:results.append(run_case(listener,speaker,row,args.out/row['scenario_id'],args.lookup_delay_s))
        finally:speaker.close()
    report=dict(schema='dl-concurrent-hybrid-development-v1',component_execution_completed=True,results=results,
        stt_assets_sha256=sha256(args.stt_assets),tts_assets_sha256=sha256(args.tts_assets),
        listener_loads=listener.loads,speaker_evidence=speaker.evidence,tts_warmup=warmup,
        same_process_same_gpu=True,external_controller='gpt-6-luna',native_model_reasoning=False,
        paced_recorded_train_user_pcm=True,fresh_holdout=False,microphone_acceptance=False,
        native_model_qualified=False,hybrid_system_qualified=False,training_approved=False,optimizer_steps=0)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
