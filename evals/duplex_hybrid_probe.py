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
from evals.recovery_worker import RecordWorld,run_worker,WorkerContractError,lookup_receipts
from evals.utterance_playout import render_utterances
from evals.duplex_dispatch import dispatch_status


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


def run_case(listener,speaker,row,out,delay_s,controller_protocol="legacy",interrupt_policy="acoustic",controller=None,scope_read_tools=False,include_tool_context=False,worker_kind="records",live_input=None,output_observer=None):
    import numpy as np
    import soundfile as sf
    from data.synthetic.filter import norm_words,wer
    live=live_input is not None
    if live:
        if row['split']!='live-transport' or worker_kind!='reasoning-development':raise ValueError('Live prototype requires its explicit reasoning domain')
        rate=24000;user=np.zeros(0,dtype=np.float32)
    else:
        source,rate=sf.read(row['path'],dtype='float32',always_2d=True)
        if rate!=24000 or not np.isfinite(source).all():raise RuntimeError('Invalid source PCM')
        if row['split']=='fresh-evaluation':
            if source.shape[1]!=1:raise RuntimeError('Fresh evaluation expects user-only mono PCM')
            user=source[:,0].copy()
        else:
            if source.shape[1]!=2:raise RuntimeError('Approved TRAIN input expects stereo PCM')
            user=source[:,1].copy()
    out.mkdir()
    if not live:sf.write(out/'input-user.wav',user,rate,subtype='FLOAT')
    from queue import SimpleQueue,Empty
    control_queue=SimpleQueue();control_evidence=[]
    worker=run_worker;world_type=RecordWorld;delivered_receipts=lookup_receipts
    if worker_kind=='reasoning-development':
        from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker,reasoning_receipts
        worker=run_reasoning_worker;world_type=ReasoningWorld;delivered_receipts=reasoning_receipts
    world=world_type(row['world'] if row['split'] in ('fresh-evaluation','live-transport') else counterfactual_records(row['scenario_id']));context=[];policy=TurnEndpoint(listener.spm.decode)
    listener.reset();speaker.stop();queue=[];futures={};held=[];packets=[];stops={};events=[];steps=[];backlogs=[]
    active=None;clock=None;spoken=[];asr_tokens=[];asr_events=[];utterance_count=0;generation_count=0;last_voice_s=-float('inf')
    # Tail allows an actual delayed result, a resumed answer and its PCM to finish.
    real_frames=math.ceil(len(user)/1920);max_frames=(math.ceil(live_input.max_seconds/.08) if live else real_frames)+math.ceil(45/.08)
    padded=None if live else np.pad(user,(0,max_frames*1920-len(user)));frame=0;start=time.perf_counter()
    monotonic_origin=time.monotonic()-(time.perf_counter()-start);drain_next_s=0.
    executor=ThreadPoolExecutor(max_workers=4)
    def now():return time.perf_counter()-start
    def log(kind,**kw):
        event=dict(kind=kind,at_s=now(),**kw);events.append(event)
        if output_observer is not None and kind!='tts_step':output_observer.event(event)
    def valid(item):
        if item['result']['route']=='delegate':
            reads=delivered_receipts(item['result']['receipts'])
            return bool(reads) and all(world.deliverable(r) for r in reads)
        return not (item['result']['route']=='clarify' and item['turn']<utterance_count)
    def interrupt(reason="acoustic_speech_start"):
        nonlocal active,clock
        if active is None:return
        at=now();identity=active['event_id'];stops[identity]=at;clock.stop(at);speaker.stop()
        log('speech_interrupted',event_id=identity,task_id=active['task_id'],actual_stop_s=at,reason=reason)
        # Conservative baseline: retain ordinary answers in full. This can repeat
        # a heard prefix; no word-level heard-position inference is claimed.
        if valid(active) and active['result']['route']!='clarify':
            queue.append({k:v for k,v in active.items() if k!='event_id'})
        active=None;clock=None
    def drain_control():
        if not include_tool_context:return
        from evals.duplex_tool_context import project_execution
        while True:
            try:entry=control_queue.get_nowait()
            except Empty:break
            projection=project_execution(entry['call'],entry['receipt']);context.extend(projection)
            control_evidence.append(dict(available_s=now(),**entry))
            log('tool_execution_context_added',task_id=entry['task_id'],call_id=entry['call']['call_id'],
                operation=entry['call']['name'],private_values_withheld=True)
    def dispatch(text):
        nonlocal utterance_count
        if controller_protocol=='state-aware' and active is not None:interrupt('completed_user_request')
        drain_control()
        utterance_count+=1;task='turn-'+str(utterance_count);context.append(dict(role='user',text=text))
        revision=world.reserve(task,[e.copy() for e in context]);submitted=now()
        from evals.duplex_speech_obligations import snapshot
        pending=snapshot(queue,active) if controller_protocol=='state-aware' else []
        future=executor.submit(worker,[e.copy() for e in context],world,task,expected_revision=revision,dispatch_protocol=controller_protocol,pending_speech=pending,
            model=controller.model_id if controller else 'gpt-6-luna',response_executor=controller,scope_read_tools=scope_read_tools,execution_observer=control_queue.put if include_tool_context else None)
        futures[future]=dict(task_id=task,revision=revision,turn=utterance_count,submitted_s=submitted,request_text=text,pending_speech=pending)
        log('controller_dispatched',task_id=task,text=text,reference_text_supplied=False)
    try:
        if live and output_observer is not None:output_observer.event(dict(kind='session_ready',at_s=now()))
        while frame<max_frames:
            if live and now()>live_input.max_seconds+45:raise RuntimeError('Live session wall-clock limit reached')
            drain_control()
            t=now();next_frame=(frame+1)*.08 if not live else (drain_next_s if live_input.finished else t+.08)
            incoming=live_input.take() if live else None
            if live:
                if incoming is not None:
                    pcm,received=incoming;frame_available=received-monotonic_origin;synthetic_frame=False
                elif live_input.finished and t>=next_frame:
                    pcm=np.zeros(1920,dtype=np.float32);frame_available=t;synthetic_frame=True;live_input.drain_frames+=1
                else:pcm=None
            else:
                pcm=padded[frame*1920:(frame+1)*1920];frame_available=next_frame;synthetic_frame=frame>=real_frames
            if (live and pcm is not None) or (not live and t>=next_frame):
                before=time.perf_counter()
                backlogs.append(max(0.,before-start-frame_available));result=listener.step(pcm);compute=time.perf_counter()-before
                steps.append(compute);token=result['token']
                if token is not None and token>=4:asr_tokens.append(token)
                score=result['heads'][2][0] if result['heads'] is not None else None
                rms=float(np.sqrt(np.mean(pcm**2)))
                if rms>policy.energy_floor:last_voice_s=frame_available
                boundary=policy.update(token,score,rms,frame_available)
                asr_events.append(dict(frame=frame,input_available_s=frame_available,available_wall_s=now(),
                    token=token,heads=result['heads'],compute_s=compute,arrival_backlog_s=backlogs[-1],
                    real_samples_seen=live_input.sequence*1920 if live else min((frame+1)*1920,len(user)),synthetic_tail=synthetic_frame))
                if boundary['speech_started']:
                    log('user_speech_started',input_available_s=frame_available)
                    if interrupt_policy=='acoustic':interrupt()
                if (interrupt_policy=='lexical' and active is not None and token is not None and token>=4
                        and frame_available-last_voice_s<=1.
                        and any(ch.isalnum() for ch in listener.spm.decode([token]))):
                    interrupt('first_arrived_user_lexical_token')
                if boundary['utterance']:
                    log('user_utterance_complete',**boundary['utterance']);dispatch(boundary['utterance']['text'])
                frame+=1
                if live:drain_next_s=now()+.08
                continue  # Any arrived input always precedes further GPU speech work.
            for future,meta in list(futures.items()):
                if not future.done():continue
                del futures[future]
                try:result=future.result()
                except Exception as error:
                    failure=dict(schema='dl-hybrid-controller-failure-v1',task=meta,
                        error_type=type(error).__name__,message=str(error),
                        controller_evidence=error.result if isinstance(error,WorkerContractError) else None,
                        actual_world_receipts=world.receipts,actual_world_tasks=world.tasks,
                        controller_protocol=controller_protocol,hybrid_system_qualified=False)
                    (out/(meta['task_id']+'-failure.json')).write_text(json.dumps(failure,indent=2)+'\n')
                    log('controller_failed',task_id=meta['task_id'],error_type=type(error).__name__)
                    raise
                if controller_protocol=='state-aware':
                    from evals.duplex_speech_obligations import consume
                    consumed=result['consume_pending_speech_ids']
                    queue[:]=consume(queue,consumed,meta['pending_speech'])
                    for identity in consumed:log('speech_obligation_consumed',task_id=identity,fulfilled_by=meta['task_id'])
                item=dict(meta,result=result)
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
                    for receipt in delivered_receipts(item['result']['receipts']):world.mark_ready(receipt)
                queue.append(item)
            # Acoustic quiet alone is insufficient: await delayed ASR words and
            # the newest semantic dispatch before releasing any queued result.
            dispatch_state=dispatch_status(world,futures.values())
            barrier=policy.speech_active or bool(policy.tokens) or bool(dispatch_state['unresolved_decisions'])
            if interrupt_policy=='lexical':barrier=barrier or now()-last_voice_s<.66
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
                log('speech_started',event_id=active['event_id'],task_id=active['task_id'],text=active['result']['content'],active_background_task_ids=dispatch_state['actual_background_tasks'])
            # Generate only bounded PCM lead. A step may overrun; never disguise
            # that as a fast clock: the next ASR backlog records the actual delay.
            if (active and speaker.active and clock.end/rate-now()<.24
                    and (next_frame-now()>.037 or (live and incoming is None))):
                rendered=speaker.step();at=now()
                for pcm in rendered['pcm']:
                    packets.append(dict(event_id=active['event_id'],available_s=at,pcm=pcm))
                    clock.append(at,len(pcm))
                    if output_observer is not None:output_observer.pcm(active['event_id'],pcm,at)
                log('tts_step',event_id=active['event_id'],compute_s=rendered['compute_s'],packets=len(rendered['pcm']))
                continue
            if ((live_input.finished and live_input.drain_frames>=7) if live else frame>=real_frames+7) and not futures and not held and not queue and active is None and not policy.tokens:
                break
            time.sleep(min(.002,max(0.,start+next_frame-time.perf_counter())))
        if frame>=max_frames:raise RuntimeError('Concurrent replay exceeded explicit 45s tail; unresolved work retained')
    finally:
        executor.shutdown(wait=True,cancel_futures=True);speaker.stop()
        if live:
            user=live_input.pcm();real_frames=math.ceil(len(user)/1920)
            sf.write(out/'input-user.wav',user,rate,subtype='FLOAT')
            (out/'capture-arrivals.json').write_text(json.dumps(live_input.receipts,indent=2)+'\n')
        (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
        if include_tool_context:(out/'control-executions.json').write_text(json.dumps(control_evidence,indent=2)+'\n')
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
    if live:reference=None
    else:
        side=json.loads((Path(row['reference_path']) if row['split']=='fresh-evaluation' else Path(row['path']).with_suffix('.json')).read_text());reference=' '.join(w['text'] for w in side['user_words'])
    recognized=listener.spm.decode(asr_tokens)
    report=dict(scenario=row['scenario_id'],source_split=row['split'],source_wav_sha256=sha256(out/'input-user.wav') if live else row['wav_sha256'],
        user_pcm_sha256=hashlib.sha256(user.tobytes()).hexdigest(),recognized_text=recognized,
        reference_user_text=reference,recognition_wer=None if live else wer(norm_words(reference),norm_words(recognized)),
        reference_read_after_generation=not live,reference_transcript_supplied=False,live_transport=live,case_monotonic_origin=monotonic_origin,assistant_audio_supplied=False,
        user_samples=len(user),real_user_samples_processed=min(frame*1920,len(user)),all_live_user_rows_exact_at_temporal_forward=True,
        frames=frame,synthetic_tail_frames=max(0,frame-real_frames),asr_compute_p95_s=float(np.percentile(steps,95)),
        maximum_input_backlog_s=max(backlogs),declared_counterfactual_world=world.records,
        controlled_lookup_delivery_minimum_s=delay_s,controller_protocol=controller_protocol,interrupt_policy=interrupt_policy,
        foreground_barrier='actual-execution-aware',worker_kind=worker_kind,actual_tool_context_enabled=include_tool_context,control_executions_sha256=sha256(out/'control-executions.json') if include_tool_context else None,
        read_tools_scoped_to_user_identified_available_projects=scope_read_tools,
        controller_provider=controller.provider_name if controller else 'openai',controller_model=controller.model_id if controller else 'gpt-6-luna',
        lexical_interruption_is_not_a_backchannel_classifier=interrupt_policy=='lexical',world_tasks=world.tasks,actual_tool_receipts=world.receipts,
        fully_played_responses=spoken,playout=playout,audio_sha256=sha256(out/'model.wav'),wall_s=now(),
        unended_text=listener.spm.decode(policy.tokens),native_model_qualified=False,hybrid_system_qualified=False,
        microphone_acceptance=False,acoustic_echo_cancellation=False,read_only_tools=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('stt-assets','tts-assets','mimi-load-approval','data','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--scenario',action='append',required=True);ap.add_argument('--lookup-delay-s',type=float,default=12.)
    ap.add_argument('--worker-kind',choices=('records','reasoning-development'),default='records')
    ap.add_argument('--include-tool-context',action='store_true')
    ap.add_argument('--scope-read-tools',action='store_true')
    ap.add_argument('--controller-provider',choices=('openai','local-qwen','local-qwen35'),default='openai')
    ap.add_argument('--local-controller-assets',type=Path)
    ap.add_argument('--local-controller-kernel-assets',type=Path)
    ap.add_argument('--local-controller-spoken-output-contract',action='store_true')
    ap.add_argument('--local-controller-compact-replies',action='store_true')
    ap.add_argument('--local-controller-pending-speech-protocol',choices=('array','explicit-actions'),default='array')
    ap.add_argument('--local-controller-device',default='cuda:1')
    ap.add_argument('--local-controller-prompt-profile',choices=('base','contract-examples'),default='contract-examples')
    ap.add_argument('--interrupt-policy',choices=('acoustic','lexical'),default='acoustic')
    ap.add_argument('--input-profile',choices=('approved-train','fresh-evaluation'),default='approved-train')
    ap.add_argument('--input-quality',type=Path)
    ap.add_argument('--input-validator-source',type=Path)
    ap.add_argument('--input-validator-commit')
    ap.add_argument('--evaluation-use',choices=('development','fresh'),default='development')
    ap.add_argument('--controller-protocol',choices=('legacy','required-action','state-aware'),default='legacy')
    args=ap.parse_args()
    if args.worker_kind=='reasoning-development' and (args.controller_provider!='local-qwen35' or args.controller_protocol!='state-aware' or args.include_tool_context or args.scope_read_tools):raise ValueError('Reasoning trial requires isolated local state-aware control without record-specific options')
    if (args.local_controller_kernel_assets is not None or args.local_controller_spoken_output_contract or args.local_controller_compact_replies) and args.controller_provider!='local-qwen35':raise ValueError('Optional newer-controller settings require local-qwen35')
    if args.include_tool_context and args.controller_protocol!='state-aware':raise ValueError('First tool-context proof requires state-aware protocol')
    if not 0<=args.lookup_delay_s<=30:raise ValueError('Invalid bounded controlled delivery delay')
    if args.input_profile=='fresh-evaluation':
        from evals.duplex_input_validator import cases
        if args.input_quality is None:raise ValueError('Fresh evaluation requires a separate exact quality receipt')
        rows,validator=cases(args.data,args.input_quality,args.scenario,args.input_validator_source,args.input_validator_commit)
    else:rows=selected_inputs(args.data,args.scenario);validator=None
    args.out.mkdir(parents=True,exist_ok=False)
    listener=StreamingListener(args.stt_assets,args.mimi_load_approval)
    torch=listener.torch;controller=None
    if args.controller_provider in ('local-qwen','local-qwen35'):
        if args.controller_protocol not in ('required-action','state-aware') or args.local_controller_assets is None:
            raise ValueError('Local controller requires exact assets and explicit action protocol')
        if torch.device(args.local_controller_device).index!=1 or torch.cuda.device_count()<2:
            raise RuntimeError('First local integration requires a distinct second allocated GPU; no resource fallback')
        if args.controller_provider=='local-qwen35':
            if args.local_controller_prompt_profile!='contract-examples':raise ValueError('Resident newer controller currently proves only the contract-examples profile')
            from evals.qwen35_rpc import ResidentQwen35
            controller=ResidentQwen35(args.local_controller_assets,args.out,args.local_controller_device,args.local_controller_pending_speech_protocol,kernel_assets=args.local_controller_kernel_assets,spoken_output_contract=args.local_controller_spoken_output_contract,compact_replies=args.local_controller_compact_replies)
        else:
            from evals.qwen_response_executor import QwenResponseExecutor
            controller=QwenResponseExecutor(args.local_controller_assets,args.local_controller_device,args.local_controller_prompt_profile,pending_speech_protocol=args.local_controller_pending_speech_protocol)
    try:
        with listener.session():
            torch.manual_seed(42424242);torch.cuda.manual_seed_all(42424242)
            speaker=StreamingSpeaker(args.tts_assets,args.mimi_load_approval);speaker.open()
            warmup=speaker.warmup()
            listener.reset()
            import numpy as np
            for _ in range(20):listener.step(np.zeros(1920,dtype=np.float32),verify_input=False)
            results=[];failures=[]
            try:
                for row in rows:
                    try:results.append(run_case(listener,speaker,row,args.out/row['scenario_id'],args.lookup_delay_s,args.controller_protocol,args.interrupt_policy,controller,args.scope_read_tools,args.include_tool_context,args.worker_kind))
                    except Exception as error:
                        if args.input_profile!='fresh-evaluation':raise
                        failure=dict(scenario=row['scenario_id'],error_type=type(error).__name__,message=str(error),
                            case_execution_completed=False,hybrid_system_qualified=False)
                        failures.append(failure)
                        (args.out/row['scenario_id']/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            finally:speaker.close()
    finally:
        if controller is not None and hasattr(controller,'close'):controller.close()
    report=dict(schema='dl-concurrent-hybrid-development-v1',component_execution_completed=not failures,results=results,failures=failures,
        worker_kind=args.worker_kind,actual_tool_context_enabled=args.include_tool_context,
        read_tools_scoped_to_user_identified_available_projects=args.scope_read_tools,
        component_code_sha256={name:sha256(Path(__file__).with_name(name)) for name in
            ('recovery_worker.py','duplex_endpoint.py','duplex_speaker.py','kyutai_stt_probe.py','duplex_speech_obligations.py','duplex_tool_context.py','duplex_reasoning_worker.py','duplex_dispatch.py')},
        probe_code_sha256=sha256(Path(__file__)),
        input_profile=args.input_profile,input_validator=validator,evaluation_use=args.evaluation_use,input_quality_sha256=sha256(args.input_quality) if args.input_quality else None,
        stt_assets_sha256=sha256(args.stt_assets),tts_assets_sha256=sha256(args.tts_assets),
        listener_loads=listener.loads,speaker_evidence=speaker.evidence,tts_warmup=warmup,
        speech_services_same_process_same_gpu=True,same_process_same_gpu=controller is None,
        controller_provider=controller.provider_name if controller else 'openai',
        controller_model=controller.model_id if controller else 'gpt-6-luna',
        local_controller_proof=controller.proof if controller else None,
        local_controller_warmup=controller.warmup if controller else None,
        controller_transport=getattr(controller,'transport_info',dict(kind='in-process')) if controller else dict(kind='remote-api'),
        local_controller_pending_speech_protocol=args.local_controller_pending_speech_protocol if controller else None,
        external_controller=None if controller else 'gpt-6-luna',
        external_reasoner='gpt-6-astra' if args.worker_kind=='reasoning-development' else None,
        remote_api_used=controller is None or any(r.get('remote_api_used',False) for c in results for r in c['actual_tool_receipts']),native_model_reasoning=False,
        paced_recorded_train_user_pcm=args.input_profile=='approved-train',
        fresh_holdout=args.input_profile=='fresh-evaluation' and args.evaluation_use=='fresh',candidate_qualification_derived_from_execution=False,microphone_acceptance=False,
        native_model_qualified=False,hybrid_system_qualified=False,training_approved=False,optimizer_steps=0)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
