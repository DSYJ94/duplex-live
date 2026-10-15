"""Actual-worker result integration diagnostic; dispatch is explicitly scheduled.

The native voice receives completed text in its nonspoken instruction prefix.
It continues hearing every arrived user sample and samples all assistant output.
Backend context uses measured user words, not live ASR. This cannot qualify native
foreground delegation choice or a microphone product.
"""
import concurrent.futures
from contextlib import nullcontext
import hashlib
import json
import time
from pathlib import Path


def run(framework, args, loading, side_audit):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from evals.recovery_data import validate_conversion
    from evals.recovery_context import causal_dialogue
    from evals.recovery_worker import RecordWorld,run_worker
    from evals.lychee_context import create_session,refresh_session,role_text
    from evals.lychee_paced import run_incremental
    from evals.lychee_delivery import QuietDelivery
    delivery=QuietDelivery(args.result_delivery)
    data=validate_conversion(args.data)
    row=next(r for r in data['accepted'] if r['scenario_id']==args.scenario)
    required_split='train' if args.collect_training_state else 'validation'
    if row['split']!=required_split or row['kind']!='private':
        raise RuntimeError('Worker-context mode requires approved private '+required_split+' scene')
    sidecar=json.loads(Path(row['path']).with_suffix('.json').read_text());meta=sidecar['timeline']['meta']
    value=args.private_value or meta['phrase']
    if not value or len(value.split())!=2:raise ValueError('Development world requires one two-word value')
    world_data={meta['project']:{meta['field']:value}}
    world=RecordWorld(world_data);events=[];actual=[];holder={};future=None
    pcm,rate=sf.read(row['path'],dtype='float32',always_2d=True)
    if rate!=24000 or pcm.shape[1]!=2 or not np.isfinite(pcm).all():raise RuntimeError('Invalid approved stereo stimulus')
    audio=resample_poly(pcm[:,1],2,3).astype(np.float32)
    audio=np.pad(audio,(0,20*16000));sf.write(args.out/'user_in.wav',audio,16000)
    first=next(e for e in sidecar['timeline']['events'] if e['ch']=='user' and e['kind']=='utterance')
    trigger=first['actual_end_ms']/1000+.16
    static=args.static_context_diagnostic
    initial=role_text([{'status':'completed','project':meta['project'],'field':meta['field'],'value':value}]) if static else role_text()
    if static:
        holder['observed']=True
        events.append({'event':'static_oracle_context_at_start','project':meta['project'],'field':meta['field'],'value':value,
            'purpose':'reference-reading control only; not real-time result integration'})
    session=create_session(framework,initial,start_speak_token_factor=1.,start_listen_token_factor=1.,
        bc_speak_token_factor=1.,end_speak_token_factor=1.,audio_incremental_mode=True)
    if framework.vllm_engine._effective_prefix_caching:
        raise RuntimeError('Native audio/branch KV cannot reuse text-only prefix caching')
    def work(context,revision):
        result=run_worker(context,world,'lookup',expected_revision=revision)
        receipts=[r for r in result['receipts'] if r['tool']=='lookup_record']
        if result['route']!='delegate' or len(receipts)!=1 or receipts[0]['value']!=value:
            raise RuntimeError('Private diagnostic worker did not execute the actual intended record')
        world.mark_ready(receipts[0])
        return result,time.monotonic()
    from evals.lychee_transport import EventSender
    transport=EventSender(args.event_socket) if args.event_socket else nullcontext()
    with transport as sender,concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        def started(epoch):
            holder['start']=epoch
            if sender:sender.start(epoch)
        def before(current,frame,prior):
            nonlocal future
            if static:return current
            begin=round(frame['input_start_s']*16000);end=round(frame['input_available_s']*16000)
            holder['user_chunk_quiet']=float(np.sqrt(np.mean(audio[begin:end].astype(np.float32)**2)))<.0015
            can_deliver=delivery.observe(holder['user_chunk_quiet'],current.get_listening_state())
            if 'start' not in holder:holder['start']=time.monotonic()-frame['input_consumed_s']
            now=time.monotonic()-holder['start']
            if future is not None and future.done() and not holder.get('observed') and can_deliver:
                result,ready=future.result();holder['observed']=True
                receipt=result['receipts'][0]
                if not world.deliverable(receipt):raise RuntimeError('Stale tool receipt cannot enter native instructions')
                actual[:]=[{'status':'completed','tool':receipt['tool'],'project':receipt['project'],
                    'field':receipt['field'],'value':receipt['value'],'received_at_s':now}]
                began=time.monotonic()
                current,proof=refresh_session(current,role_text(actual))
                holder['result_wakeup_pending']=args.result_response=='application-event-diagnostic'
                events.append({'event':'actual_result_context_applied','ready_at_s':ready-holder['start'],
                    'observed_at_s':now,'delivery_hold_s':now-(ready-holder['start']),
                    'native_state_at_delivery':current.get_listening_state(),
                    'quiet_arrived_chunks':delivery.quiet_chunks,
                    'context_rebuild_s':time.monotonic()-began,'result':result,'preservation':proof})
            if future is None and frame['input_available_s']>=trigger:
                generated=[{'text':framework.tokenizer.decode([r['event']['text_token']],skip_special_tokens=True),
                    'packet_available_s':r['available_s']} for r in prior
                    if r['event'].get('text_token',151688)<151688]
                context=causal_dialogue(sidecar,frame['input_available_s'],generated,frame['input_consumed_s'])
                revision=world.reserve('lookup',context=context)
                future=pool.submit(work,context,revision)
                events.append({'event':'scheduled_worker_started','at_s':now,'context':context,
                    'input_available_s':frame['input_available_s'],'native_foreground_choice':False})
                actual[:]=[{'status':'pending','task_id':'lookup','project':meta['project'],'field':meta['field']}]
                began=time.monotonic()
                current,proof=refresh_session(current,role_text(actual))
                events.append({'event':'pending_context_applied','observed_at_s':now,
                    'context_rebuild_s':time.monotonic()-began,'preservation':proof})
            if args.collect_training_state and actual and actual[0]['status']=='completed':
                from evals.lychee_training_state import save
                captured_frame={**frame,'snapshot_elapsed_s':time.monotonic()-holder['start']}
                if not holder.get('ready_snapshot'):
                    holder['ready_snapshot']=save(current,args.out,'actual-ready-state',row,captured_frame,actual[0],role_text(actual))
                if (not holder.get('idle_snapshot') and current.get_listening_state()=='l'
                        and delivery.quiet_chunks>=2):
                    holder['idle_snapshot']=save(current,args.out,'actual-idle-state',row,captured_frame,actual[0],role_text(actual))
            return current
        try:
            from evals.lychee_wakeup import result_response_event
            response_event=result_response_event(holder,events) if args.result_response=='application-event-diagnostic' else nullcontext()
            with response_event:
                generated,timing=run_incremental(framework,audio,args.out,before_chunk=before,session=session,
                    on_start=started,on_event=sender.event if sender else None)
                if sender:sender.finish(holder['start'])
        finally:
            (args.out/'worker-events.json').write_text(json.dumps(events,indent=2)+'\n')
    if not holder.get('observed'):raise RuntimeError('Actual worker never reached native language context')
    if args.collect_training_state and not holder.get('idle_snapshot'):
        raise RuntimeError('Actual TRAIN trajectory never reached a quiet native listening boundary after completion')
    if any(not e['preservation'].get('actual_engine_request_verified') for e in events if 'preservation' in e):
        raise RuntimeError('Refreshed instructions were not verified at actual native engine boundary')
    (args.out/'native-trace.json').write_text(json.dumps(generated['trace'])+'\n')
    (args.out/'events.json').write_text(json.dumps(generated['events'],indent=2)+'\n')
    report={'text':' '.join(e['text'] for e in generated['events']),'paced_tokens':timing,'strict_loading':loading,
        'native_side_token_audit':side_audit,
        'scenario':args.scenario,'world_sha256':hashlib.sha256(json.dumps(world_data,sort_keys=True).encode()).hexdigest(),
        'concurrent_pcm_transport_requested':bool(sender),'native_transport_messages_sent':sender.sent if sender else 0,
        'source_split':row['split'],'actual_training_history_collected':args.collect_training_state,
        'training_state_collection_invalidates_timing':args.collect_training_state,
        'training_state_snapshots':[holder[k] for k in ('ready_snapshot','idle_snapshot') if k in holder],
        'worker_events':events,'scheduled_dispatch':not static,'native_foreground_tool_choice':False,
        'actual_worker_results_supplied':not static,'spoken_result_use_verified':False,'oracle_private_values_supplied':static,'assistant_output_tokens_forced':any(e['event']=='application_response_start_control' for e in events),
        'word_or_acoustic_tokens_forced':False,'result_response_mode':args.result_response,
        'result_delivery_mode':args.result_delivery,'application_schedules_result_delivery':args.result_delivery!='immediate',
        'native_speaking_control_forced':any(e['event']=='application_response_start_control' for e in events),
        'context_channel':'static supplied reference at start; oracle diagnostic' if static else 'nonspoken native instruction prefix rebuilt after actual receipt',
        'backend_context':'none; static reference diagnostic' if static else 'causal measured user words and generated native text; not live ASR',
        'live_pcm_tested':False,'microphone_acceptance':False,'quality_qualified':False,'training_approved':False,
        'task_success':'pending independent spoken-value and receipt grading'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('worker_events','strict_loading')}),flush=True)
