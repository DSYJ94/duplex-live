"""Actual-paced reasoning teacher transport, separate from scoring and routing.

Only native delegation events start backend calls. Only arrived USER transcripts
enter them. Follow-up audio waits for the first actual HTTP dispatch. No oracle,
intent classifier, output filtering, retry, answer-dependent release or cropping.
"""
import asyncio
import base64
import hashlib
import json
import math
from pathlib import Path
import time

from evals.native_control_teacher import MODEL, safe_event, atomic_json
from evals.native_reasoning_backend import request as backend_request
from evals.native_reasoning_lifecycle import NativeReasoningLifecycle

RATE=24000
FRAME=1920
MAXIMUM_INPUT_S=88.
FOLLOWUP_GAP_S=.56
RESPONSE_WINDOW_S=5.
MAXIMUM_DRAIN_S=20.


def configuration():
    return dict(model=MODEL,delegation=dict(type='client'),audio=dict(
        format=dict(type='audio/pcm',rate=RATE),output=dict(voice='marin')),
        instructions='You are a concise English voice assistant. Listen to the complete user question before answering or requesting reasoning; do not request help from unfinished premises. Answer basic questions and simple arithmetic yourself without delegation. Delegate difficult multi-constraint reasoning to the external backend instead of guessing. Continue listening and answer independent basic questions directly while work is pending. When the user cancels or replaces pending work, request backend help to update the actual task state; do not claim cancellation or replacement succeeded before its result arrives. Keep the full original premises when a later correction changes one rule. Never speak a stale cancelled or superseded result. After a current verified bundle result arrives, say its chosen items, total cost and total points concisely, once, in at most twenty words. A cancellation status contains no answer to a new basic question: answer that basic question yourself. Do not invent tool results, identifiers or user premises. There are no reference answers in this session.')


def input_frame(clips,starts,first):
    import numpy as np
    chunk=np.zeros(FRAME,dtype='<i2')
    for pcm,start in zip(clips,starts):
        if start is None:continue
        a=max(first,start);b=min(first+FRAME,start+len(pcm))
        if a<b:chunk[a-first:b-first]=pcm[a-start:b-start]
    return chunk


async def capture(ws,clips,out,emit,*,backend_model,env_file,minimum_delay_s,
                  input_sink,audio_sink,runner=backend_request,
                  now=time.monotonic,sleep=asyncio.sleep):
    """Retain every sample/event. A raw transport result is never an approval."""
    import numpy as np
    if (not 1<=len(clips)<=2 or any(not isinstance(p,np.ndarray) or p.dtype!=np.dtype('<i2')
            or p.ndim!=1 or not len(p) for p in clips)
            or type(minimum_delay_s) not in (int,float) or not math.isfinite(minimum_delay_s)
            or not 0<=minimum_delay_s<=30):raise ValueError('One or two complete PCM16 sources and explicit bounded delay required')
    ready=asyncio.Event();first_http=asyncio.Event();origin=None;closing=False
    starts=[round(FOLLOWUP_GAP_S*RATE)]+[None]*(len(clips)-1)
    transcripts=[];users=[];packets=[];errors=[];tasks={};http_starts={};acks=set()
    sent=0;cursor=0;closed_at=None;usage=None;source_completed=False
    first_delay=max(float(minimum_delay_s),FOLLOWUP_GAP_S+(len(clips[1])/RATE if len(clips)>1 else 0)+4.)

    def relative():return now()-origin
    def record(event):
        e=dict(safe_event(event),received_at_s=None if origin is None else relative());emit(e);return e
    world=NativeReasoningLifecycle(relative,first_delay)
    def save_world():atomic_json(out/'lifecycle.json',world.evidence())

    async def send_result(payload):
        if closing:raise RuntimeError('Session is closing; no late commentary or retry')
        await ws.send(json.dumps(payload))
        record(dict(type='local.commentary.write',**{k:v for k,v in payload.items() if k!='type'}))

    async def worker(identity):
        entered=asyncio.Event();folder=out/('backend-'+hashlib.sha256(identity.encode()).hexdigest()[:24])
        call=asyncio.create_task(runner(world,identity,backend_model,folder,env_file,started=entered))
        waiter=asyncio.create_task(entered.wait())
        try:
            done,_=await asyncio.wait((call,waiter),return_when=asyncio.FIRST_COMPLETED)
            # A completed HTTP call may race its start notification. The durable
            # trace is authoritative; never invent dispatch on an early failure.
            path=folder/'transport.json'
            if path.is_file():
                trace=json.loads(path.read_text())
                if 'http_call_started_monotonic' in trace:
                    actual_start=trace['http_call_started_monotonic']-origin
                    if not 0<=actual_start<=relative():raise ValueError('Actual HTTP clock outside measured episode')
                    http_starts[identity]=actual_start
                    record(dict(type='local.backend.http.started',task_id=identity,
                        actual_http_started_at_s=actual_start,transport_file=str(path.relative_to(out))))
                    first_http.set()
            receipt=await call
            if identity not in http_starts:raise ValueError('Actual HTTP-start evidence missing')
            save_world()
            if receipt is not None:
                eligible=max(receipt['delivery_eligible_at_s'],http_starts[identity]+receipt['minimum_delivery_delay_s'])
                while relative()<eligible and not world.tasks[identity]['cancelled']:
                    await sleep(min(.08,eligible-relative()))
                if not await world.publish(identity,send_result):
                    record(dict(type='local.backend.delivery.retired',task_id=identity))
                save_world()
        except Exception as error:
            errors.append(dict(type='backend_failure',task_id=identity,error_type=type(error).__name__))
            record(errors[-1]);save_world()
        finally:
            waiter.cancel();await asyncio.gather(waiter,return_exceptions=True)
            if not call.done():await asyncio.gather(call,return_exceptions=True)

    async def sender():
        nonlocal sent,closing,source_completed
        await asyncio.wait_for(ready.wait(),30)
        while sent/RATE<MAXIMUM_INPUT_S:
            if len(clips)>1 and starts[1] is None and first_http.is_set():
                target=max(starts[0]+len(clips[0])+round(FOLLOWUP_GAP_S*RATE),
                           math.ceil((min(http_starts.values())+FOLLOWUP_GAP_S)*RATE),sent)
                starts[1]=math.ceil(target/FRAME)*FRAME
                record(dict(type='local.followup.scheduled',first_sample=starts[1],
                    actual_first_HTTP_dispatch_at_s=min(http_starts.values()),
                    followup_gap_s=FOLLOWUP_GAP_S,answer_dependent=False))
            end=sent+FRAME
            await sleep(max(0.,origin+end/RATE-now()))
            chunk=input_frame(clips,starts,sent).tobytes()
            await ws.send(json.dumps(dict(type='session.input_audio.append',audio=base64.b64encode(chunk).decode())))
            input_sink(chunk)
            record(dict(type='local.input.append',first_sample=sent,sample_count=FRAME,
                        pcm_sha256=hashlib.sha256(chunk).hexdigest()))
            sent=end
            all_sent=all(start is not None and sent>=start+len(pcm) for pcm,start in zip(clips,starts))
            if all_sent and not source_completed:
                source_completed=True
                record(dict(type='local.source.completed',source_starts=starts,
                            source_lengths=[len(p) for p in clips],sent_input_samples=sent))
            if all_sent:
                not_before=max(start+len(pcm) for pcm,start in zip(clips,starts))/RATE+RESPONSE_WINDOW_S
                for identity,start in http_starts.items():
                    if world.tasks[identity]['cancelled']:continue
                    not_before=max(not_before,start+world.tasks[identity]['minimum_delivery_delay_s']+RESPONSE_WINDOW_S)
                if relative()>=not_before and all(t.done() for t in tasks.values()):break
            elif len(clips)>1 and not first_http.is_set() and relative()>(starts[0]+len(clips[0]))/RATE+RESPONSE_WINDOW_S:
                errors.append(dict(type='missing_native_dispatch_followup_not_invented'));record(errors[-1]);break
        if not source_completed or any(not t.done() for t in tasks.values()):
            errors.append(dict(type='incomplete_source_or_backend_at_bounded_close'));record(errors[-1])
        closing=True;record(dict(type='local.session.close.requested',sent_input_samples=sent))
        await ws.send(json.dumps(dict(type='session.close')))

    await ws.send(json.dumps(dict(type='session.start',event_id='reasoning32_start',session=configuration())))
    sender_task=asyncio.create_task(sender())
    try:
        async with asyncio.timeout(MAXIMUM_INPUT_S+60):
            async for message in ws:
                event=json.loads(message);kind=event.get('type')
                if kind=='session.started':
                    if ready.is_set() or event['session']['model']!=MODEL:raise ValueError('Unexpected live teacher session')
                    origin=now();ready.set()
                if kind=='session.output_audio.delta':
                    if origin is None:raise ValueError('Output before actual session start')
                    data=base64.b64decode(event['delta'],validate=True)
                    if not data or len(data)%2:raise ValueError('Invalid original output PCM16')
                    offset=sum(p['bytes'] for p in packets);audio_sink(data)
                    e=record({k:v for k,v in event.items() if k!='delta'}|dict(audio_bytes=len(data),
                        output_byte_offset=offset,pcm_sha256=hashlib.sha256(data).hexdigest()))
                    packets.append(dict(arrival_s=e['received_at_s'],byte_offset=offset,bytes=len(data),pcm_sha256=e['pcm_sha256']))
                    cursor=max(cursor,math.ceil(e['received_at_s']*RATE))+len(data)//2
                else:e=record(event)
                if kind in ('session.input_transcript.delta','session.output_transcript.delta'):
                    if origin is None:raise ValueError('Transcript before actual session start')
                    transcripts.append(e)
                    if kind=='session.input_transcript.delta' and event.get('delta','').strip():
                        users.append(dict(speaker='user',text=event['delta'],received_at_s=e['received_at_s']))
                if kind=='session.delegation.created':
                    delegation=event['delegation'];identity=delegation['id']
                    if closing or delegation.get('target')!='client' or identity in tasks:raise ValueError('Unexpected/repeated native delegation')
                    await world.reserve_native_request(identity,list(users),minimum_delivery_delay_s=first_delay if not tasks else 0.)
                    save_world();tasks[identity]=asyncio.create_task(worker(identity))
                if kind=='session.commentary.appended':acks.add(event.get('client_event_id'))
                if kind=='error':
                    errors.append(safe_event(event))
                    if origin is None:break
                if kind=='session.closed':
                    closed_at=e['received_at_s'];usage=event.get('usage');break
    except Exception as error:
        errors.append(dict(type='capture_failure',error_type=type(error).__name__))
        record(errors[-1])
    finally:
        closing=True;sender_task.cancel()
        result=await asyncio.gather(sender_task,return_exceptions=True)
        for value in result:
            if isinstance(value,BaseException) and not isinstance(value,asyncio.CancelledError):
                errors.append(dict(type='input_sender_failure',error_type=type(value).__name__))
        # Do not cancel an HTTP thread or discard its late raw response. Each
        # request already has its own finite socket timeout and no retry path.
        await asyncio.gather(*tasks.values(),return_exceptions=True);save_world()
    observed=0. if origin is None else relative()
    if closed_at is not None and packets:
        target=cursor/RATE+.16
        if target-observed>MAXIMUM_DRAIN_S:
            errors.append(dict(type='output_backlog_exceeds_drain_bound'))
        else:
            record(dict(type='local.output.drain.started',playout_end_sample=cursor))
            await sleep(max(0.,target-relative()));observed=relative()
            record(dict(type='local.output.drain.completed',playout_end_sample=cursor))
    for identity,task in world.tasks.items():
        if task['phase']=='wire_sent' and 'reasoning_result_'+identity not in acks:
            errors.append(dict(type='missing_commentary_ack',task_id=identity))
    report=dict(schema='dl-native-reasoning32-raw-stream-v1',session_finalized=closed_at is not None,
        server_closed_at_s=closed_at,observed_until_s=observed,usage=usage,errors=errors,
        source_starts=starts,source_lengths=[len(p) for p in clips],all_source_clips_sent=source_completed,
        sent_input_samples=sent,delegations=len(tasks),backend_HTTP_starts=http_starts,
        maximum_input_s=MAXIMUM_INPUT_S,first_result_minimum_delivery_delay_s=first_delay,
        output_packets=len(packets),commentary_acks=sorted(x for x in acks if isinstance(x,str)),
        native_request_forced=False,operator_or_oracle_inputs=False,output_suppression=False,
        automatic_retry=False,training_approved=False,semantic_qualified=False,
        complete_native_episode_bound_verified=False)
    report['transport_complete']=closed_at is not None and source_completed and bool(packets) and not errors
    return report,packets,transcripts
