"""Native backend-request markers, real arrived-audio ASR and actual tools.

The speech model decides when to consult the backend. The backend selects tools
or supplies reasoning; this does not make tool selection speech-model-native.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from evals.moshi_model_download import sha256


def run(engine,user,world,args,loading):
    import numpy as np
    import soundfile as sf
    import torch
    from evals.context_spanning_asr import context as recognized_context
    from evals.recovery_worker import run_worker,WorkerContractError
    from evals.audio_playout import render_packets
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=args.asr_gpu
    env.pop('OPENAI_API_KEY',None)
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    asr_log=(args.out/'asr.log').open('w')
    process=subprocess.Popen([str(args.asr_python),'-u','-m','evals.context_spanning_asr',
        '--root',str(args.out)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=asr_log,text=True,bufsize=1)
    lock=threading.Lock();events=[];pending={};tokens=[];rows=[];packets=[];epoch=None
    def work(identity,path,due,assistant,available,revision):
        with lock:
            process.stdin.write(json.dumps({'id':identity,'path':str(path)})+'\n');process.stdin.flush()
            line=process.stdout.readline()
            if not line:raise RuntimeError('ASR service exited; inspect complete asr.log')
            recognized=json.loads(line)
            if recognized['id']!=identity or abs(recognized['input_seconds']-due)>.0001:
                raise RuntimeError('ASR request/source clock mismatch')
        context=recognized_context(recognized,due,assistant,available)
        with world.lock:world.tasks[identity]['context']=context
        try:
            result=run_worker(context,world,identity,expected_revision=revision)
        except WorkerContractError as error:
            from evals.context_backend_failure import notice
            result=notice(error,identity)
        if result['route']!='error':
            for receipt in result['receipts']:
                if receipt['tool']=='lookup_record':world.mark_ready(receipt)
        return result,recognized,context,time.monotonic()
    try:
        asr_ready=json.loads(process.stdout.readline())
        if not asr_ready.get('ready'):raise RuntimeError('ASR failed readiness')
        epoch=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            try:
                with (args.out/'events.jsonl').open('w') as log:
                    for offset in range(0,len(user),engine.frame_size):
                        end=min(len(user),offset+engine.frame_size);due=end/24000
                        time.sleep(max(0,epoch+due-time.monotonic()));before=time.monotonic()-epoch
                        for identity,future in list(pending.items()):
                            if not future.done():continue
                            with world.lock:cancelled=world.tasks[identity]['cancelled']
                            if cancelled:
                                del pending[identity]
                                events.append({'event':'cancelled_backend_rejected','task_id':identity,
                                    'exception':str(future.exception()) if future.exception() else None})
                                continue
                            result,recognized,context,ready_at=future.result();del pending[identity]
                            record={'event':'backend_request_failed' if result['route']=='error' else 'backend_result_ready',
                                'task_id':identity,'result':result,
                                'asr':recognized,'context':context,'ready_at_s':ready_at-epoch}
                            with world.lock:state=world.tasks[identity]
                            if state['cancelled']:
                                record.update(disposition='rejected; cancelled task',applied_at_s=None)
                            else:
                                if any(not world.deliverable(r) for r in result['receipts'] if r['tool']=='lookup_record'):
                                    raise RuntimeError('Stale actual record receipt')
                                history=getattr(engine,'_dl_student_history',None)
                                if history:history.mode='span'
                                try:
                                    frames=engine.inject_context_span(result['content'])
                                finally:
                                    if history:history.mode='live'
                                record.update(disposition='failure notice supplied; no verified result' if result['route']=='error'
                                    else 'actual backend text injected',applied_at_s=time.monotonic()-epoch,
                                    prefill_ms=engine.last_prefill_ms,virtual_span_frames=frames)
                                if history:
                                    record['history_virtual_frames_after_span']=history.frames
                                with world.lock:world.tasks[identity]['phase']='failed' if result['route']=='error' else 'delivered'
                            events.append(record)
                        chunk=user[offset:end]
                        if len(chunk)<engine.frame_size:chunk=np.pad(chunk,(0,engine.frame_size-len(chunk)))
                        history=getattr(engine,'_dl_student_history',None)
                        history_before=history.frames if history else None
                        result=engine.step(chunk);torch.cuda.synchronize();available=time.monotonic()-epoch
                        tokens.append(result['text_token']);pcm=result['agent_pcm']
                        if pcm is not None:
                            pcm=np.asarray(pcm,dtype=np.float32)
                            if pcm.shape!=(engine.frame_size,) or not np.isfinite(pcm).all():raise RuntimeError('Invalid native PCM')
                            packets.append((available,pcm))
                        row={'input_start_s':offset/24000,'input_available_s':due,'started_s':before,
                            'available_s':available,'compute_s':available-before,'input_backlog_s':before-due,
                            'input_samples':end-offset,'output_samples':0 if pcm is None else len(pcm),
                            'text_token':result['text_token'],'native_retrieval_marker':result['is_ret']}
                        if history:
                            row.update(history_virtual_frames_before_step=history_before,
                                history_virtual_frames_after_step=history.frames)
                        rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                        if result['is_ret']:
                            count=sum(e['event']=='native_backend_requested' for e in events)
                            if count>=6:raise RuntimeError('Native model exceeded six bounded backend requests')
                            identity=f'ret-{count}';path=args.out/f'{identity}-arrived-user.wav'
                            sf.write(path,user[:end],24000,subtype='FLOAT')
                            revision=world.reserve(identity)
                            pending[identity]=pool.submit(work,identity,path,due,engine.decode_text(tokens),available,revision)
                            events.append({'event':'native_backend_requested','task_id':identity,'at_s':available,
                                'input_available_s':due,'input_samples':end,'audio_sha256':sha256(path),
                                'native_trigger':True,'reference_transcript_supplied':False})
                            if args.pending_status:
                                reference=('The current backend request is awaiting a reply. '
                                    'No verified answer for this request has been delivered yet.')
                                if history:history.mode='span'
                                try:
                                    frames=engine.inject_context_span(reference)
                                finally:
                                    if history:history.mode='live'
                                events.append({'event':'backend_pending_status_applied','task_id':identity,
                                    'applied_at_s':time.monotonic()-epoch,'virtual_span_frames':frames,
                                    'reference':reference,'source':'actual native request dispatch; no result supplied',
                                    'word_or_acoustic_tokens_forced':False})
            finally:
                for identity,future in pending.items():
                    events.append({'event':'input_exhausted_with_backend_pending','task_id':identity,'done_at_input_end':future.done()})
    finally:
        if process.stdin:process.stdin.close()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
        asr_log.close()
        (args.out/'worker-events.json').write_text(json.dumps(events,indent=2)+'\n')
    if not packets or epoch is None:raise RuntimeError('Native consultation produced no PCM')
    if sum(r['input_samples'] for r in rows)!=len(user):raise RuntimeError('Real user samples skipped')
    output,playout=render_packets(packets,24000,.16)
    sf.write(args.out/'model.wav',output,24000)
    sf.write(args.out/'packet-stream.wav',np.concatenate([p for _,p in packets]),24000)
    history=getattr(engine,'_dl_student_history',None)
    history_receipt=history.finish() if history else None
    report={'schema':'dl-context-spanning-native-asr-v1','text':engine.decode_text(tokens),'loading':loading,
        'scenario':args.scenario,'source_split':args.source_split,'fresh_holdout':False,'worker_events':events,
        'student_history':history_receipt,
        'scheduled_dispatch':False,'native_foreground_backend_request':True,'native_foreground_tool_choice':False,
        'explicit_pending_status':args.pending_status,
        'backend_contract_failures':sum(e['event']=='backend_request_failed' for e in events),
        'backend_context':'actual arrived user PCM recognized by Whisper turbo; no measured/reference transcript',
        'distinct_lm_asr_gpu_processes':True,'native_retrieval_markers':sum(r['native_retrieval_marker'] for r in rows),
        'asr_environment':asr_ready,
        'input_seconds':len(user)/24000,'wall_seconds':time.monotonic()-epoch,
        'compute_p95_s':float(np.percentile([r['compute_s'] for r in rows],95)),
        'maximum_input_backlog_s':max(r['input_backlog_s'] for r in rows),
        'real_user_samples_processed':sum(r['input_samples'] for r in rows),'real_user_samples_expected':len(user),
        'playout':playout,'audio_sha256':sha256(args.out/'model.wav'),
        'word_or_acoustic_tokens_forced':False,'response_start_forced':False,
        'actual_lm_and_pcm_executed_together':True,'playout_is_causal_simulation':True,
        'microphone_acceptance':False,'spoken_value_use_verified':False,'quality_qualified':False,'training_approved':False}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('loading','worker_events','playout')}),flush=True)
