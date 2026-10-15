"""Actual-worker result integration diagnostic; dispatch is explicitly scheduled.

The native voice receives completed text in its nonspoken instruction prefix.
It continues hearing every arrived user sample and samples all assistant output.
Backend context uses measured user words, not live ASR. This cannot qualify native
foreground delegation choice or a microphone product.
"""
import concurrent.futures
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
    data=validate_conversion(args.data)
    row=next(r for r in data['accepted'] if r['scenario_id']==args.scenario)
    if row['split']!='validation' or row['kind']!='private':
        raise RuntimeError('Worker-context diagnostic requires an approved private validation scene')
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
    initial=role_text()
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        def before(current,frame,prior):
            nonlocal future
            if 'start' not in holder:holder['start']=time.monotonic()-frame['input_consumed_s']
            now=time.monotonic()-holder['start']
            if future is not None and future.done() and not holder.get('observed'):
                result,ready=future.result();holder['observed']=True
                receipt=result['receipts'][0]
                if not world.deliverable(receipt):raise RuntimeError('Stale tool receipt cannot enter native instructions')
                actual[:]=[{'status':'completed','tool':receipt['tool'],'project':receipt['project'],
                    'field':receipt['field'],'value':receipt['value'],'received_at_s':now}]
                began=time.monotonic()
                current,proof=refresh_session(current,role_text(actual))
                events.append({'event':'actual_result_context_applied','ready_at_s':ready-holder['start'],
                    'observed_at_s':now,'context_rebuild_s':time.monotonic()-began,'result':result,'preservation':proof})
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
            return current
        try:
            generated,timing=run_incremental(framework,audio,args.out,before_chunk=before,session=session)
        finally:
            (args.out/'worker-events.json').write_text(json.dumps(events,indent=2)+'\n')
    if not holder.get('observed'):raise RuntimeError('Actual worker never reached native language context')
    (args.out/'native-trace.json').write_text(json.dumps(generated['trace'])+'\n')
    (args.out/'events.json').write_text(json.dumps(generated['events'],indent=2)+'\n')
    report={'text':' '.join(e['text'] for e in generated['events']),'paced_tokens':timing,'strict_loading':loading,
        'native_side_token_audit':side_audit,
        'scenario':args.scenario,'world_sha256':hashlib.sha256(json.dumps(world_data,sort_keys=True).encode()).hexdigest(),
        'worker_events':events,'scheduled_dispatch':True,'native_foreground_tool_choice':False,
        'actual_worker_results_used':True,'oracle_private_values_supplied':False,'assistant_output_tokens_forced':False,
        'context_channel':'nonspoken native instruction prefix rebuilt after actual receipt',
        'backend_context':'causal measured user words and generated native text; not live ASR',
        'live_pcm_tested':False,'microphone_acceptance':False,'quality_qualified':False,'training_approved':False,
        'task_success':'pending independent spoken-value and receipt grading'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('worker_events','strict_loading')}),flush=True)
