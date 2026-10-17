"""Paced released Context Spanning component audition with actual late receipts.

Dispatch is scheduled and backend context uses measured words, not live ASR.
The native model samples every output word and audio code; no response wakeup.
"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evals.moshi_model_download import sha256
from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256,VOICE_SHA256


@contextmanager
def audited_loads(records):
    import torch
    original=torch.load;load_state=torch.nn.Module.load_state_dict
    def safe(*args,**kwargs):
        kwargs['weights_only']=True
        return original(*args,**kwargs)
    def checked(module,state,*args,**kwargs):
        result=load_state(module,state,*args,**kwargs)
        records.append({'module_identity':id(module),'class':type(module).__name__,
            'strict':kwargs.get('strict',args[0] if args else True),'source_keys':len(state),
            'missing':list(result.missing_keys),'unexpected':list(result.unexpected_keys)})
        return result
    torch.load=safe;torch.nn.Module.load_state_dict=checked
    try:yield
    finally:torch.load=original;torch.nn.Module.load_state_dict=load_state


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','base','data','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--scenario',default='recovery-validation-000')
    ap.add_argument('--private-value',required=True)
    ap.add_argument('--voice',choices=('released','accepted-synthetic'),default='released')
    ap.add_argument('--expected-kind',choices=('private','simple','clarify','cancel'))
    ap.add_argument('--voice-reference',type=Path)
    ap.add_argument('--voice-provenance',type=Path)
    ap.add_argument('--backend-mode',choices=('scheduled','native-asr','observe-only'),default='scheduled')
    ap.add_argument('--asr-python',type=Path)
    ap.add_argument('--warm-audio',action='store_true')
    ap.add_argument('--adapter-proof',type=Path)
    ap.add_argument('--adapter-disabled',action='store_true')
    ap.add_argument('--adapter-zero',action='store_true')
    ap.add_argument('--pending-status',action='store_true',
        help='Supply actual request-started/delivery-pending state as a context span; diagnostic')
    ap.add_argument('--math-attention', action='store_true',
        help='Explicit numerical replay diagnostic; use Torch math attention only')
    ap.add_argument('--train-history',action='store_true',
        help='Capture actual native conditioning on approved TRAIN stimuli; never supervised past labels')
    ap.add_argument('--layout-audit',type=Path,
        help='Explicit native-layout candidate; require a successful actual input trace')
    args=ap.parse_args()
    if args.backend_mode=='observe-only' and (
            not args.train_history or args.expected_kind not in ('simple','clarify') or args.pending_status):
        ap.error('Observe-only requires unapproved TRAIN simple/clarification conditioning without backend state')
    if args.backend_mode=='native-asr':
        devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        if len(devices)!=2 or len(set(devices))!=2 or not args.asr_python:
            ap.error('Native-ASR consultation requires two distinct assigned GPUs and its isolated Python')
        os.environ['CUDA_VISIBLE_DEVICES']=devices[0];args.asr_gpu=devices[1]
    import numpy as np
    import soundfile as sf
    import torch
    if args.math_attention:
        from evals.context_execution import math_attention
        math_attention()
    from evals.personaplex_train import MODEL_HASHES
    from evals.recovery_data import validate_conversion
    from evals.recovery_context import causal_dialogue
    from evals.recovery_worker import RecordWorld,run_worker
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.audio_playout import render_packets
    assets=json.loads(args.assets.read_text());source=Path(assets['source']);weights=Path(assets['path'])
    if (assets['source_commit']!=SOURCE_COMMIT or assets['revision']!=MODEL_REVISION
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Context Spanning source/revision changed')
    for name,record in assets['files'].items():
        if sha256(weights/name)!=record['sha256']:raise RuntimeError('Context Spanning asset changed')
    if (assets['files']['context_spanning_7b.pt']['sha256']!=WEIGHT_SHA256
            or assets['files']['voices/main.pt']['sha256']!=VOICE_SHA256):
        raise RuntimeError('Assets differ from pinned upstream LFS digests')
    for name,digest in MODEL_HASHES.items():
        if sha256(args.base/name)!=digest:raise RuntimeError('PersonaPlex base asset changed')
    data=validate_conversion(args.data)
    row=next(r for r in data['accepted'] if r['scenario_id']==args.scenario)
    if args.expected_kind and row['kind'] != args.expected_kind:
        raise RuntimeError('Selected scene differs from the explicitly requested behavior kind')
    expected_split='train' if args.train_history else 'validation'
    if row['split']!=expected_split or row['kind'] not in ('private','simple','clarify','cancel'):
        raise RuntimeError('Approved development behavior scene required')
    if args.train_history and args.backend_mode not in ('native-asr','observe-only'):
        raise RuntimeError('Student histories require actual native requests and arrived-audio ASR')
    args.source_split=row['split']
    if row['kind']!='private' and args.backend_mode not in ('native-asr','observe-only'):
        raise RuntimeError('Control scenes require native backend decisions and actual ASR')
    if len(args.private_value.split())!=2:raise RuntimeError('Expected declared two-word development value')
    side=json.loads(Path(row['path']).with_suffix('.json').read_text());meta=side['timeline']['meta']
    audio,rate=sf.read(row['path'],dtype='float32',always_2d=True)
    if rate!=24000 or audio.shape[1]!=2 or not np.isfinite(audio).all():raise RuntimeError('Invalid native stereo source')
    user=np.pad(audio[:,1],(0,20*rate));args.out.mkdir(exist_ok=False)
    sf.write(args.out/'user_in.wav',user,rate)
    os.environ.update(CS_BASE_DIR=str(args.base),CS_WEIGHTS_DIR=str(weights),CS_VOICES_DIR=str(weights/'voices'))
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-m','evals.cuda_linker_path',
        '--out',str(args.out/'driver')],text=True).strip()
    checkpoint=weights/'context_spanning_7b.pt'
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    if unsafe:raise RuntimeError('Checkpoint requires unsupported pickle globals: '+repr(unsafe))
    sys.path.insert(0,str(source))
    from contextspan.model.engine import Engine,seed_all
    import contextspan.model.engine as engine_module
    from evals.context_adapter_load import before_engine
    from contextspan.model.weights import load_voice
    import inspect
    if Path(inspect.getfile(Engine)).resolve()!=(source/'contextspan/model/engine.py').resolve():
        raise RuntimeError('Wrong Context Spanning engine module imported')
    records=[];started=time.monotonic()
    with before_engine(engine_module,args.adapter_proof,args.adapter_disabled,args.adapter_zero) as adapter:
        with audited_loads(records):
            engine=Engine(checkpoint=str(checkpoint),device='cuda',temp=.8,temp_text=.7)
            voice=load_voice('main')
    lm_loads=[r for r in records if r['module_identity']==id(engine.lm)]
    if (type(engine.lm).load_state_dict is not torch.nn.Module.load_state_dict
            or not lm_loads or not lm_loads[-1]['strict'] or lm_loads[-1]['missing'] or lm_loads[-1]['unexpected']):
        raise RuntimeError('Final Context Spanning checkpoint did not strictly cover native LM')
    if engine.frame_size!=1920 or engine.frame_rate!=12.5:raise RuntimeError('Native frame clock changed')
    layout=None
    if args.layout_audit:
        layout=json.loads(args.layout_audit.read_text())
        if (not layout['candidate_passed'] or layout['source_commit']!=SOURCE_COMMIT
                or layout['layout_code_sha256']!=sha256(Path(__file__).with_name('context_native_layout.py'))):
            raise RuntimeError('Native-layout candidate lacks matching actual input parity')
        from evals.context_native_layout import install
        if not getattr(engine,'_dl_checkpoint_native_protocol',None):install(engine)
        layout={'audit_sha256':sha256(args.layout_audit),'native_layout':'bootstrap prefix and preserve provided acoustics',
            'native_input_parity_passed':True,'quality_qualified':False}
        from evals.context_span_trace import attach
        layout['actual_free_running_span_checks']=attach(engine)
    elif getattr(engine,'_dl_checkpoint_native_protocol',None):
        layout={'native_layout':'bootstrap prefix and preserve provided acoustics',
            'checkpoint_required':True,'native_protocol':engine._dl_checkpoint_native_protocol,
            'quality_qualified':False}
        from evals.context_span_trace import attach
        layout['actual_free_running_span_checks']=attach(engine)
    if args.voice=='accepted-synthetic':
        if not args.voice_reference or not args.voice_provenance:raise RuntimeError('Synthetic voice provenance required')
        receipt=json.loads(args.voice_provenance.read_text())
        if (receipt['schema']!='dl-voice-codec-probe-v1' or receipt['source_model']!='gpt-live-1'
                or receipt['reference_sha256']!=sha256(args.voice_reference)):
            raise RuntimeError('Accepted synthetic voice provenance changed')
        reference,voice_rate=sf.read(args.voice_reference,dtype='float32')
        if reference.ndim!=1 or not np.isfinite(reference).all():raise RuntimeError('Invalid synthetic reference')
        if args.adapter_proof:
            if adapter['voice_reference_sha256']!=sha256(args.voice_reference):
                raise RuntimeError('Adapter training voice differs from audition voice')
            voice=torch.load(args.adapter_proof/'voice.pt',map_location='cpu',weights_only=True)['codes']
        else:
            voice=engine.clone_voice(reference,voice_rate)
    elif args.adapter_proof:
        raise RuntimeError('Adapter comparisons require the exact accepted synthetic voice codes')
    warm=None
    if args.warm_audio:
        began=time.monotonic()
        for _ in range(16):engine.step(np.zeros(engine.frame_size,dtype=np.float32))
        torch.cuda.synchronize();warm={'frames':16,'seconds':time.monotonic()-began,
            'before_conversation_seed_reset':True,'all_streams_reset_afterwards':True}
    seed_all(42424242);engine.reset()
    history=None
    if args.train_history:
        from evals.context_student_history import Recorder
        torch.save({'codes':voice.detach().cpu()},args.out/'student-voice.pt')
        history=Recorder(engine,args.out,{
            'split':row['split'],'scenario':row['scenario_id'],'kind':row['kind'],
            'conversion':str(args.data),'ingestion_sha256':sha256(args.data/'ingestion.json'),
            'wav_sha256':row['wav_sha256'],'sidecar_sha256':row['sidecar_sha256'],
            'assets_manifest_sha256':sha256(args.assets),
            'voice_codes_sha256':sha256(args.out/'student-voice.pt'),
            'recorder_code_sha256':sha256(Path(__file__).with_name('context_student_history.py')),
            'probe_code_sha256':sha256(Path(__file__)),
            'native_run_code_sha256':sha256(Path(__file__).with_name('context_spanning_native.py')),
            'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    engine.set_persona(BASE_INSTRUCTIONS,voice)
    if history:
        history.mode='live'
        engine._dl_student_history=history
    torch.cuda.synchronize()
    loading={'loads':records,'final_lm_strict':True,'dep_q':engine.lm.dep_q,
        'scenario':args.scenario,'scenario_kind':row['kind'],
        'model_load_and_prefix_s':time.monotonic()-started,'pickle_globals':unsafe,
        'assets_manifest_sha256':sha256(args.assets),'base_hashes':MODEL_HASHES,
        'voice':args.voice,'voice_codes_shape':list(voice.shape),'input_warmup':warm,'adapter':adapter,'layout':layout,
        'torch':torch.__version__,'torch_file':torch.__file__,'cuda':torch.version.cuda}
    (args.out/'loading.json').write_text(json.dumps(loading,indent=2)+'\n')
    world=RecordWorld({meta['project']:{meta['field']:args.private_value}} if row['kind']!='simple' else {})
    if history:
        history.source['declared_counterfactual_world']=world.records
        history.source['world_available_to_model_before_actual_receipt']=False
    if args.backend_mode=='observe-only':
        from evals.context_spanning_native import observe_only
        observe_only(engine,user,args,loading)
        return
    if args.backend_mode=='native-asr':
        from evals.context_spanning_native import run
        run(engine,user,world,args,loading)
        return
    first=next(e for e in side['timeline']['events'] if e['ch']=='user' and e['kind']=='utterance')
    trigger=first['actual_end_ms']/1000+.16
    rows=[];tokens=[];packets=[];events=[];future=None;observed=False
    def worker(context,revision):
        result=run_worker(context,world,'lookup',expected_revision=revision)
        actual=[r for r in result['receipts'] if r['tool']=='lookup_record']
        if result['route']!='delegate' or len(actual)!=1 or actual[0]['value']!=args.private_value:
            raise RuntimeError('Worker failed to execute actual declared private world')
        world.mark_ready(actual[0]);return result,time.monotonic()
    epoch=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with (args.out/'events.jsonl').open('w') as log:
                for offset in range(0,len(user),engine.frame_size):
                    end=min(len(user),offset+engine.frame_size);due=end/rate
                    time.sleep(max(0,epoch+due-time.monotonic()));before=time.monotonic()-epoch
                    if future is not None and future.done() and not observed:
                        result,ready=future.result();receipt=result['receipts'][0]
                        if not world.deliverable(receipt):raise RuntimeError('Stale receipt cannot enter a context span')
                        reference=(f"The lookup completed successfully. The actual {receipt['field']} for "
                            f"Project {receipt['project']} is {receipt['value']}.")
                        frames=engine.inject_context_span(reference);observed=True
                        events.append({'event':'actual_result_span_applied','ready_at_s':ready-epoch,
                            'applied_at_s':time.monotonic()-epoch,'virtual_span_frames':frames,
                            'prefill_ms':engine.last_prefill_ms,'result':result,
                            'real_user_samples_skipped':0,'speaking_control_forced':False})
                    if future is None and due>=trigger:
                        prior=[{'text':engine.decode_text(tokens),'packet_available_s':rows[-1]['available_s']}] if rows else []
                        context=causal_dialogue(side,due,prior,before);revision=world.reserve('lookup',context=context)
                        future=pool.submit(worker,context,revision)
                        events.append({'event':'scheduled_worker_started','at_s':before,'context':context,
                            'native_foreground_choice':False})
                    chunk=user[offset:end]
                    if len(chunk)<engine.frame_size:chunk=np.pad(chunk,(0,engine.frame_size-len(chunk)))
                    result=engine.step(chunk);torch.cuda.synchronize();available=time.monotonic()-epoch
                    token=result['text_token'];tokens.append(token)
                    pcm=result['agent_pcm']
                    if pcm is not None:
                        pcm=np.asarray(pcm,dtype=np.float32)
                        if pcm.shape!=(engine.frame_size,) or not np.isfinite(pcm).all():raise RuntimeError('Invalid native PCM')
                        packets.append((available,pcm))
                    record={'input_start_s':offset/rate,'input_available_s':due,'started_s':before,
                        'available_s':available,'compute_s':available-before,'input_backlog_s':before-due,
                        'input_samples':end-offset,'output_samples':0 if pcm is None else len(pcm),
                        'text_token':token,'native_retrieval_marker':result['is_ret']}
                    rows.append(record);log.write(json.dumps(record)+'\n');log.flush()
        finally:(args.out/'worker-events.json').write_text(json.dumps(events,indent=2)+'\n')
    if not observed or not packets:raise RuntimeError('No actual span or native PCM')
    if sum(r['input_samples'] for r in rows)!=len(user):raise RuntimeError('Actual user samples were skipped')
    output,playout=render_packets(packets,rate,.16)
    if not np.any(np.abs(output)>.001):raise RuntimeError('Native output is silent')
    sf.write(args.out/'model.wav',output,rate)
    sf.write(args.out/'packet-stream.wav',np.concatenate([pcm for _,pcm in packets]),rate)
    report={'schema':'dl-context-spanning-probe-v1','text':engine.decode_text(tokens),'scenario':args.scenario,
        'loading':loading,'source_split':'validation','fresh_holdout':False,'worker_events':events,
        'scheduled_dispatch':True,'native_foreground_tool_choice':False,
        'native_retrieval_markers':sum(r['native_retrieval_marker'] for r in rows),
        'word_or_acoustic_tokens_forced':False,'response_start_forced':False,
        'backend_context':'causal measured words; diagnostic oracle, not live ASR',
        'input_seconds':len(user)/rate,'wall_seconds':time.monotonic()-epoch,
        'compute_p95_s':float(np.percentile([r['compute_s'] for r in rows],95)),
        'maximum_input_backlog_s':max(r['input_backlog_s'] for r in rows),
        'real_user_samples_processed':sum(r['input_samples'] for r in rows),'real_user_samples_expected':len(user),
        'playout':playout,'audio_sha256':sha256(args.out/'model.wav'),
        'actual_lm_and_pcm_executed_together':True,'playout_is_causal_simulation':True,
        'microphone_acceptance':False,'spoken_value_use_verified':False,'quality_qualified':False,'training_approved':False}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('loading','worker_events','playout')}),flush=True)


if __name__=='__main__':main()
