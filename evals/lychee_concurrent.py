"""Launch isolated LM and vocoder processes on two assigned GPUs."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','batch','runtime','preflight','voice','voice-provenance','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--worker-context',action='store_true')
    ap.add_argument('--data',type=Path)
    ap.add_argument('--scenario')
    ap.add_argument('--private-value')
    ap.add_argument('--result-response',choices=('model','application-event-diagnostic'),default='model')
    ap.add_argument('--semantic-adapter',type=Path)
    ap.add_argument('--semantic-proof',type=Path)
    ap.add_argument('--semantic-activation',choices=('always','result-response'),default='always')
    args=ap.parse_args()
    if args.worker_context and (not args.data or not args.scenario):ap.error('Actual worker capture needs approved data/scenario')
    if bool(args.semantic_adapter)!=bool(args.semantic_proof):ap.error('Adapter and native implementation proof are both required')
    if args.semantic_activation!='always' and not args.semantic_adapter:ap.error('Conditional activation requires a checked adapter')
    if not args.worker_context and (args.private_value or args.result_response!='model'):
        ap.error('Private world and response event are actual-worker diagnostics only')
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not d for d in devices):
        raise RuntimeError('Exactly two distinct assigned GPUs required for Exclusive_Process mode')
    args.out.mkdir(exist_ok=False)
    children=[];handles=[]
    with tempfile.TemporaryDirectory(prefix='dl-lychee-') as directory:
        socket_path=Path(directory)/'events.sock';ready=Path(directory)/'ready'
        def spawn(module,device,overlay,extra):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
            env['PYTHONPATH']=os.environ[overlay]
            if module.endswith('_vocoder'):env.pop('OPENAI_API_KEY',None)
            log=(args.out/(module.rsplit('.',1)[-1]+'.log')).open('w');handles.append(log)
            child=subprocess.Popen([sys.executable,'-u','-m',module,*extra],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(child);return child
        def flags(names):
            return [part for name in names for part in ('--'+name,str(getattr(args,name.replace('-','_'))))]
        began=time.monotonic()
        try:
            tts=spawn('evals.lychee_concurrent_vocoder',devices[1],'DL_VOCODER_PYTHONPATH',
                flags(('assets','voice','voice-provenance'))+['--socket',str(socket_path),'--ready',str(ready),
                    '--out',str(args.out/'vocoder'),'--hop','8'])
            while not ready.exists():
                if tts.poll() is not None:raise RuntimeError('Vocoder startup failed; inspect its complete log')
                if time.monotonic()-began>150:raise RuntimeError('Vocoder readiness timeout')
                time.sleep(.2)
            lm_flags=flags(('assets','batch','runtime','preflight'))+['--warm-input','--event-socket',str(socket_path),
                    '--out',str(args.out/'lm')]
            if args.worker_context:
                lm_flags+=['--worker-context',*flags(('data','scenario','result-response'))]
                if args.private_value:lm_flags+=flags(('private-value',))
            if args.semantic_adapter:lm_flags+=flags(('semantic-adapter','semantic-proof','semantic-activation'))
            lm=spawn('evals.lychee_vllm_probe',devices[0],'DL_LM_PYTHONPATH',lm_flags)
            deadline=time.monotonic()+600
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None,0) for child in children):
                    raise RuntimeError('Concurrent child failed; inspect both complete logs')
                if tts.poll()==0 and lm.poll() is None:
                    # Native LM may still be writing its final tensor/report audit.
                    pass
                if time.monotonic()>deadline:raise RuntimeError('Concurrent execution exceeded bounded deadline')
                time.sleep(.2)
            if any(child.returncode!=0 for child in children):raise RuntimeError('Concurrent child failed')
            model=json.loads((args.out/'lm/report.json').read_text())
            audio=json.loads((args.out/'vocoder/report.json').read_text())
            events=json.loads((args.out/'lm/events.json').read_text())
            expected=sum(len(e['audio']) for e in events)
            if (expected!=audio['expected_codec_units'] or model['native_side_token_audit']['fallbacks']!=0
                    or not model['strict_loading']['model']['all_parameter_shards_loaded']):
                raise RuntimeError('Concurrent model/transport/audio accounting mismatch')
            result={'schema':'dl-lychee-concurrent-components-v1','distinct_gpu_processes':True,
                'text':model['text'],'paced_tokens':model['paced_tokens'],
                'expected_codec_units':expected,'pcm_capture':audio,
                'wall_including_both_model_loads_s':time.monotonic()-began,
                'microphone_acceptance':False,'browser_playback_tested':False,'audio_device_tested':False,
                'native_delegation_tested':False,'quality_qualified':False,'training_approved':False}
            result['semantic_activation_mode']=model['semantic_activation_mode']
            if args.worker_context:
                result.update(scheduled_worker_dispatch=model['scheduled_dispatch'],
                    actual_worker_results_supplied=model['actual_worker_results_supplied'],
                    native_foreground_tool_choice=model['native_foreground_tool_choice'],
                    result_response_mode=model['result_response_mode'],
                    native_speaking_control_forced=model['native_speaking_control_forced'],
                    application_chooses_semantic_result_turn=model['application_chooses_semantic_result_turn'],
                    worker_events=model['worker_events'],spoken_result_use_verified=False)
            (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='pcm_capture'}),flush=True)
        finally:
            for child in children:
                if child.poll() is None:child.terminate()
            for child in children:
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
            for handle in handles:handle.close()


if __name__=='__main__':main()
