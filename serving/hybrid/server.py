"""Single-session live transport development server, separate from acceptance.

Main thread owns both speech GPU services. Network thread transports actual PCM
and receipts through bounded queues. Browser permission and physical AEC are not
validated by replay or by requested capture settings.
"""
import argparse
import asyncio
from concurrent.futures import Future
from http import HTTPStatus
import json
import os
from pathlib import Path
import queue
import re
import secrets
import struct
import threading
import time
from urllib.parse import urlsplit
from serving.hybrid.frames import LiveFrames


class Output:
    def __init__(self):self.queue=queue.Queue(maxsize=64);self.sequence=0;self.ready=threading.Event();self.disconnected=threading.Event()
    def put(self,value):
        if self.disconnected.is_set():return
        try:self.queue.put_nowait(value)
        except queue.Full as error:raise RuntimeError('Live output backpressure limit reached; no silent packet loss') from error
    def event(self,value):
        if value['kind']=='session_ready':self.ready.set()
        self.put(json.dumps(value,allow_nan=False))
    def pcm(self,identity,pcm,available_s):
        import numpy as np
        number=int(identity.removeprefix('speech-'));header=struct.pack('<II',number,self.sequence);self.sequence+=1
        self.put(header+np.asarray(pcm,dtype='<f4').tobytes())


class Network:
    def __init__(self,host,port,token,requests,stop):
        self.host=host;self.port=port;self.token=token;self.requests=requests;self.stop=stop;self.ready=threading.Event();self.error=None;self.busy=False
    async def http(self,connection,request):
        path=urlsplit(request.path).path
        if path=='/ws':return None
        files={'/':'index.html','/client.js':'client.js','/worklet.js':'worklet.js','/playout.js':'playout.js','/qa.html':'qa.html'}
        if path not in files:return connection.respond(HTTPStatus.NOT_FOUND,'Not found\n')
        body=(Path(__file__).parent/files[path]).read_text()
        response=connection.respond(HTTPStatus.OK,body);del response.headers['Content-Type'];response.headers['Content-Type']='text/html; charset=utf-8' if path in ('/','/qa.html') else 'text/javascript; charset=utf-8'
        response.headers['Cache-Control']='no-store';response.headers['Referrer-Policy']='no-referrer';return response
    async def handler(self,ws):
        incoming=None;output=None;sender=None;future=None;claimed=False
        try:
            hello=json.loads(await asyncio.wait_for(ws.recv(),timeout=5))
            if not isinstance(hello,dict) or not isinstance(hello.get('token'),str) or not secrets.compare_digest(hello['token'],self.token):
                await ws.close(code=1008,reason='Unauthorized');return
            if self.busy:await ws.close(code=1013,reason='A session is already running');return
            if hello.get('sample_rate')!=24000 or hello.get('frame_samples')!=1920:await ws.close(code=1008,reason='Unsupported PCM format');return
            settings=hello.get('capture_settings',{})
            if not isinstance(settings,dict) or len(json.dumps(settings))>2048:raise ValueError('Invalid bounded capture settings')
            self.busy=True;claimed=True;incoming=LiveFrames();output=Output();future=Future()
            self.requests.put_nowait(dict(incoming=incoming,output=output,future=future,capture_settings=settings,client_kind=hello.get('client_kind','unspecified')))
            async def send():
                try:
                    while True:
                        if output.disconnected.is_set():return
                        try:packet=await asyncio.to_thread(output.queue.get,True,.2)
                        except queue.Empty:continue
                        if packet is None:return
                        await ws.send(packet)
                except Exception:
                    output.disconnected.set();incoming.finish()
                    raise
            sender=asyncio.create_task(send())
            async for message in ws:
                if isinstance(message,bytes):
                    if not output.ready.is_set():raise ValueError('Capture arrived before model session readiness')
                    incoming.feed(message)
                else:
                    control=json.loads(message)
                    if control.get('kind')=='finish':incoming.finish();break
                    if control.get('kind')=='playback_receipt':
                        if len(json.dumps(control))>2048:raise ValueError('Oversized playback receipt')
                        # Receipts never feed the model or become acceptance labels.
                        incoming.receipts.append(dict(client_playback_receipt=control,received_monotonic=time.monotonic()))
                    else:raise ValueError('Unknown live control message')
            incoming.finish()
            await asyncio.wrap_future(future)
            await sender
        except Exception as error:
            if output is not None:output.event(dict(kind='session_error',message=type(error).__name__+': '+str(error)))
        finally:
            if incoming is not None:incoming.finish()
            if output is not None:output.disconnected.set()
            if sender is not None:sender.cancel()
            if future is not None:
                try:await asyncio.wrap_future(future)
                except Exception:pass
            if claimed:self.busy=False
    async def serve(self):
        import websockets
        if websockets.__version__!='15.0.1':raise RuntimeError('Live trial requires existing websockets15.0.1')
        from websockets.asyncio.server import serve
        origins=[None,re.compile(r'http://(?:localhost|127\.0\.0\.1)(?::[0-9]+)?')]
        async with serve(self.handler,self.host,self.port,process_request=self.http,origins=origins,max_size=8192,max_queue=16,compression=None) as bound:
            self.port=bound.sockets[0].getsockname()[1]
            self.ready.set()
            while not self.stop.is_set():await asyncio.sleep(.2)
    def run(self):
        try:asyncio.run(self.serve())
        except BaseException as error:self.error=error;self.ready.set()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('stt-assets','tts-assets','mimi-load-approval','controller-assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8788);ap.add_argument('--idle-seconds',type=int,default=300);ap.add_argument('--max-sessions',type=int,default=1)
    args=ap.parse_args()
    if not 1<=args.idle_seconds<=14400 or not 1<=args.max_sessions<=20:raise ValueError('Invalid bounded live service lifetime')
    args.out.mkdir(parents=True,exist_ok=False);token=secrets.token_urlsafe(32)
    token_path=args.out/'session-token'
    with os.fdopen(os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as handle:handle.write(token+'\n')
    from evals.kyutai_stt_probe import StreamingListener
    from evals.duplex_speaker import StreamingSpeaker
    from evals.qwen35_rpc import ResidentQwen35
    from evals.duplex_hybrid_probe import run_case
    from evals.moshi_model_download import sha256
    listener=StreamingListener(args.stt_assets,args.mimi_load_approval)
    controller=ResidentQwen35(args.controller_assets,args.out,device='cuda:1',pending_speech_protocol='explicit-actions',kernel_assets=args.kernel_assets)
    requests=queue.Queue(maxsize=1);stop=threading.Event();network=Network(args.host,args.port,token,requests,stop);thread=None;speaker=None
    try:
        with listener.session():
            listener.torch.manual_seed(42424242);listener.torch.cuda.manual_seed_all(42424242)
            speaker=StreamingSpeaker(args.tts_assets,args.mimi_load_approval);speaker.open();warmup=speaker.warmup()
            import numpy as np
            for _ in range(20):listener.step(np.zeros(1920,dtype=np.float32),verify_input=False)
            thread=threading.Thread(target=network.run,name='live-pcm-network',daemon=True);thread.start()
            if not network.ready.wait(10) or network.error:raise RuntimeError('Live network did not start') from network.error
            ready=dict(schema='dl-live-transport-ready-v1',host=args.host,port=network.port,controller_transport=controller.transport_info,controller_load=controller.proof,
                listener_loads=listener.loads,speaker_evidence=speaker.evidence,warmup=warmup,external_reasoner='gpt-6-astra',
                microphone_tested=False,physical_aec_tested=False,duplex_qualified=False,token_file=str(token_path),code_sha256=sha256(Path(__file__)))
            (args.out/'ready.json').write_text(json.dumps(ready,indent=2)+'\n');print(json.dumps(dict(ready=True,port=network.port,token_file=str(token_path))),flush=True)
            completed=[];failures=[]
            for number in range(args.max_sessions):
                try:request=requests.get(timeout=args.idle_seconds)
                except queue.Empty:break
                folder=args.out/('live-session-'+str(number));source=dict(split='live-transport',scenario_id='live-session-'+str(number),world={})
                try:
                    report=run_case(listener,speaker,source,folder,0.,controller_protocol='state-aware',controller=controller,
                        worker_kind='reasoning-development',live_input=request['incoming'],output_observer=request['output'])
                    report.update(capture_settings=request['capture_settings'],client_kind=request['client_kind'],microphone_acceptance=False,physical_aec_tested=False,
                        browser_playout_acceptance=False,server_model_wav_is_not_a_device_recording=True)
                    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                    completed.append(report)
                    suite=dict(schema='dl-concurrent-hybrid-development-v1',component_execution_completed=not failures,results=completed,failures=failures,
                        worker_kind='reasoning-development',live_transport=True,input_profile='live-transport',fresh_holdout=False,
                        speech_services_same_process_same_gpu=True,same_process_same_gpu=False,controller_provider=controller.provider_name,
                        controller_model=controller.model_id,local_controller_proof=controller.proof,controller_transport=controller.transport_info,
                        external_controller=None,external_reasoner='gpt-6-astra',remote_api_used=any(c['actual_tool_receipts'] for c in completed),
                        listener_loads=listener.loads,speaker_evidence=speaker.evidence,hybrid_system_qualified=False,native_model_qualified=False,
                        microphone_acceptance=False,physical_aec_tested=False,server_model_wav_is_not_a_device_recording=True,optimizer_steps=0,
                        code_sha256=sha256(Path(__file__)),core_code_sha256=sha256(Path(__file__).parents[2]/'evals/duplex_hybrid_probe.py'))
                    (args.out/'report.json').write_text(json.dumps(suite,indent=2)+'\n')
                    request['output'].event(dict(kind='session_complete',scenario=source['scenario_id']))
                    request['output'].put(None);request['future'].set_result(report)
                except BaseException as error:
                    failure=dict(error_type=type(error).__name__,message=str(error),controller_evidence=getattr(error,'result',None),duplex_qualified=False)
                    folder.mkdir(exist_ok=True);(folder/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                    if not request['future'].done():request['future'].set_exception(error)
                    try:
                        request['output'].event(dict(kind='session_error',message=str(error)));request['output'].put(None)
                    except Exception:request['output'].disconnected.set()
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=5)
        if speaker is not None:speaker.close()
        controller.close()


if __name__=='__main__':main()
