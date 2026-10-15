"""Concurrent native-token to PCM capture on a dedicated GPU.

PCM availability is measured during actual LM execution. The final WAV simulates
bounded causal playout; no audio device, browser or microphone is exercised.
"""
import argparse
import json
import queue
import socket
import threading
import time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','voice','voice-provenance','socket','ready','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--hop',type=int,choices=(8,25),default=8)
    args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from evals.lychee_live_assets import load
    from evals.utterance_playout import render_utterances
    from evals.moshi_model_download import sha256
    args.out.mkdir(exist_ok=False)
    vocoder,decoder,provenance=load(args.assets,args.voice,args.voice_provenance)
    lookahead=int(vocoder.flow.pre_lookahead_len)
    with torch.inference_mode():
        vocoder.set_stream_cache(str(args.voice))
        vocoder.stream([0]*(args.hop+lookahead+1),str(args.voice),last_chunk=False)
        vocoder.stream([0],str(args.voice),last_chunk=True)
    torch.cuda.synchronize();torch.manual_seed(424242)
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(str(args.socket));server.listen(1);server.settimeout(180)
    args.ready.write_text('warmed streaming vocoder listening\n')
    actions=queue.Queue(maxsize=512)
    state={'epoch':None,'stops':{},'error':None,'complete':False,'max_queue':0,
        'transport_delays_s':[],'native_events_received':0}
    reader_done=threading.Event()
    def receive():
        try:
            connection,_=server.accept()
            connection.settimeout(180)
            with connection,connection.makefile('rb') as stream,(args.out/'transport-receipts.jsonl').open('w') as log:
                while True:
                    line=stream.readline(65537)
                    if not line:break
                    if len(line)>65536 or not line.endswith(b'\n'):
                        raise RuntimeError('Unbounded or incomplete transport message')
                    message=json.loads(line)
                    if message['type']=='clock_start':
                        if state['epoch'] is not None:raise RuntimeError('Duplicate input epoch')
                        state['epoch']=message['monotonic_epoch'];continue
                    if state['epoch'] is None:raise RuntimeError('Native event before input clock')
                    arrived=time.monotonic()-state['epoch']
                    if message['type']=='input_complete':
                        raw={'type':'generation_complete','reason':'actual input/session exhaustion'}
                        state['complete']=True
                    elif message['type']=='native_event':
                        record=message['record'];raw=record['event']
                        delay=arrived-record['available_s']
                        if delay<-.001:raise RuntimeError('Processes do not share the monotonic input clock')
                        state['transport_delays_s'].append(delay)
                        state['native_events_received']+=1
                    else:raise RuntimeError('Unknown concurrent transport message')
                    log.write(json.dumps({'received_s':arrived,'message':message})+'\n');log.flush()
                    for event in decoder.feed(raw):
                        # A control stop is received independently of a busy TTS
                        # GPU. Late old PCM is capped at this causal receipt time.
                        if event['type']=='event_end' and event.get('interrupt'):
                            state['stops'][event['event_id']]=arrived
                        if event['type'] in ('event_start','audio_chunk','event_end'):
                            actions.put_nowait((arrived,event))
                            state['max_queue']=max(state['max_queue'],actions.qsize())
                    if state['complete']:break
            if not state['complete']:raise RuntimeError('LM disconnected without explicit input completion')
        except BaseException as exc:
            state['error']=repr(exc)
        finally:
            reader_done.set()
    thread=threading.Thread(target=receive,daemon=True);thread.start()
    packets=[];rows=[];groups={};active=None;buffer=[];expected=0;consumed=0;cancelled=0
    def emit(tokens,last,advance,due):
        nonlocal consumed,cancelled
        if active in state['stops']:
            cancelled+=advance
            return
        began=time.monotonic()-state['epoch']
        with torch.inference_mode():blob=vocoder.stream(tokens,str(args.voice),last_chunk=last)
        torch.cuda.synchronize();available=time.monotonic()-state['epoch']
        pcm=np.frombuffer(blob,dtype='<i2').astype(np.float32)/32768.
        if not np.isfinite(pcm).all():raise RuntimeError('Nonfinite streaming PCM')
        consumed+=advance
        rows.append({'event_id':active,'tokens':len(tokens),'advance_tokens':advance,'last':last,
            'tokens_received_s':due,'started_s':began,'pcm_available_s':available,
            'compute_s':available-began,'samples':len(pcm)})
        if len(pcm):packets.append({'event_id':active,'available_s':available,'pcm':pcm})
    try:
        while not reader_done.is_set() or not actions.empty():
            if state['error']:raise RuntimeError('Concurrent reader failed: '+state['error'])
            try:due,event=actions.get(timeout=.2)
            except queue.Empty:continue
            kind=event['type']
            if kind=='event_start':
                if active is not None:raise RuntimeError('Overlapping native TTS events')
                active=event['event_id'];groups[active]=0;buffer=[]
                if active not in state['stops']:vocoder.set_stream_cache(str(args.voice))
            elif kind=='audio_chunk':
                if active is None:raise RuntimeError('Acoustic units outside a native event')
                values=event['stoken_ids']
                if any(not isinstance(t,int) or not 0<=t<6561 for t in values):
                    raise RuntimeError('Invalid native acoustic unit; never discard silently')
                groups[active]+=len(values)
                if active in state['stops']:
                    cancelled+=len(values);continue
                buffer.extend(values);window=args.hop+lookahead
                while len(buffer)>window:
                    emit(buffer[:window],False,args.hop,due);buffer=buffer[args.hop:]
            elif kind=='event_end':
                if active!=event['event_id']:raise RuntimeError('Native event identity lost')
                expected+=len(event['stoken_ids'])
                if groups[active]!=len(event['stoken_ids']):raise RuntimeError('Streaming decoder lost codec units')
                if event.get('interrupt') or active in state['stops']:
                    cancelled+=len(buffer)
                elif buffer:emit(buffer,True,len(buffer),due)
                buffer=[];active=None
        if state['error']:raise RuntimeError('Concurrent reader failed: '+state['error'])
        if not state['complete'] or active is not None or consumed+cancelled!=expected or not packets:
            raise RuntimeError('Incomplete concurrent token/PCM accounting')
    finally:
        server.close()
    output,playout=render_utterances(packets,24000,.24,state['stops'])
    sf.write(args.out/'model.wav',output,24000)
    sf.write(args.out/'packet-stream.wav',np.concatenate([p['pcm'] for p in packets]),24000)
    (args.out/'pcm-packets.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (args.out/'playout.json').write_text(json.dumps(playout,indent=2)+'\n')
    report={**provenance,'schema':'dl-lychee-concurrent-pcm-v1','hop_tokens':args.hop,
        'lookahead_tokens':lookahead,'actual_concurrent_lm_execution':True,
        'pcm_availability_measured':True,'playout_mode':'causal fixed-buffer simulation from actual concurrent availability',
        'microphone_acceptance':False,'audio_device_tested':False,'training_approved':False,
        'quality_qualified':False,'native_events_received':state['native_events_received'],
        'transport_delay_p95_s':float(np.percentile(state['transport_delays_s'],95)),
        'transport_delay_max_s':max(state['transport_delays_s']),
        'maximum_bounded_action_queue':state['max_queue'],'action_queue_capacity':512,
        'expected_codec_units':expected,'consumed_codec_units':consumed,'cancelled_unplayed_codec_units':cancelled,
        'received_native_interrupts':state['stops'],'vocoder_compute_p95_s':float(np.percentile([r['compute_s'] for r in rows],95)),
        'first_pcm_available_s':packets[0]['available_s'],'playout':playout,
        'audio_sha256':sha256(args.out/'model.wav')}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='playout'}),flush=True)


if __name__=='__main__':main()
