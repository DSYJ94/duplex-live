"""Unattended real-transport trial; no physical microphone or device playback.

Sends independently approved user PCM at real frame arrival times. Saves every
received model packet and checks exact server/client PCM identity afterwards.
Reference text is accessed only after network generation, for ASR review.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import struct
import time


async def trial(args):
    import numpy as np
    import soundfile as sf
    import websockets
    if websockets.__version__!='15.0.1':raise RuntimeError('Expected existing websockets15.0.1')
    deadline=time.monotonic()+240
    while not (args.server/'ready.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Live server did not become ready')
        await asyncio.sleep(.2)
    ready=json.loads((args.server/'ready.json').read_text());token=(args.server/'session-token').read_text().strip()
    from evals.duplex_input_validator import cases
    rows,validator=cases(args.data,args.quality,[args.scenario],args.validator_source,args.validator_commit)
    if len(rows)!=1:raise RuntimeError('Expected one independently approved user stream')
    row=rows[0];pcm,rate=sf.read(row['path'],dtype='float32',always_2d=True)
    if rate!=24000 or pcm.shape[1]!=1:raise RuntimeError('Loopback requires real user-only24k PCM')
    source=pcm[:,0];padded=np.pad(source,(0,(-len(source))%1920)).astype('<f4');received=[];events=[];sends=[];event=asyncio.Event()
    args.out.mkdir(parents=True,exist_ok=False)
    from websockets.asyncio.client import connect
    async with connect('ws://127.0.0.1:'+str(ready['port'])+'/ws',max_size=16384,compression=None,proxy=None) as ws:
        await ws.send(json.dumps(dict(token=token,sample_rate=24000,frame_samples=1920,client_kind='paced-recorded-pcm-network-test',
            capture_settings=dict(physical_microphone=False,physical_device_playback=False,echo_cancellation_applied=False))))
        async def transmit():
            await asyncio.wait_for(event.wait(),timeout=30);start=time.monotonic()
            for sequence,first in enumerate(range(0,len(padded),1920)):
                await asyncio.sleep(max(0.,start+(sequence+1)*.08-time.monotonic()))
                await ws.send(struct.pack('<I',sequence)+padded[first:first+1920].tobytes());sends.append(dict(sequence=sequence,sent_s=time.monotonic()-start))
            await ws.send(json.dumps(dict(kind='finish')))
        sending=asyncio.create_task(transmit())
        try:
            async for message in ws:
                if isinstance(message,bytes):
                    if len(message)!=8+1920*4:raise RuntimeError('Changed network output frame size')
                    identity,sequence=struct.unpack_from('<II',message)
                    if sequence!=len(received):raise RuntimeError('Network output packet dropped or reordered')
                    received.append(dict(event_id='speech-'+str(identity),pcm=np.frombuffer(message,dtype='<f4',offset=8).copy(),received_monotonic=time.monotonic()))
                    with (args.out/'received-wire-packets.bin').open('ab') as handle:handle.write(message)
                else:
                    item=json.loads(message);events.append(item)
                    with (args.out/'network-events.jsonl').open('a') as handle:handle.write(json.dumps(item)+'\n')
                    if item['kind']=='session_ready':event.set()
                    if item['kind']=='session_error':raise RuntimeError('Live model session failed: '+item['message'])
                    if item['kind']=='session_complete':break
        finally:
            if not sending.done():sending.cancel()
        await sending
    if not received:raise RuntimeError('No actual model PCM arrived across the transport')
    suite=json.loads((args.server/'report.json').read_text());case=suite['results'][0];folder=args.server/case['scenario']
    actual,rate=sf.read(folder/'input-user.wav',dtype='float32');np.testing.assert_array_equal(actual,padded)
    ledger=json.loads((folder/'packets.json').read_text());saved=np.load(folder/'generated-packets.npz')
    if len(received)!=len(ledger) or set(saved.files)!={str(i) for i in range(len(ledger))}:raise RuntimeError('Network/server packet inventory differs')
    for i,(packet,entry) in enumerate(zip(received,ledger)):
        if packet['event_id']!=entry['event_id']:raise RuntimeError('Network/server speech identity differs')
        np.testing.assert_array_equal(packet['pcm'],saved[str(i)])
    np.savez(args.out/'received-packets.npz',**{str(i):p['pcm'] for i,p in enumerate(received)})
    (args.out/'network-events.json').write_text(json.dumps(events,indent=2)+'\n');(args.out/'input-arrivals.json').write_text(json.dumps(sends,indent=2)+'\n')
    from data.synthetic.filter import norm_words,wer
    reference=json.loads(Path(row['reference_path']).read_text());text=' '.join(w['text'] for w in reference['user_words'])
    report=dict(schema='dl-actual-live-transport-loopback-v1',component_execution_completed=True,input_validator=validator,source_scenario=args.scenario,
        source_wav_sha256=row['wav_sha256'],actual_input_pcm_exact=True,all_received_model_packets_exact=True,received_packets=len(received),
        received_pcm_sha256=hashlib.sha256(b''.join(p['pcm'].tobytes() for p in received)).hexdigest(),
        recognized_text=case['recognized_text'],reference_read_after_generation=True,recognition_wer=wer(norm_words(text),norm_words(case['recognized_text'])),
        controller_provider=suite['controller_provider'],external_reasoner=suite['external_reasoner'],remote_api_used=suite['remote_api_used'],
        physical_microphone=False,physical_device_playback=False,physical_aec_tested=False,duplex_qualified=False,optimizer_steps=0)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('server','data','quality','validator-source','out'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--validator-commit',required=True);ap.add_argument('--scenario',default='hybrid-fresh-024');args=ap.parse_args();asyncio.run(trial(args))


if __name__=='__main__':main()
