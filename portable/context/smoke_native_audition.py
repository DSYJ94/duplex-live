"""Exercise localhost native voice with one hash-pinned synthetic TRAIN user stem.

Records actual sampled output; no teacher words/audio are sent. This transport
check is not behavioral qualification and never accesses the owner microphone.
"""
import argparse
import array
import asyncio
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import wave

INPUTS = {
    'native2-train-008': '342a0ee096f1c3ce47f45950af7d8f9a36ead7195b3ef21cf1ca9a3b8e41d90b',
    'native2-train-014': '570b6c0710f17e2a9ad49f516fd9d2807046a3de7789f1709ae908f2dec37891',
}


async def run(args):
    import aiohttp
    diagnostic = None
    if args.diagnostic_manifest:
        manifest = json.loads(args.diagnostic_manifest.read_text())
        assert manifest['schema'] == 'dl-native-local-synthetic-cue-diagnostic-v1'
        assert manifest['owner_microphone_used'] is False
        assert manifest['speech_provider'] == 'installed-macos-say'
        diagnostic = next(c for c in manifest['cases'] if c['id'] == args.case)
        assert args.wav.resolve() == (args.diagnostic_manifest.parent/diagnostic['wav']).resolve()
        input_sha = diagnostic['sha256']
    else:
        input_sha = INPUTS[args.case]
    assert hashlib.sha256(args.wav.read_bytes()).hexdigest() == input_sha
    assert not args.out.exists(), 'Use a new receipt directory'
    with wave.open(str(args.wav)) as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 24000)
        pcm = array.array('h', source.readframes(source.getnframes()))
    if sys.byteorder != 'little':
        pcm.byteswap()
    frames = []
    for i in range(0, len(pcm), 1920):
        values = [x / 32768. for x in pcm[i:i+1920]]
        values += [0.] * (1920 - len(values))
        frames.append(struct.pack('<1920f', *values))
    args.out.mkdir(parents=True)
    receipt = {'schema': 'dl-native-audition-synthetic-transport-v1',
               'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'input_sha256': input_sha, 'input_scene': args.case,
               'only_user_stem_sent': True, 'teacher_audio_or_words_sent': False,
               'owner_microphone_accessed': False, 'behavior_qualified': False,
               'input_frames': len(frames), 'sent_frames': 0, 'events': []}
    if diagnostic:
        receipt['synthetic_diagnostic_plan'] = diagnostic
        receipt['training_data_or_heldout_approval_claimed'] = False
    audio = bytearray()
    started = time.monotonic()
    first_audio = None
    error = None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as client:
            async with client.get('http://127.0.0.1:8765/health') as response:
                response.raise_for_status()
                receipt['health'] = await response.json()
            async with client.ws_connect('http://127.0.0.1:8765/ws',
                    origin='http://127.0.0.1:8765', max_msg_size=2**24) as ws:
                ready = await asyncio.wait_for(ws.receive_json(), 60)
                assert ready['type'] == 'ready', ready
                assert (ready['sample_rate'], ready['frame_size']) == (24000, 1920)
                receipt['ready'] = ready

                async def send():
                    origin = time.monotonic()
                    for i, frame in enumerate(frames):
                        await asyncio.sleep(max(0., origin+i*.08-time.monotonic()))
                        await ws.send_bytes(frame)
                        receipt['sent_frames'] += 1

                sender = asyncio.create_task(send())
                try:
                    async with asyncio.timeout(120):
                        while len(audio) < len(frames)*1920*4:
                            # The native codec may initially return no output.
                            # Drain after every input was sent; record exact counts
                            # rather than pretending one output per input is owed.
                            try:
                                message = await ws.receive(timeout=5 if sender.done() else 60)
                            except asyncio.TimeoutError:
                                if sender.done():
                                    break
                                raise
                            if message.type == aiohttp.WSMsgType.BINARY:
                                assert len(message.data) % 4 == 0
                                if first_audio is None:
                                    first_audio = time.monotonic()-started
                                audio.extend(message.data)
                            elif message.type == aiohttp.WSMsgType.TEXT:
                                event = json.loads(message.data)
                                receipt['events'].append({**event,
                                    'output_frame': len(audio)//(1920*4),
                                    'received_seconds': time.monotonic()-started})
                                if event.get('type') in ('error', 'busy'):
                                    raise RuntimeError(str(event))
                            else:
                                raise RuntimeError('WebSocket closed before all audio frames returned')
                    await sender
                finally:
                    if not sender.done():
                        sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    values = struct.unpack('<'+'f'*(len(audio)//4), audio)
    finite = all(math.isfinite(v) for v in values)
    peak = max((abs(v) for v in values), default=0.) if finite else None
    rms = math.sqrt(sum(v*v for v in values)/len(values)) if finite and values else 0.
    # IEEE float WAV preserves the model samples, including any clipping.
    header = b'RIFF'+struct.pack('<I',36+len(audio))+b'WAVEfmt '+struct.pack(
        '<IHHIIHH',16,3,1,24000,96000,4,32)+b'data'+struct.pack('<I',len(audio))
    (args.out/'sampled-agent.wav').write_bytes(header+audio)
    receipt.update(error=error, elapsed_seconds=time.monotonic()-started,
                   first_audio_seconds=first_audio, output_samples=len(values),
                   output_frames=len(values)/1920,
                   output_finite=finite, output_peak=peak, output_rms=rms,
                   output_sha256=hashlib.sha256(header+audio).hexdigest(),
                   sampled_text=''.join(e.get('delta','') for e in receipt['events'] if e.get('type')=='text'))
    receipt['transport_passed'] = (error is None and receipt['sent_frames']==len(frames)
        and len(audio)>0 and finite and rms>1e-5)
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    return receipt['transport_passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wav', type=Path, required=True)
    p.add_argument('--case', default='native2-train-008')
    p.add_argument('--diagnostic-manifest', type=Path)
    p.add_argument('--out', type=Path, required=True)
    raise SystemExit(0 if asyncio.run(run(p.parse_args())) else 1)
