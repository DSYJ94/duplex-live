"""One synthetic control006 WebRTC transport diagnostic; never training approval.

Keep every decoded media frame and original24k source. WebRTC negotiates Opus;
received media is separately retained before explicit mono/24k conversion. No
microphone, speaker, external backend, RTP loss concealment approval or silent retry.
SDP/ICE credentials and the API key are never archived.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

from data.synthetic.native_control_render import sha256, validate_raw
from evals.native_control_source_quality import verify as verify_source
from evals.native_control_teacher import atomic_json, collection_lock, local_key, safe_event
from evals.native_control_teacher_scoped import configuration as scoped_configuration
from evals.audio_playout import render_packets

SID = 'native-control3-train-006'
PROTOCOL = 'native-control3-webrtc-transport-probe-v1'


async def close_sideband(session_id, key, emit, *, connector=None):
    """Close only the session this probe created, using documented sideband control."""
    if connector is None:
        from websockets.asyncio.client import connect
        connector = connect
    from urllib.parse import quote
    async with asyncio.timeout(20):
        async with connector('wss://api.openai.com/v1/live/sessions/' + quote(session_id, safe='') + '/attach',
                             additional_headers={'Authorization': 'Bearer ' + key}, open_timeout=8,
                             close_timeout=2, max_size=8 * 1024 * 1024) as socket:
            await socket.send(json.dumps(dict(type='session.close')))
            async for payload in socket:
                event = json.loads(payload)
                # Sideband PCM is not the media-track recording; do not merge it.
                if event.get('type') in ('session.closed', 'error'):
                    emit(dict(type='local.webrtc.cleanup.event', event=redact(event)))
                if event.get('type') == 'session.closed':
                    return True
    return False


async def observe_playout(packets, origin, emit, *, now=time.monotonic, sleep=asyncio.sleep):
    """Measure real post-close observation; never backdate or trim packet audio."""
    cursor = 0
    for row in packets:
        cursor = max(cursor, math.ceil(row['arrival_s'] * 24000)) + row['bytes'] // 2
    started = now() - origin
    target = cursor / 24000 + .16
    emit(dict(type='local.webrtc.drain.started', playout_end_sample=cursor,
              maximum_drain_s=20, margin_s=.16))
    if not packets or target - started > 20:
        raise TimeoutError('Causal media-track playout exceeds bounded observation')
    while now() - origin < target:
        await sleep(min(.25, max(.001, target - (now() - origin))))
    ended = now() - origin
    if ended - started > 20.08:
        raise TimeoutError('Actual media-track drain exceeded bound')
    emit(dict(type='local.webrtc.drain.completed', playout_end_sample=cursor))
    return ended


def configuration():
    config = scoped_configuration()
    config['audio'].pop('format')  # Required: media format is SDP-negotiated.
    return config


def redact(event):
    value = safe_event(event)
    def clean(v):
        if isinstance(v, dict):
            return {k: '<REDACTED>' if k.casefold() in {'sdp', 'password', 'credential', 'ice_servers'} else clean(x)
                    for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v
    return clean(value)


def mono_plane(frame):
    """A stereo decoder may duplicate mono. Never silently mix distinct channels."""
    import numpy as np
    data = frame.to_ndarray()
    channels = len(frame.layout.channels)
    planar = data if frame.format.is_planar else data.reshape(-1, channels).T
    if channels > 1 and not np.all(planar == planar[:1]):
        raise ValueError('Decoded remote channels differ; no silent downmix allowed')
    return np.ascontiguousarray(planar[:1])


async def capture(raw, source_audit, out, key, lock):
    import numpy as np
    import soundfile as sf
    import requests
    from av import AudioFrame, AudioResampler
    from aiortc import (AudioStreamTrack, RTCPeerConnection, RTCConfiguration,
                        RTCSessionDescription, RTCRtpSender, MediaStreamError)
    verify_source(raw, source_audit)
    manifest, _ = validate_raw(raw)
    scene = next(r for r in manifest['cases'] if r['id'] == SID)
    with collection_lock(lock):
        out.mkdir(parents=True, exist_ok=False)
        source = raw / SID / 'stem_user.wav'
        audio, rate = sf.read(source, dtype='float32')
        if rate != 24000 or audio.ndim != 1 or len(audio) != scene['frames'] or not np.isfinite(audio).all():
            raise ValueError('Sealed source changed')
        audio = np.concatenate([audio, np.zeros(15 * rate, dtype='float32')])
        pcm = (audio * 32767).astype('<i2')
        sf.write(out / 'user_in.wav', audio, rate, subtype='FLOAT')
        (out / 'input-stream.pcm').write_bytes(pcm.tobytes())
        env = dict(schema='dl-native-control3-webrtc-environment-v1', protocol=PROTOCOL,
                   source_id=SID, model='gpt-live-1', session_config=configuration(),
                   source_manifest_sha256=sha256(raw / 'raw-source.json'), source_audit_sha256=sha256(source_audit),
                   source_user_wav_sha256=sha256(source), code_sha256=sha256(Path(__file__)),
                   source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   packages={p: importlib.metadata.version(p) for p in ('aiortc', 'av', 'aioice', 'pylibsrtp')},
                   source_pcm_provided_to_encoder_exact=True, wire_codec='SDP-negotiated Opus, lossy',
                   decoded_remote_media_preserved=True, output_conversion='Duplicate-mono channel check, PyAV s16/mono/24k resampler',
                   owner_recordings_used=False, microphone_or_speaker_accessed=False,
                   external_backend_enabled=False, diagnostic_only=True, training_approved=False)
        atomic_json(out / 'environment.json', env)
        ready, done, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        origin, cursor, group_cursor, errors, packets, receiver_tasks = None, 0, 0, [], [], []
        preready_samples, preready_nonzero = 0, 0
        log = (out / 'events.jsonl').open('x')
        spool = (out / 'output-stream.pcm').open('xb')
        media = (out / 'received-media.pcm').open('xb')
        state = dict(schema='dl-native-control3-webrtc-probe-v1', status='connecting',
                     protocol=PROTOCOL, started_utc=datetime.now(timezone.utc).isoformat(),
                     sent_input_samples=0, expected_input_samples=len(pcm), session_finalized=False,
                     diagnostic_only=True, training_approved=False, original_audio_changed=False)
        target = out / 'report.json'
        atomic_json(target, state)

        def emit(event):
            value = dict(redact(event), received_at_s=None if origin is None else time.monotonic() - origin)
            log.write(json.dumps(value) + '\n'); log.flush()
            return value

        class InputTrack(AudioStreamTrack):
            async def recv(self):
                nonlocal cursor, group_cursor
                await ready.wait()
                if cursor >= len(pcm):
                    await asyncio.Future()  # No extra source frames while finalizing.
                end = min(cursor + 480, len(pcm))
                await asyncio.sleep(max(0., origin + end / rate - time.monotonic()))
                samples = pcm[cursor:end]
                frame = AudioFrame.from_ndarray(samples[None], format='s16', layout='mono')
                frame.sample_rate, frame.pts, frame.time_base = rate, cursor, Fraction(1, rate)
                emit(dict(type='local.webrtc.input.frame', first_sample=cursor, sample_count=len(samples),
                          pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest()))
                cursor = end
                if cursor % 1920 == 0 or cursor == len(pcm):
                    first = group_cursor
                    emit(dict(type='local.input.append', first_sample=first, sample_count=cursor-first,
                              pcm_sha256=hashlib.sha256(pcm[first:cursor].tobytes()).hexdigest(),
                              availability='Submitted to negotiated Opus encoder; no server-receipt claim'))
                    group_cursor = cursor
                if cursor == len(pcm):
                    emit(dict(type='local.input.completed', source_samples=scene['frames'], sent_input_samples=cursor))
                    done.set()
                return frame

        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        sender = peer.addTrack(InputTrack())
        transceiver = next(t for t in peer.getTransceivers() if t.sender is sender)
        transceiver.setCodecPreferences([c for c in RTCRtpSender.getCapabilities('audio').codecs
                                        if c.mimeType.lower() == 'audio/opus'])
        channel = peer.createDataChannel('oai-events')

        @channel.on('message')
        def message(payload):
            nonlocal origin
            try:
                event = json.loads(payload)
            except (ValueError, TypeError):
                errors.append(dict(error_type='InvalidDataChannelJSON')); return
            if event['type'] == 'session.started':
                if origin is not None:
                    errors.append(dict(error_type='DuplicateSessionStarted')); return
                origin = time.monotonic(); ready.set()
            row = emit(event)
            if event['type'] == 'error':
                errors.append(dict(error_type='ProviderError', code=event.get('error', {}).get('code')))
            if event['type'].startswith('session.delegation.'):
                errors.append(dict(error_type='UnexpectedDelegation', event_type=event['type']))
            if event['type'] == 'session.closed':
                state['server_closed_at_s'] = row['received_at_s']; closed.set()

        async def receive(track):
            nonlocal preready_samples, preready_nonzero
            resampler = AudioResampler(format='s16', layout='mono', rate=24000)
            def save(frame):
                nonlocal preready_samples, preready_nonzero
                data = frame.to_ndarray().astype('<i2', copy=False).tobytes()
                if origin is None:
                    preready_samples += len(data)//2
                    preready_nonzero += int(np.count_nonzero(np.frombuffer(data, dtype='<i2')))
                    # Preserve in the raw decoded-media spool; exclude from the
                    # post-session.started conversation only as explicit preready evidence.
                    emit(dict(type='local.webrtc.preready.audio', samples=len(data)//2)); return
                offset = spool.tell(); spool.write(data); spool.flush()
                row = emit(dict(type='local.webrtc.output.frame', output_byte_offset=offset,
                                audio_bytes=len(data), pcm_sha256=hashlib.sha256(data).hexdigest(),
                                media_pts=frame.pts, media_time_base=str(frame.time_base)))
                packets.append(dict(arrival_s=row['received_at_s'], byte_offset=offset,
                                    bytes=len(data), pcm_sha256=row['pcm_sha256']))
            try:
                while True:
                    frame = await track.recv()
                    original = frame.to_ndarray().tobytes(); offset = media.tell(); media.write(original); media.flush()
                    emit(dict(type='local.webrtc.decoded.media', byte_offset=offset, bytes=len(original),
                              sha256=hashlib.sha256(original).hexdigest(), rate=frame.sample_rate,
                              channels=len(frame.layout.channels), format=frame.format.name, pts=frame.pts,
                              time_base=str(frame.time_base)))
                    plane = mono_plane(frame)
                    if plane.dtype not in (np.dtype('int16'), np.dtype('float32')):
                        raise ValueError('Unsupported decoded media sample format')
                    fmt = 's16' if plane.dtype == np.int16 else 'flt'
                    mono = AudioFrame.from_ndarray(plane, format=fmt, layout='mono')
                    mono.sample_rate, mono.pts, mono.time_base = frame.sample_rate, frame.pts, frame.time_base
                    for converted in resampler.resample(mono): save(converted)
            except MediaStreamError:
                for converted in resampler.resample(None): save(converted)
            except asyncio.CancelledError:
                for converted in resampler.resample(None): save(converted)
                raise
            except Exception as e:
                errors.append(dict(error_type=type(e).__name__, component='media_receiver'))

        @peer.on('track')
        def on_track(track):
            if track.kind != 'audio':
                errors.append(dict(error_type='UnexpectedRemoteTrack')); return
            receiver_tasks.append(asyncio.create_task(receive(track)))

        try:
            async with asyncio.timeout(len(pcm)/rate + 75):
                await peer.setLocalDescription(await peer.createOffer())
                body = dict(session=configuration(), transport=dict(type='webrtc', sdp=peer.localDescription.sdp))
                def create():
                    r = requests.post('https://api.openai.com/v1/live/sessions', json=body,
                                      headers={'Authorization': 'Bearer ' + key,
                                               'OpenAI-Safety-Identifier': hashlib.sha256(b'duplex-live-evaluation').hexdigest()},
                                      timeout=(10, 30))
                    state['create_http_status'] = r.status_code
                    if r.status_code != 201:
                        # Error messages may echo a request: retain only stable identifiers.
                        try: error = r.json().get('error', {})
                        except ValueError: error = {}
                        state['create_error'] = {k:error[k] for k in ('type', 'code', 'param') if k in error}
                        raise RuntimeError('LiveWebRTCCreateRejected')
                    return r.json()
                response = await asyncio.to_thread(create)
                state['created_session_id'] = response['session']['id']
                # SDP contains ICE credentials: retain only non-reversible hashes.
                state['offer_sha256'] = hashlib.sha256(peer.localDescription.sdp.encode()).hexdigest()
                state['answer_sha256'] = hashlib.sha256(response['transport']['sdp'].encode()).hexdigest()
                await peer.setRemoteDescription(RTCSessionDescription(sdp=response['transport']['sdp'], type='answer'))
                await asyncio.wait_for(ready.wait(), 30)
                state['status'] = 'streaming'; atomic_json(target, state)
                await done.wait(); await asyncio.sleep(2)
                channel.send(json.dumps(dict(type='session.close')))
                await asyncio.wait_for(closed.wait(), 15)
                state['session_finalized'] = True
                await asyncio.sleep(.3)
        except Exception as e:
            errors.append(dict(error_type=type(e).__name__))
        finally:
            if state.get('created_session_id') and not state['session_finalized']:
                try:
                    state['cleanup_session_finalized'] = await close_sideband(state['created_session_id'], key, emit)
                except Exception as e:
                    errors.append(dict(error_type=type(e).__name__, component='sideband_cleanup'))
            await peer.close()
            for task in receiver_tasks: task.cancel()
            await asyncio.gather(*receiver_tasks, return_exceptions=True)
            if origin is not None and packets:
                try:
                    state['observed_until_s'] = await observe_playout(packets, origin, emit)
                except Exception as e:
                    errors.append(dict(error_type=type(e).__name__, component='output_drain'))
            if preready_nonzero:
                errors.append(dict(error_type='NonzeroAudioBeforeSessionStarted'))
            log.close(); spool.close(); media.close()
        output = (out / 'output-stream.pcm').read_bytes()
        if packets:
            chunks = [(p['arrival_s'], np.frombuffer(output[p['byte_offset']:p['byte_offset']+p['bytes']], dtype='<i2'))
                      for p in packets]
            played, ledger = render_packets(chunks, 24000, 0)
            sf.write(out / 'model_out.wav', played, 24000, subtype='PCM_16')
            atomic_json(out / 'playout.json', ledger)
        else:
            played = np.zeros(1, dtype='int16'); sf.write(out / 'model_out.wav', played, 24000, subtype='PCM_16')
        state.update(status='complete_diagnostic_unapproved' if state['session_finalized'] and not errors else 'failed_preserve_artifacts',
                     sent_input_samples=cursor, output_samples=len(played), output_packets=len(packets),
                     preready_samples=preready_samples, preready_nonzero_samples=preready_nonzero, errors=errors,
                     observed_until_s=state.get('observed_until_s', None if origin is None else time.monotonic()-origin),
                     ended_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out / 'audio-packets.json', packets)
        atomic_json(target, state)
        atomic_json(out / 'artifacts.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'artifacts.json'})
        return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'source-audit', 'out', 'env-file'): ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--lock', type=Path, default=Path('runs/local-migration-20261021/.private/control3-capture.lock'))
    a = ap.parse_args()
    report = asyncio.run(capture(a.raw, a.source_audit, a.out, local_key(a.env_file), a.lock))
    print(json.dumps({k:report[k] for k in ('status','sent_input_samples','output_samples','errors','training_approved')}))
    if report['status'] != 'complete_diagnostic_unapproved': raise SystemExit(1)

if __name__ == '__main__': main()
