"""Rebuild WebRTC capture evidence without granting teacher or training approval."""
import hashlib
import importlib.metadata
import json
from fractions import Fraction
from pathlib import Path

from data.synthetic.native_control_render import sha256
from evals.native_control_teacher_artifacts import input_availability
from evals.native_control_webrtc_capture import PROTOCOL, SID, configuration, configuration_matches
from evals.native_control_webrtc_probe import mono_plane

FILES = {'audio-packets.json', 'received-media.pcm', 'report.json', 'transcripts.json',
         'server-input.pcm', 'model_out.wav', 'output-stream.pcm', 'input-stream.pcm',
         'user_in.wav', 'playout.json', 'events.jsonl', 'environment.json', 'server-input.wav'}


def spool(rows, data, offset_name, size_name, hash_name):
    cursor = 0
    for row in rows:
        count = row[size_name]
        if type(count) is not int or count <= 0 or row[offset_name] != cursor:
            raise ValueError('Incomplete sequential media spool')
        block = data[cursor:cursor+count]
        if len(block) != count or hashlib.sha256(block).hexdigest() != row[hash_name]:
            raise ValueError('Changed media spool bytes')
        cursor += count
    if not rows or cursor != len(data):
        raise ValueError('Unaccounted media spool bytes')


def verify(raw, source_row, source_audit, folder):
    import numpy as np
    import soundfile as sf
    from av import AudioFrame, AudioResampler
    from evals.audio_playout import render_packets
    artifacts = json.loads((folder / 'artifacts.json').read_text())
    if set(artifacts) != FILES:
        raise ValueError('Complete original WebRTC capture inventory required')
    for name, digest in artifacts.items():
        if (folder / name).is_symlink() or sha256(folder / name) != digest:
            raise ValueError('Original WebRTC artifact changed')
    env = json.loads((folder / 'environment.json').read_text())
    session = json.loads((folder / 'report.json').read_text())
    if (source_row['id'] != SID or env['schema'] != 'dl-native-control3-webrtc-environment-v2'
            or env['protocol'] != PROTOCOL or env['source_id'] != SID or env['model'] != 'gpt-live-1'
            or env['session_config'] != configuration()
            or env['source_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or env['source_audit_sha256'] != sha256(source_audit)
            or env['source_user_wav_sha256'] != sha256(raw / SID / 'stem_user.wav')
            or env['code_sha256'] != sha256(Path('evals/native_control_webrtc_capture.py'))
            or env['upstream_probe_code_sha256'] != sha256(Path('evals/native_control_webrtc_probe.py'))
            or env['packages'] != {p:importlib.metadata.version(p) for p in ('aiortc','av','aioice','pylibsrtp')}
            or env['training_approved'] is not False or env['diagnostic_only'] is not True
            or env['owner_recordings_used'] is not False or env['microphone_or_speaker_accessed'] is not False
            or env['external_backend_enabled'] is not False or env['server_reflected_input_preserved'] is not True
            or session['status'] != 'complete_diagnostic_unapproved' or session['errors']
            or session['session_finalized'] is not True or session['sideband_session_finalized'] is not True
            or session['session_config_echo_matches'] is not True or session['preready_samples'] != 0
            or session['source_samples'] != source_row['frames'] or session['added_silence_samples'] != 360000
            or session['training_approved'] is not False):
        raise ValueError('Unchanged complete same-source WebRTC capture required')
    source, rate = sf.read(raw / SID / 'stem_user.wav', dtype='float32')
    actual, actual_rate = sf.read(folder / 'user_in.wav', dtype='float32')
    wanted = np.concatenate([source, np.zeros(360000, dtype='float32')])
    stream = (folder / 'input-stream.pcm').read_bytes()
    if (rate != 24000 or actual_rate != rate or not np.array_equal(actual,wanted)
            or stream != (wanted*32767).astype('<i2').tobytes()
            or session['sent_input_samples'] != len(wanted)):
        raise ValueError('Encoder source differs from all original source PCM plus15 seconds')
    events = [json.loads(x) for x in (folder / 'events.jsonl').read_text().splitlines()]
    input_availability(events, len(wanted), stream)
    starts = [i for i,e in enumerate(events) if e['type']=='session.started']
    attaches = [i for i,e in enumerate(events) if e['type']=='local.webrtc.sideband.attached']
    closed = [e for e in events if e['type']=='session.closed']
    completed = [e for e in events if e['type']=='local.input.completed']
    drains = [e for e in events if e['type']=='local.webrtc.drain.completed']
    if (len(starts)!=1 or len(attaches)!=1 or attaches[0]>=starts[0] or len(closed)!=1
            or len(completed)!=1 or len(drains)!=1
            or not configuration_matches(events[starts[0]]['session'],configuration())
            or session['server_closed_at_s']!=closed[0]['received_at_s']
            or not completed[0]['received_at_s']<=closed[0]['received_at_s']<=session['observed_until_s']
            or session['observed_until_s']<drains[0]['received_at_s']-.001
            or any(e['type']=='error' or e['type'].startswith('session.delegation.') for e in events)):
        raise ValueError('Sideband, input, actual close or observed playout trace incomplete')
    server = (folder / 'server-input.pcm').read_bytes()
    rows = [e for e in events if e['type']=='local.webrtc.sideband.input']
    spool(rows,server,'byte_offset','bytes','pcm_sha256')
    server_wave,sr = sf.read(folder / 'server-input.wav',dtype='int16')
    if (sr!=24000 or not np.array_equal(server_wave,np.frombuffer(server,dtype='<i2'))
            or len(server_wave)!=session['server_reflected_input_samples'] or len(server_wave)<len(wanted)
            or len(server_wave)/sr>session['observed_until_s']+.08):
        raise ValueError('Complete original reflected input PCM is not preserved')
    media = (folder / 'received-media.pcm').read_bytes()
    decoded = [e for e in events if e['type']=='local.webrtc.decoded.media']
    spool(decoded,media,'byte_offset','bytes','sha256')
    resampler = AudioResampler(format='s16',layout='mono',rate=24000)
    rebuilt = []
    for row in decoded:
        data = media[row['byte_offset']:row['byte_offset']+row['bytes']]
        fmt,channels = row['format'],row['channels']
        if fmt not in ('s16','s16p','flt','fltp') or channels not in (1,2):
            raise ValueError('Unsupported preserved decoded format')
        values = np.frombuffer(data,dtype=np.int16 if fmt.startswith('s16') else np.float32)
        values = values.reshape(channels,-1) if fmt.endswith('p') else values.reshape(1,-1)
        frame = AudioFrame.from_ndarray(values.copy(),format=fmt,layout='mono' if channels==1 else 'stereo')
        frame.sample_rate,frame.pts,frame.time_base = row['rate'],row['pts'],Fraction(row['time_base'])
        plane = mono_plane(frame)
        mono = AudioFrame.from_ndarray(plane,format='s16' if plane.dtype==np.int16 else 'flt',layout='mono')
        mono.sample_rate,mono.pts,mono.time_base = frame.sample_rate,frame.pts,frame.time_base
        rebuilt.extend(f.to_ndarray().astype('<i2',copy=False).tobytes() for f in resampler.resample(mono))
    rebuilt.extend(f.to_ndarray().astype('<i2',copy=False).tobytes() for f in resampler.resample(None))
    output = (folder / 'output-stream.pcm').read_bytes()
    if b''.join(rebuilt)!=output:
        raise ValueError('Output PCM is not the exact duplicate-mono checked decoded-media conversion')
    packets = json.loads((folder / 'audio-packets.json').read_text())
    spool(packets,output,'byte_offset','bytes','pcm_sha256')
    received = [e for e in events if e['type']=='local.webrtc.output.frame']
    if len(received)!=len(packets) or any(p!=dict(arrival_s=e['received_at_s'],byte_offset=e['output_byte_offset'],
            bytes=e['audio_bytes'],pcm_sha256=e['pcm_sha256']) for p,e in zip(packets,received)):
        raise ValueError('Original packet arrival clocks changed')
    played,ledger = render_packets([(p['arrival_s'],np.frombuffer(output[p['byte_offset']:p['byte_offset']+p['bytes']],dtype='<i2'))
                                   for p in packets],24000,0)
    actual,sr = sf.read(folder / 'model_out.wav',dtype='int16')
    if (sr!=24000 or not np.array_equal(actual,played) or json.loads((folder / 'playout.json').read_text())!=ledger
            or len(played)/sr>session['observed_until_s']+.08
            or drains[0]['playout_end_sample']!=len(played) or session['output_samples']!=len(played)):
        raise ValueError('Client waveform differs from complete causal original playout')
    transcripts = [dict(speaker='user' if 'input_' in e['type'] else 'assistant',text=e['delta'],
                        start_ms=e.get('start_ms'),end_ms=e.get('end_ms')) for e in events
                   if e['type'] in ('session.input_transcript.delta','session.output_transcript.delta')]
    if json.loads((folder / 'transcripts.json').read_text())!=transcripts:
        raise ValueError('Original transcript deltas changed')
    return dict(id=SID,transport_integrity_passed=True,server_received_input_pcm_preserved=True,
                source_encoder_pcm_exact=True,wire_codec_lossy=True,independent_input_words_required=True,
                training_approved=False,behavior_or_label_quality_approved=False)
