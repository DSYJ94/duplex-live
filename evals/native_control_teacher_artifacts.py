"""Revalidate all40 raw teachers, input causality and every returned PCM byte.

Transport integrity is separate from teacher label quality and native capability.
No approval flag can replace reconstruction from the original stream artifacts.
"""
import hashlib
import json
import math
from pathlib import Path

from data.synthetic.native_control_render import sha256, validate_raw
from evals.native_control_source_quality import verify as verify_source
from evals.native_control_teacher import ARTIFACTS, MODEL, PROTOCOL, configuration


def input_availability(events, samples, stream=None):
    rows = [e for e in events if e['type'] == 'local.input.append']
    cursor, previous = 0, 0.
    for row in rows:
        count, timestamp = row['sample_count'], row['received_at_s']
        if (type(count) is not int or row['first_sample'] != cursor
                or count != min(1920, samples - cursor) or count <= 0
                or type(timestamp) not in (int, float) or not math.isfinite(timestamp)
                or timestamp < previous or timestamp + 1e-6 < (cursor + count) / 24000):
            raise ValueError('Incomplete, reordered or unpaced actual input packet trace')
        if stream is not None:
            chunk = stream[cursor * 2:(cursor + count) * 2]
            if len(chunk) != count * 2 or hashlib.sha256(chunk).hexdigest() != row['pcm_sha256']:
                raise ValueError('Actually sent packet differs from the sealed input PCM')
        cursor += count
        previous = timestamp
    if cursor != samples or not rows:
        raise ValueError('Actual input packet trace does not cover the complete stream')
    return rows


def available_at(rows, sample):
    if sample <= 0:
        return 0.
    row = next((r for r in rows if r['first_sample'] < sample <= r['first_sample'] + r['sample_count']), None)
    if row is None:
        raise ValueError('Measured user completion is not covered by the actual sent input')
    return row['received_at_s']


def verify_record(raw, row, folder, source_audit_sha, record):
    import numpy as np
    import soundfile as sf
    from evals.audio_playout import render_packets
    if (record['id'] != row['id'] or record['dir'] != row['id']
            or set(record['artifacts']) != set(ARTIFACTS)
            or record['training_approved'] is not False or record['transport_complete'] is not True):
        raise ValueError('Missing complete unapproved raw teacher record')
    for name, digest in record['artifacts'].items():
        path = folder / name
        if path.is_symlink() or sha256(path) != digest:
            raise ValueError('Raw teacher artifact changed or is not self-contained')
    environment = json.loads((folder / 'environment.json').read_text())
    expected = dict(schema='dl-native-control3-teacher-environment-v1', purpose='teacher',
                    model=MODEL, protocol=PROTOCOL, source_id=row['id'],
                    source_raw_manifest_sha256=sha256(raw / 'raw-source.json'),
                    source_word_audit_sha256=source_audit_sha,
                    input_user_wav_sha256=sha256(raw / row['id'] / 'stem_user.wav'),
                    input_stream_sha256=sha256(folder / 'input-stream.pcm'),
                    session_config=configuration(), teacher_reference_conditioning='authored-control3-TRAIN-only',
                    training_approved=False, owner_recordings_used=False,
                    external_backend_enabled=False, output_suppression=False)
    if environment != expected:
        raise ValueError('Teacher model, instructions or source bindings changed')
    source, rate = sf.read(raw / row['id'] / 'stem_user.wav', dtype='float32')
    actual, actual_rate = sf.read(folder / 'user_in.wav', dtype='float32')
    wanted = np.concatenate([source, np.zeros(15 * 24000, dtype='float32')])
    if rate != 24000 or actual_rate != rate or not np.array_equal(actual, wanted):
        raise ValueError('Recorded user input differs from original PCM plus15 silent seconds')
    stream = (folder / 'input-stream.pcm').read_bytes()
    if stream != (wanted * 32767).astype('<i2').tobytes():
        raise ValueError('Actually sent PCM16 differs from the original waveform quantization')
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    sent = input_availability(events, len(wanted), stream)
    started = [e for e in events if e['type'] == 'session.started']
    closed = [e for e in events if e['type'] == 'session.closed']
    completed = [e for e in events if e['type'] == 'local.input.completed']
    report = json.loads((folder / 'report.json').read_text())
    if (len(started) != 1 or started[0]['session']['model'] != MODEL or len(closed) != 1
            or len(completed) != 1 or not report['session_finalized'] or report['errors']
            or report['source_samples'] != row['frames'] or report['added_silence_samples'] != 15 * 24000
            or report['sent_input_samples'] != len(wanted)
            or report['observed_until_s'] != closed[0]['received_at_s']
            or report['observed_until_s'] < completed[0]['received_at_s']
            or completed[0]['received_at_s'] < sent[-1]['received_at_s']
            or completed[0]['source_samples'] != row['frames']
            or completed[0]['sent_input_samples'] != len(wanted)
            or report['training_approved'] is not False or report['behavior_qualified'] is not False):
        raise ValueError('Incomplete actual session, input or observation evidence')
    if any(e['type'] == 'error' for e in events):
        raise ValueError('Teacher transport logged an API error')
    received = [e for e in events if e['type'] == 'session.output_audio.delta']
    packets = json.loads((folder / 'audio-packets.json').read_text())
    output = (folder / 'output-stream.pcm').read_bytes()
    if not packets or len(packets) != len(received) or report['received_audio_packets'] != len(packets):
        raise ValueError('Incomplete raw output packet inventory')
    chunks, cursor = [], 0
    for packet, event in zip(packets, received):
        if (type(packet['bytes']) is not int or packet['bytes'] <= 0 or packet['bytes'] % 2
                or packet['byte_offset'] != cursor or packet['arrival_s'] != event['received_at_s']
                or packet['bytes'] != event['audio_bytes'] or cursor != event['output_byte_offset']
                or packet['arrival_s'] > report['observed_until_s']):
            raise ValueError('Output packet clock or coverage changed')
        chunk = output[cursor:cursor + packet['bytes']]
        digest = hashlib.sha256(chunk).hexdigest()
        if len(chunk) != packet['bytes'] or digest != packet['pcm_sha256'] or digest != event['pcm_sha256']:
            raise ValueError('Raw output PCM changed')
        chunks.append((packet['arrival_s'], np.frombuffer(chunk, dtype='<i2')))
        cursor += packet['bytes']
    if cursor != len(output):
        raise ValueError('Raw output has unaccounted PCM bytes')
    played, _ = render_packets(chunks, 24000, 0)
    waveform, rate = sf.read(folder / 'model_out.wav', dtype='int16')
    if rate != 24000 or not np.array_equal(waveform, played):
        raise ValueError('Teacher WAV is not the exact unbuffered original packet playout')
    expected_transcripts = [dict(speaker='user' if 'input_' in e['type'] else 'assistant',
                                 text=e['delta'], start_ms=e.get('start_ms'), end_ms=e.get('end_ms'))
                            for e in events if e['type'] in ('session.input_transcript.delta', 'session.output_transcript.delta')]
    if json.loads((folder / 'transcripts.json').read_text()) != expected_transcripts:
        raise ValueError('Teacher transcripts differ from the original received deltas')
    return dict(id=row['id'], transport_integrity_passed=True, training_approved=False,
                behavior_or_label_quality_approved=False)


def validate_collection(raw, source_audit, batch):
    verify_source(raw, source_audit)
    source, plan = validate_raw(raw)
    manifest = json.loads((batch / 'teacher.json').read_text())
    if (manifest['schema'] != 'dl-native-control3-teacher-collection-v1'
            or manifest['model'] != MODEL or manifest['protocol'] != PROTOCOL
            or manifest['source_raw_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or manifest['source_word_audit_sha256'] != sha256(source_audit)
            or manifest['source_plan_sha256'] != source['plan_sha256']
            or manifest['code_sha256'] != sha256(Path(__file__).with_name('native_control_teacher.py'))
            or manifest['selected_ids'] != [s['id'] for s in plan['cases']]
            or manifest['maximum_sessions'] != 40 or manifest['concurrent_sessions'] != 2
            or manifest['status'] != 'raw_complete_unapproved' or manifest['training_approved'] is not False
            or [r['id'] for r in manifest['recordings']] != manifest['selected_ids']):
        raise ValueError('Complete40 immutable raw teachers required, never a passing subset')
    for row, record in zip(source['cases'], manifest['recordings']):
        verify_record(raw, row, batch / row['id'], manifest['source_word_audit_sha256'], record)
    return manifest
