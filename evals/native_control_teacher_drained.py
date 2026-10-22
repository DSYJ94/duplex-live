"""Repair only the diagnosed capture backlog; preserve exact prompt and every PCM sample.

The v1/v2 collectors remain immutable. This v3 collector waits for causal playout
and a two-second output-settle interval BEFORE requesting actual session closure.
It sends no extra input, never trims audio, and fails after20 drain seconds.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from data.synthetic.native_control_render import sha256, validate_raw
from evals import native_control_teacher as original
from evals import native_control_teacher_exact as exact
from evals.native_control_teacher_artifacts import input_availability, verify_record
from evals.native_control_teacher_exact_quality import verify as verify_parent_quality

MODEL = original.MODEL
PROTOCOL = 'native-control3-exact-reference-playout-drain-v3'
ARTIFACTS = original.ARTIFACTS
FAILED_IDS = ['native-control3-train-006']
atomic_json, collection_lock, local_key = original.atomic_json, original.collection_lock, original.local_key
configuration = exact.configuration
DRAIN_MARGIN_S = .16
OUTPUT_SETTLE_S = 2.
MAXIMUM_DRAIN_S = 20.


class ConfiguredSocket:
    def __init__(self, socket, emit, close):
        self.socket, self.emit, self.close = socket, emit, close

    async def send(self, raw):
        message = json.loads(raw)
        if message['type'] == 'session.start':
            if message['session'] != original.configuration():
                raise ValueError('Original recorder configuration changed')
            message['session'] = configuration()
            self.emit(dict(type='local.teacher.configuration', protocol=PROTOCOL,
                           session_config_sha256=hashlib.sha256(json.dumps(configuration(), sort_keys=True).encode()).hexdigest(),
                           received_at_s=None))
        if message['type'] == 'session.close':
            await self.close()
        await self.socket.send(json.dumps(message))

    def __aiter__(self):
        return self.socket.__aiter__()


async def capture_stream(ws, pcm16, source_samples, emit, *, now=time.monotonic, sleep=asyncio.sleep, audio_sink=None):
    origin, cursor, packet_count, last_arrival = None, 0, 0, None

    def clock():
        nonlocal origin
        value = now()
        # Original recorder's first clock read is session.started, before any
        # sender or receive timestamp. Preserve that same monotonic origin.
        if origin is None:
            origin = value
        return value

    def record(event):
        nonlocal cursor, packet_count, last_arrival
        if event['type'] == 'session.output_audio.delta':
            last_arrival = event['received_at_s']
            cursor = max(cursor, math.ceil(last_arrival * 24000)) + event['audio_bytes'] // 2
            packet_count += 1
        emit(event)

    async def close_after_drain():
        if origin is None or not packet_count:
            raise ValueError('Cannot drain an unobserved teacher output')
        started = now() - origin
        record(dict(type='local.output.drain.started', received_at_s=started,
                    maximum_drain_s=MAXIMUM_DRAIN_S, playout_margin_s=DRAIN_MARGIN_S,
                    output_settle_s=OUTPUT_SETTLE_S))
        while True:
            elapsed = now() - origin
            target = max(cursor / 24000 + DRAIN_MARGIN_S, last_arrival + OUTPUT_SETTLE_S)
            if elapsed >= target:
                break
            if target - started > MAXIMUM_DRAIN_S or elapsed - started >= MAXIMUM_DRAIN_S:
                raise TimeoutError('Actual output playout cannot drain within20 seconds')
            await sleep(min(.25, max(.001, target - elapsed)))
        record(dict(type='local.output.drain.completed', received_at_s=now() - origin,
                    playout_end_sample=cursor, output_packets=packet_count,
                    last_output_arrival_s=last_arrival))

    return await original.capture_stream(ConfiguredSocket(ws, record, close_after_drain), pcm16,
                                         source_samples, record, now=clock, sleep=sleep, audio_sink=audio_sink)


def verify_drain(events):
    begin = [e for e in events if e['type'] == 'local.output.drain.started']
    end = [e for e in events if e['type'] == 'local.output.drain.completed']
    closed = [e for e in events if e['type'] == 'session.closed']
    completed = [e for e in events if e['type'] == 'local.input.completed']
    outputs = [e for e in events if e['type'] == 'session.output_audio.delta']
    if len(begin) != 1 or len(end) != 1 or len(closed) != 1 or len(completed) != 1 or not outputs:
        raise ValueError('Complete actual pre-close output-drain trace required')
    a, b = begin[0], end[0]
    cursor = 0
    for event in outputs:
        cursor = max(cursor, math.ceil(event['received_at_s'] * 24000)) + event['audio_bytes'] // 2
    if (a['maximum_drain_s'] != MAXIMUM_DRAIN_S or a['playout_margin_s'] != DRAIN_MARGIN_S
            or a['output_settle_s'] != OUTPUT_SETTLE_S
            or not completed[0]['received_at_s'] <= a['received_at_s'] <= b['received_at_s'] <= closed[0]['received_at_s']
            or b['received_at_s'] - a['received_at_s'] > MAXIMUM_DRAIN_S + .08
            or b['playout_end_sample'] != cursor or b['output_packets'] != len(outputs)
            or b['last_output_arrival_s'] != outputs[-1]['received_at_s']
            or b['received_at_s'] < max(cursor / 24000 + DRAIN_MARGIN_S,
                                         outputs[-1]['received_at_s'] + OUTPUT_SETTLE_S)):
        raise ValueError('Actual teacher closure preceded complete unchanged packet playout')
    return dict(actual_preclose_drain_verified=True, output_samples_preserved=True)


async def capture_one(raw, row, folder, source_audit_sha, key, *, connector=None):
    import numpy as np
    import soundfile as sf
    if connector is None:
        from websockets.asyncio.client import connect
        connector = connect
    folder.mkdir(parents=True, exist_ok=False)
    source = raw / row['id'] / 'stem_user.wav'
    pcm, rate = sf.read(source, dtype='float32')
    if rate != 24000 or pcm.ndim != 1 or len(pcm) != row['frames'] or not np.isfinite(pcm).all():
        raise ValueError('Invalid sealed user input')
    pcm = np.concatenate([pcm, np.zeros(15 * rate, dtype='float32')])
    sf.write(folder / 'user_in.wav', pcm, rate, subtype='FLOAT')
    pcm16 = (pcm * 32767).astype('<i2')
    (folder / 'input-stream.pcm').write_bytes(pcm16.tobytes())
    environment = dict(schema='dl-native-control3-teacher-environment-v3', purpose='teacher',
                       model=MODEL, protocol=PROTOCOL, source_id=row['id'],
                       source_raw_manifest_sha256=sha256(raw / 'raw-source.json'),
                       source_word_audit_sha256=source_audit_sha, input_user_wav_sha256=sha256(source),
                       input_stream_sha256=sha256(folder / 'input-stream.pcm'),
                       session_config=configuration(), drained_teacher_code_sha256=sha256(Path(__file__)), exact_teacher_code_sha256=sha256(Path('evals/native_control_teacher_exact.py')), teacher_reference_conditioning='authored-control3-TRAIN-only',
                       training_approved=False, owner_recordings_used=False,
                       external_backend_enabled=False, output_suppression=False)
    atomic_json(folder / 'environment.json', environment)
    try:
        with (folder / 'events.jsonl').open('x') as log, (folder / 'output-stream.pcm').open('xb') as spool:
            def emit(event):
                log.write(json.dumps(event) + '\n')
                log.flush()
            def save_audio(data):
                spool.write(data)
                spool.flush()
            async with connector('wss://api.openai.com/v1/live/sessions',
                                 additional_headers={'Authorization': 'Bearer ' + key,
                                                     'OpenAI-Safety-Identifier': hashlib.sha256(b'duplex-live-evaluation').hexdigest()},
                                 open_timeout=20, max_size=2**24) as ws:
                session, audio, packets, transcripts = await capture_stream(
                    ws, pcm16, row['frames'], emit, audio_sink=save_audio)
    except Exception as error:
        session = dict(session_finalized=False, errors=[dict(error_type=type(error).__name__)],
                       source_samples=row['frames'], added_silence_samples=15 * rate,
                       sent_input_samples=0, observed_until_s=0., received_audio_packets=0,
                       training_approved=False, behavior_qualified=False)
        # Recover all already-flushed audio/metadata even if the connection's
        # context manager fails after the receive loop. Never truncate the spool.
        audio = (folder / 'output-stream.pcm').read_bytes()
        saved = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
        packets = [dict(arrival_s=e['received_at_s'], byte_offset=e['output_byte_offset'],
                        bytes=e['audio_bytes'], pcm_sha256=e['pcm_sha256'])
                   for e in saved if e['type'] == 'session.output_audio.delta']
        transcripts = [dict(speaker='user' if 'input_' in e['type'] else 'assistant', text=e['delta'],
                            start_ms=e.get('start_ms'), end_ms=e.get('end_ms')) for e in saved
                       if e['type'] in ('session.input_transcript.delta', 'session.output_transcript.delta')]
    if (folder / 'output-stream.pcm').read_bytes() != audio:
        raise ValueError('Raw output spool differs from the received PCM stream')
    atomic_json(folder / 'audio-packets.json', packets)
    atomic_json(folder / 'transcripts.json', transcripts)
    if packets:
        from evals.audio_playout import render_packets
        chunks = [(p['arrival_s'], np.frombuffer(audio[p['byte_offset']:p['byte_offset'] + p['bytes']], dtype='<i2'))
                  for p in packets]
        played, _ = render_packets(chunks, rate, 0)
        sf.write(folder / 'model_out.wav', played, rate)
    else:
        sf.write(folder / 'model_out.wav', np.zeros(1), rate)
    atomic_json(folder / 'report.json', session)
    return dict(id=row['id'], dir=row['id'],
                artifacts={name: sha256(folder / name) for name in ARTIFACTS},
                transport_complete=session['session_finalized'] and not session['errors']
                                   and session['sent_input_samples'] == len(pcm16) and bool(packets),
                training_approved=False)


def verify_exact_record(raw, row, folder, source_audit_sha, record):
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
    expected = dict(schema='dl-native-control3-teacher-environment-v3', purpose='teacher',
                    model=MODEL, protocol=PROTOCOL, source_id=row['id'],
                    source_raw_manifest_sha256=sha256(raw / 'raw-source.json'),
                    source_word_audit_sha256=source_audit_sha,
                    input_user_wav_sha256=sha256(raw / row['id'] / 'stem_user.wav'),
                    input_stream_sha256=sha256(folder / 'input-stream.pcm'),
                    session_config=configuration(), drained_teacher_code_sha256=sha256(Path(__file__)), exact_teacher_code_sha256=sha256(Path('evals/native_control_teacher_exact.py')), teacher_reference_conditioning='authored-control3-TRAIN-only',
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
    configured = [e for e in events if e['type'] == 'local.teacher.configuration']
    config_sha = hashlib.sha256(json.dumps(configuration(), sort_keys=True).encode()).hexdigest()
    if (len(configured) != 1 or configured[0]['protocol'] != PROTOCOL
            or configured[0]['session_config_sha256'] != config_sha):
        raise ValueError('Actual v2 session configuration trace changed or missing')
    if (len(started) != 1 or started[0]['session']['model'] != MODEL or started[0]['session']['instructions'] != configuration()['instructions'] or len(closed) != 1
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
    verify_drain(events)
    return dict(id=row['id'], transport_integrity_passed=True, training_approved=False,
                behavior_or_label_quality_approved=False)


def parent_evidence(raw, source_audit, parent_batch, quality, diagnosis):
    report = verify_parent_quality(raw, source_audit, parent_batch, quality, require_pass=False)
    parent = json.loads((parent_batch / 'teacher.json').read_text())
    failed = [r for r in report['results'] if not r['checks']['passed']]
    diag = json.loads(diagnosis.read_text())
    if (report['passed'] is not False or [r['id'] for r in failed] != FAILED_IDS
            or failed[0]['checks']['reasons'] != ['Raw teacher waveform exceeds the actual observed session']
            or diag['id'] != FAILED_IDS[0] or diag['training_approved'] is not False
            or diag['original_audio_changed'] is not False or diag['word_agreement_passed'] is not True
            or diag['quiet_interval_passed'] is not True or diag['beyond_observation_nonzero'] != 0
            or diag['wav_sha256'] != next(r for r in parent['recordings'] if r['id'] == FAILED_IDS[0])['artifacts']['model_out.wav']):
        raise ValueError('Only diagnosed complete39/40 capture backlog is eligible for this one-record repair')
    return parent

def validate_collection(raw, source_audit, batch):
    manifest = json.loads((batch / 'teacher.json').read_text())
    lineage = manifest['parent_evidence']
    parent_batch, quality, diagnosis = (Path(lineage[k]) for k in ('batch', 'quality', 'diagnosis'))
    parent = parent_evidence(raw, source_audit, parent_batch, quality, diagnosis)
    source, plan = validate_raw(raw)
    ids = [s['id'] for s in plan['cases']]
    if (manifest['schema'] != 'dl-native-control3-teacher-collection-v3'
            or manifest['model'] != MODEL or manifest['protocol'] != 'mixed-v1-v2-and-' + PROTOCOL
            or manifest['code_sha256'] != sha256(Path(__file__))
            or manifest['source_raw_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or manifest['source_word_audit_sha256'] != sha256(source_audit)
            or manifest['source_plan_sha256'] != source['plan_sha256']
            or manifest['selected_ids'] != ids or len(ids) != 40
            or [r['id'] for r in manifest['recordings']] != ids
            or manifest['maximum_sessions'] != 40 or manifest['maximum_new_sessions'] != 1
            or type(manifest['concurrent_sessions']) is not int
            or manifest['concurrent_sessions'] != 1 or manifest['training_approved'] is not False
            or manifest['status'] != 'raw_complete_unapproved'
            or manifest['new_protocol_ids'] != FAILED_IDS or manifest['retained_records'] != 39
            or lineage['batch_sha256'] != sha256(parent_batch / 'teacher.json')
            or lineage['quality_sha256'] != sha256(quality) or lineage['diagnosis_sha256'] != sha256(diagnosis)):
        raise ValueError('Complete40 mixed-protocol inventory and single diagnosed capture repair scope required')
    for row, record, old in zip(source['cases'], manifest['recordings'], parent['recordings']):
        if row['id'] in FAILED_IDS:
            verify_exact_record(raw, row, batch / row['id'], manifest['source_word_audit_sha256'], record)
        else:
            if record != old:
                raise ValueError('Exact-prompt repair changed a passing retained teacher')
            environment = json.loads((batch / row['id'] / 'environment.json').read_text())
            verifier = exact.verify_exact_record if environment['schema'] == 'dl-native-control3-teacher-environment-v2' else verify_record
            verifier(raw, row, batch / row['id'], manifest['source_word_audit_sha256'], record)
    return manifest


async def collect(raw, source_audit, parent_batch, quality, diagnosis, out, key, lock, concurrency=1):
    if type(concurrency) is not int or concurrency != 1:
        raise ValueError('Exactly one bounded capture session only')
    parent = parent_evidence(raw, source_audit, parent_batch, quality, diagnosis)
    source, plan = validate_raw(raw)
    with collection_lock(lock):
        out.mkdir(parents=True, exist_ok=False)
        retained = [r for r in parent['recordings'] if r['id'] not in FAILED_IDS]
        for record in retained:
            folder = out / record['id']
            folder.mkdir()
            for name in ARTIFACTS:
                shutil.copy2(parent_batch / record['id'] / name, folder / name)
        receipt = dict(schema='dl-native-control3-teacher-collection-v3', model=MODEL,
                       protocol='mixed-v1-v2-and-' + PROTOCOL,
                       source_raw_manifest_sha256=sha256(raw / 'raw-source.json'),
                       source_word_audit_sha256=sha256(source_audit), source_plan_sha256=source['plan_sha256'],
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       code_sha256=sha256(Path(__file__)), created_utc=datetime.now(timezone.utc).isoformat(),
                       selected_ids=[s['id'] for s in plan['cases']], maximum_sessions=40, maximum_new_sessions=1,
                       concurrent_sessions=concurrency, new_protocol_ids=FAILED_IDS, retained_records=39,
                       status='recording', recordings=retained, training_approved=False,
                       parent_evidence=dict(batch=str(parent_batch), quality=str(quality), diagnosis=str(diagnosis),
                           batch_sha256=sha256(parent_batch / 'teacher.json'), quality_sha256=sha256(quality),
                           diagnosis_sha256=sha256(diagnosis)))
        target = out / 'teacher.json'
        atomic_json(target, receipt)
        semaphore = asyncio.Semaphore(concurrency)
        async def one(row):
            async with semaphore:
                record = await capture_one(raw, row, out / row['id'], receipt['source_word_audit_sha256'], key)
                receipt['recordings'].append(record)
                receipt['recordings'].sort(key=lambda r: r['id'])
                atomic_json(target, receipt)
                print(json.dumps(dict(id=row['id'], transport_complete=record['transport_complete'],
                                      complete=len(receipt['recordings']), maximum_new_sessions=1)), flush=True)
                if not record['transport_complete']:
                    raise RuntimeError('Exact-prompt teacher transport failed; preserve all attempts')
        try:
            async with asyncio.TaskGroup() as group:
                for row in source['cases']:
                    if row['id'] in FAILED_IDS:
                        group.create_task(one(row))
        finally:
            receipt['status'] = ('raw_complete_unapproved' if len(receipt['recordings']) == 40
                                 and all(r['transport_complete'] for r in receipt['recordings']) else 'incomplete_unapproved')
            atomic_json(target, receipt)
        validate_collection(raw, source_audit, out)
        return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'source-audit', 'parent-batch', 'quality', 'diagnosis', 'out', 'env-file'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--lock', type=Path, default=Path('runs/local-migration-20261021/.private/control3-capture.lock'))
    ap.add_argument('--concurrency', type=int, choices=(1,), default=1)
    args = ap.parse_args()
    result = asyncio.run(collect(args.raw, args.source_audit, args.parent_batch, args.quality,
                                args.diagnosis, args.out, local_key(args.env_file), args.lock, args.concurrency))
    print(json.dumps(dict(status=result['status'], complete=len(result['recordings']), training_approved=False)))


if __name__ == '__main__':
    main()
