"""Collect one immutable complete40 raw GPT-Live control TRAIN collection.

Requires the exact independent source-word audit. No retry, output suppression,
worker substitution, training approval or GPU provisioning. The API key stays local.
"""
import argparse
import asyncio
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from data.synthetic.native_control_repair import QA
from data.synthetic.native_control_render import sha256, validate_raw
from evals.native_control_source_quality import verify as verify_source

MODEL = 'gpt-live-1'
PROTOCOL = 'native-control3-teacher-v1'
ARTIFACTS = ('environment.json', 'user_in.wav', 'input-stream.pcm', 'events.jsonl',
             'output-stream.pcm', 'audio-packets.json', 'model_out.wav',
             'transcripts.json', 'report.json')


def configuration():
    instructions = (
        'You are a concise English voice assistant collecting synthetic TRAIN labels. '
        'Answer ordinary questions using the stated premises. Follow the entire spoken instruction. '
        'When asked to stay quiet, produce no acknowledgment, clarification, backchannel or speech. '
        'A cue mentioned inside its defining instruction is not a release. Wait for a later user turn. '
        'Unrelated speech or a similar-sounding word does not release you. When the user changes the cue, '
        'the old cue no longer releases you; naming the new cue inside that change is not a release. '
        'If no later valid cue arrives, keep listening silently until the session ends. '
        'For two-step instructions, say the first commanded step exactly once after hearing the complete '
        'instruction. After the later pause, wait silently for the valid cue and say only the remaining '
        'step, without restarting. After a valid cue with a question, answer that question once. '
        'These simple tasks require no delegation. Speak the complete commanded or reference words clearly. '
        'The following authored TRAIN-only ordinary-answer references guide label wording; they supply '
        'no private values and do not establish unaided reasoning. They are never student inference inputs. '
    )
    references = [dict(question=q, answer=a) for q, a in QA]
    return dict(model=MODEL, delegation=dict(type='client'),
                instructions=instructions + json.dumps(references, sort_keys=True),
                audio=dict(format=dict(type='audio/pcm', rate=24000), output=dict(voice='marin')))


def local_key(env_file=None):
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key and env_file:
        if env_file.stat().st_mode & 0o077:
            raise ValueError('Local credential file must have mode0600')
        matches = []
        for line in env_file.read_text().splitlines():
            match = re.fullmatch(r'\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*', line)
            if match:
                try:
                    values = shlex.split(match[1], comments=True)
                except ValueError:
                    raise ValueError('Malformed local credential assignment') from None
                if len(values) > 1:
                    raise ValueError('Malformed local credential assignment')
                matches.append(values[0] if values else '')
        if len(matches) == 1:
            key = matches[0]
    if not key:
        raise RuntimeError('Local OPENAI_API_KEY is not configured')
    return key


@contextmanager
def collection_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another local control3 collection is still executing') from None
        yield
    finally:
        os.close(fd)  # The OS releases the lease on normal exit or process death.


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def safe_event(event):
    """Keep synthetic protocol evidence while excluding credential-bearing fields."""
    secret_fields = {'authorization', 'api_key', 'client_secret', 'access_token',
                     'refresh_token', 'session_token', 'session-token'}
    def clean(value):
        if isinstance(value, dict):
            return {k: ('<REDACTED>' if k.casefold() in secret_fields else clean(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    if event.get('type') == 'error':
        error = event.get('error', {})
        return dict(type='error', error_type=error.get('type'), error_code=error.get('code'))
    return clean(event)


async def capture_stream(ws, pcm16, source_samples, emit, *, now=time.monotonic, sleep=asyncio.sleep,
                         audio_sink=None):
    """One continuous stream; emit is an artifact sink, never an output filter."""
    ready = asyncio.Event()
    start = None
    sent = 0
    closed = False
    closed_at = None
    errors = []
    transcripts = []
    packets = []
    output = bytearray()
    config = configuration()
    await ws.send(json.dumps(dict(type='session.start', event_id='control3_start', session=config)))

    def record(event):
        value = dict(safe_event(event), received_at_s=None if start is None else now() - start)
        emit(value)
        return value

    async def sender():
        nonlocal sent
        await asyncio.wait_for(ready.wait(), timeout=30)
        for first in range(0, len(pcm16), 1920):
            last = min(first + 1920, len(pcm16))
            await sleep(max(0., start + last / 24000 - now()))
            chunk = pcm16[first:last].tobytes()
            await ws.send(json.dumps(dict(type='session.input_audio.append', audio=base64.b64encode(chunk).decode())))
            sent = last
            record(dict(type='local.input.append', first_sample=first, sample_count=last - first,
                        pcm_sha256=hashlib.sha256(chunk).hexdigest()))
        record(dict(type='local.input.completed', source_samples=source_samples, sent_input_samples=sent))
        await sleep(2)
        await ws.send(json.dumps(dict(type='session.close')))

    task = asyncio.create_task(sender())
    try:
        async with asyncio.timeout(len(pcm16) / 24000 + 55):
            async for message in ws:
                event = json.loads(message)
                kind = event.get('type')
                if kind == 'session.started':
                    if ready.is_set() or event['session']['model'] != MODEL:
                        raise ValueError('Unexpected teacher session/model')
                    start = now()
                    ready.set()
                if kind == 'session.output_audio.delta':
                    if start is None:
                        raise ValueError('Teacher audio before session readiness')
                    data = base64.b64decode(event['delta'], validate=True)
                    if not data or len(data) % 2:
                        raise ValueError('Invalid raw PCM16 output')
                    offset = len(output)
                    if audio_sink:
                        audio_sink(data)
                    output.extend(data)
                    row = record({k: v for k, v in event.items() if k != 'delta'} | dict(
                        audio_bytes=len(data), output_byte_offset=offset,
                        pcm_sha256=hashlib.sha256(data).hexdigest()))
                    packets.append(dict(arrival_s=row['received_at_s'], byte_offset=offset, bytes=len(data),
                                        pcm_sha256=hashlib.sha256(data).hexdigest()))
                else:
                    observed_event = record(event)
                if kind in ('session.input_transcript.delta', 'session.output_transcript.delta'):
                    transcripts.append(dict(speaker='user' if 'input_' in kind else 'assistant',
                                            text=event['delta'], start_ms=event.get('start_ms'), end_ms=event.get('end_ms')))
                if kind == 'error':
                    errors.append(safe_event(event))
                    if start is None:
                        break
                if kind == 'session.closed':
                    closed = True
                    closed_at = observed_event['received_at_s']
                    break
    except BaseException as error:
        errors.append(dict(error_type=type(error).__name__))
        # Preserve partial raw evidence. The caller decides whether to abort the
        # entire collection; no failed record is silently retried or discarded.
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        task.cancel()
        sender_result = await asyncio.gather(task, return_exceptions=True)
        for result in sender_result:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                errors.append(dict(error_type=type(result).__name__, component='input_sender'))
    observed = closed_at if closed_at is not None else (0. if start is None else now() - start)
    session = dict(session_finalized=closed, errors=errors, source_samples=source_samples,
                   added_silence_samples=15 * 24000, sent_input_samples=sent, observed_until_s=observed,
                   received_audio_packets=len(packets), training_approved=False,
                   behavior_qualified=False)
    return session, bytes(output), packets, transcripts


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
    environment = dict(schema='dl-native-control3-teacher-environment-v1', purpose='teacher',
                       model=MODEL, protocol=PROTOCOL, source_id=row['id'],
                       source_raw_manifest_sha256=sha256(raw / 'raw-source.json'),
                       source_word_audit_sha256=source_audit_sha, input_user_wav_sha256=sha256(source),
                       input_stream_sha256=sha256(folder / 'input-stream.pcm'),
                       session_config=configuration(), teacher_reference_conditioning='authored-control3-TRAIN-only',
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


async def collect(raw, audit_path, out, key, lock):
    verify_source(raw, audit_path)
    manifest, plan = validate_raw(raw)
    with collection_lock(lock):
        out.mkdir(parents=True, exist_ok=False)
        receipt = dict(schema='dl-native-control3-teacher-collection-v1', model=MODEL, protocol=PROTOCOL,
                       source_raw_manifest_sha256=sha256(raw / 'raw-source.json'),
                       source_word_audit_sha256=sha256(audit_path), source_plan_sha256=manifest['plan_sha256'],
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       code_sha256=sha256(Path(__file__)), created_utc=datetime.now(timezone.utc).isoformat(),
                       selected_ids=[s['id'] for s in plan['cases']], maximum_sessions=40,
                       concurrent_sessions=2, status='recording', recordings=[], training_approved=False)
        target = out / 'teacher.json'
        atomic_json(target, receipt)
        semaphore = asyncio.Semaphore(2)
        async def one(row):
            async with semaphore:
                record = await capture_one(raw, row, out / row['id'], receipt['source_word_audit_sha256'], key)
                receipt['recordings'].append(record)
                receipt['recordings'].sort(key=lambda r: r['id'])
                atomic_json(target, receipt)
                print(json.dumps(dict(id=record['id'], transport_complete=record['transport_complete'],
                                      completed=len(receipt['recordings']), training_approved=False)), flush=True)
                if not record['transport_complete']:
                    raise RuntimeError('Raw teacher transport failed; preserve the incomplete collection')
        try:
            async with asyncio.TaskGroup() as group:
                for row in manifest['cases']:
                    group.create_task(one(row))
        finally:
            receipt['status'] = ('raw_complete_unapproved' if len(receipt['recordings']) == 40
                                 and all(r['transport_complete'] for r in receipt['recordings']) else 'incomplete_unapproved')
            atomic_json(target, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--lock', type=Path,
                        default=Path('runs/local-migration-20261021/.private/control3-capture.lock'))
    args = parser.parse_args()
    key = local_key(args.env_file)
    result = asyncio.run(collect(args.raw, args.source_audit, args.out, key, args.lock))
    print(json.dumps(dict(status=result['status'], recordings=len(result['recordings']), training_approved=False)))


if __name__ == '__main__':
    main()
