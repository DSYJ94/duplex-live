"""Exact complete34 alignment plus one predeclared independent numeric check.

Alignment is conditioned on the script, so it cannot replace independent ASR.
No retries, no changed audio and no inferred/interpolated word timestamps.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

from .context_source_quality import inventory
from .pacing import units

PAYLOAD = Path('docs/NATIVE-VENUS-CONTEXT-ASSISTANT-SPEECH-PAYLOAD-2026-11-01.json')
ROOT = Path('runs/venus-training-20261101/context-assistant-source-v1')
ADMISSION = Path('docs/NATIVE-VENUS-CONTEXT-ALIGNMENT-ADMISSION-2026-11-01.json')
NUMERIC_ID = 'd43fe31199669e0f875cb17babe99c85f796c1a63fad758576dbf5f317a7f9d7'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admission():
    a = json.loads(ADMISSION.read_text())
    rows = inventory(ROOT, PAYLOAD, 34)
    if a['payload_sha256'] != sha(PAYLOAD) or a['sources'] != [dict(id=r['id'], sha256=r['sha256']) for r in rows]:
        raise ValueError('Exact original complete34 required')
    cost = sum(max(60, r['seconds']) for r in rows) / 3600 + max(60, next(r['seconds'] for r in rows if r['id'] == NUMERIC_ID)) / 3600
    if a['maximum_calls'] != 35 or a['maximum_usd'] != .6 or cost > .6:
        raise ValueError('Exact bounded alignment and diagnostic scope required')
    if 32800 * 4 / 3600 + 32.416625 + .6 > 70:
        raise ValueError('Carried time and financial reserves must fit')
    return rows


def measured_units(text, response, seconds):
    aligned = []
    for w in response['words']:
        start, end = w['start'], w['end']
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or not 0 <= start <= end <= seconds + .02:
            raise ValueError('Invalid measured interval')
        aligned.append(dict(text=w['text'], timestamp=[start, end]))
    # Existing units() rejects omitted/changed words and any fabricated split
    # of a multiword number. Original punctuation/text are preserved.
    return units(text, aligned)


def execute(output):
    import requests
    rows = admission()
    if output.exists():
        raise ValueError('Observe original attempt; never restart')
    key = os.environ.get('ELEVENLABS_API_KEY')
    if not key:
        raise ValueError('Local private key required')
    claim = Path('runs/venus-training-20261101/context-source-claims') / (sha(ADMISSION) + '.json')
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open('x') as f:
        json.dump(dict(admission_sha256=sha(ADMISSION), output=str(output.resolve())), f)
    output.mkdir(parents=True)
    ledger = dict(admission_sha256=sha(ADMISSION), attempts=[], complete=False, automatic_retries=0, training_approved=False)
    def save():
        tmp = output / 'ledger.tmp'
        tmp.write_text(json.dumps(ledger, indent=2) + '\n')
        tmp.replace(output / 'ledger.json')
    def call(row, endpoint, data, filename):
        attempt = dict(id=row['id'], endpoint=endpoint, data=data, original_wav_sha256=row['sha256'], status='reserved', cost_upper_usd=max(60, row['seconds']) / 3600)
        ledger['attempts'].append(attempt)
        save()
        try:
            with Path(row['wav']).open('rb') as f:
                result = requests.post('https://api.elevenlabs.io/v1/' + endpoint, headers={'xi-api-key': key}, files={'file': ('original.wav', f, 'audio/wav')}, data=data, timeout=120)
            (output / filename).write_bytes(result.content)
            attempt.update(http_status=result.status_code, request_id=result.headers.get('request-id'), response_sha256=sha(output / filename))
            result.raise_for_status()
            body = result.json()
            attempt['status'] = 'complete'
            save()
            return body
        except Exception as error:
            attempt.update(status='failed_or_unknown', error_type=type(error).__name__)
            save()
            raise RuntimeError('Preserve actual attempt; no retries') from None
    save()
    numeric = next(r for r in rows if r['id'] == NUMERIC_ID)
    diagnostic = call(numeric, 'speech-to-text', dict(model_id='scribe_v2', language_code='en', diarize='false', tag_audio_events='false'), 'numeric-diagnostic.json')
    # No expected script, hints, financial values or scoring answers go to ASR.
    (output / 'numeric-recognized.txt').write_text(diagnostic['text'] + '\n')
    report = []
    for row in rows:
        response = call(row, 'forced-alignment', dict(text=row['text']), row['id'] + '.json')
        speech = measured_units(row['text'], response, row['seconds'])
        report.append(dict(id=row['id'], fold=row['fold'], source_sha256=row['sha256'], text=row['text'], measured_words=response['words'], alignment_loss=response.get('loss'), native_speech_units=speech))
    ledger.update(complete=True, conservative_cost_upper_usd=sum(x['cost_upper_usd'] for x in ledger['attempts']))
    save()
    (output / 'report.json').write_text(json.dumps(dict(all34_exact_word_timings=True, rows=report, independent_numeric_review='PENDING', alignment_is_not_independent_recognition=True, training_approved=False), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    execute(p.parse_args().output)
