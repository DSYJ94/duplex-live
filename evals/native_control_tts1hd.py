"""Explicit text-only synthesizer for the two persistent TRAIN source failures.

No native model change or automatic provider fallback. Official Speech API:
https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create
tts-1-hd supports alloy/WAV but not instructions. Original WAVs stay preserved.
"""
import hashlib
import json
import os
import time

MODEL = 'tts-1-hd'


def synthesize(text, voice, path, *, instructions=None):
    import requests
    import soundfile as sf
    if voice != 'alloy':
        raise ValueError('This diagnosed repair is scoped to the existing TRAIN alloy voice')
    key = os.environ['OPENAI_API_KEY']
    for attempt, backoff in enumerate((0, 1, 5)):
        if backoff:
            time.sleep(backoff)
        try:
            response = requests.post('https://api.openai.com/v1/audio/speech',
                                     headers={'Authorization': 'Bearer ' + key}, timeout=90,
                                     json=dict(model=MODEL, input=text, voice=voice, response_format='wav'))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            continue
        if response.status_code == 200:
            path.write_bytes(response.content)
            pcm, rate = sf.read(path)
            receipt = dict(request_id=response.headers.get('x-request-id'), source_model=MODEL,
                           voice=voice, sha256=hashlib.sha256(response.content).hexdigest(),
                           attempt=attempt, sample_rate=rate, seconds=len(pcm) / rate,
                           input_text=text, instructions=None)
            path.with_suffix('.receipt.json').write_text(json.dumps(receipt) + '\n')
            # The caller applies the unchanged finite/mono/24k/peak rules. Never
            # resample, trim, normalize or lower gain to manufacture acceptance.
            return receipt
        if response.status_code not in (429, 500, 502, 503):
            raise RuntimeError('Speech API rejected request: HTTP ' + str(response.status_code))
    raise RuntimeError('Speech API retries exhausted; no source approval')
