"""Audit every original new96 recording; no synthesis, subset or audio edits."""
import argparse
import array
import json
import math
import wave
from pathlib import Path
from .retention_speech import prepare, inherited_sources, canonical_file_payload, sha, MODEL
from .context_source_quality import wer


def inventory(original, resumed, payload):
    original, resumed, payload = map(Path, (original, resumed, payload))
    prepared = json.loads(payload.read_text())
    if prepared != prepare() or sha(payload) != canonical_file_payload(prepared):
        raise ValueError('Exact complete committed speech payload required')
    inherited, remaining = inherited_sources(prepared, original)
    ledger = json.loads((resumed/'ledger.json').read_text())
    # Absolute source paths can change across machines; immutable receipts cannot.
    if (len(inherited) != 135 or len(remaining) != 47 or not ledger['complete']
            or ledger['payload_sha256'] != sha(payload) or len(ledger['attempts']) != 47
            or len(ledger['inherited_verified_sources']) != 135
            or any(a['original_attempt'] != b['original_attempt']
                   or a['original_ledger_sha256'] != b['original_ledger_sha256']
                   for a, b in zip(ledger['inherited_verified_sources'], inherited))):
        raise ValueError('Complete original135 plus never-attempted47 receipts required')
    rows = []
    for index, request in enumerate(prepared['requests']):
        root = original if index < 135 else resumed
        attempt = inherited[index]['original_attempt'] if index < 135 else ledger['attempts'][index-135]
        if (attempt['id'] != request['id'] or attempt['payload_sha256'] != request['payload_sha256']
                or attempt['voice_id'] != request['voice_id'] or attempt['model_id'] != MODEL
                or not attempt.get('request_id')
                or (index >= 135 and attempt['status'] != 'COMPLETE')):
            raise ValueError('Actual response and exact authored text/voice required')
        folder = root/request['id']; wav = folder/'original.wav'; pcm = folder/'original.pcm'
        if sha(wav) != attempt['original_wav_sha256'] or sha(pcm) != attempt['original_pcm_sha256']:
            raise ValueError('Original source hash changed')
        with wave.open(str(wav)) as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (24000, 1, 2):
                raise ValueError('Original24k mono PCM required')
            raw = stream.readframes(stream.getnframes())
        if not raw or raw != pcm.read_bytes():
            raise ValueError('Original PCM/WAV mismatch')
        peak = max(abs(v) for v in array.array('h', raw))/32768
        seconds = len(raw)/48000
        if peak != attempt['peak'] or seconds != attempt['seconds']:
            raise ValueError('Original sample measurement differs from receipt')
        rows.append(dict(id=request['id'], fold=request['fold'], text=request['body']['text'],
            profile=request['profile'], wav=str(wav), source_sha256=sha(wav),
            seconds=seconds, original_peak=peak, peak_pass=peak <= .995,
            original_attempt_status=attempt['status']))
    if len(rows) != 182 or len({r['id'] for r in rows}) != 182:
        raise ValueError('Exact distinct182 required')
    return rows


def complete_timestamps(chunks, seconds):
    if not isinstance(chunks, list) or not chunks:
        return False
    previous = -1.
    for word in chunks:
        bounds = word.get('timestamp')
        if not word.get('text', '').strip() or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            return False
        start, end = bounds
        if (any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds)
                or not 0 <= start <= end <= seconds+.25 or start < previous):
            return False
        previous = start
    return True


def main(a):
    rows = inventory(a.original, a.resumed, a.payload)
    if a.output.exists():
        raise ValueError('Preserve original audits; never overwrite or retry blindly')
    import numpy as np
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from portable.venus.native_app import ensure_guard
    from .asr import enable_complete_word_timestamps
    ensure_guard()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One guarded scheduler GPU; never Mac CPU ASR')
    name = 'openai/whisper-large-v3-turbo'
    revision = '41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
    model = AutoModelForSpeechSeq2Seq.from_pretrained(name, revision=revision,
        local_files_only=True, torch_dtype=torch.float16, attn_implementation='eager').cuda().eval()
    enable_complete_word_timestamps(model)
    processor = AutoProcessor.from_pretrained(name, revision=revision, local_files_only=True)
    recognize = pipeline('automatic-speech-recognition', model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, torch_dtype=torch.float16, device=0)
    a.output.mkdir(parents=True, exist_ok=False)
    for count, row in enumerate(rows, 1):
        ensure_guard()
        with wave.open(row['wav']) as stream:
            raw = stream.readframes(stream.getnframes())
        audio = np.frombuffer(raw, dtype='<i2').astype(np.float32)/32768
        converted = resample_poly(audio, 2, 3).astype(np.float32)
        recognized = recognize(dict(array=converted, sampling_rate=16000), return_timestamps='word',
            generate_kwargs=dict(language='en', task='transcribe', do_sample=False))
        error = wer(row['text'], recognized['text'])
        timing = complete_timestamps(recognized.get('chunks'), row['seconds'])
        row.update(recognized=recognized['text'], word_error_rate=error, words=recognized.get('chunks'),
            word_timestamps_complete=timing, recognizer_resample_peak=float(np.abs(converted).max()),
            passed=row['peak_pass'] and error <= .05 and timing)
        (a.output/(row['id']+'.json')).write_text(json.dumps(row, indent=2)+'\n')
        print(json.dumps(dict(completed=count, required=182, id=row['id'], passed=row['passed'], wer=error)), flush=True)
    report = dict(schema='dl-retention96-complete182-quality-v1', source=a.source,
        payload_sha256=sha(a.payload), original_ledger_sha256=sha(a.original/'ledger.json'),
        resumed_ledger_sha256=sha(a.resumed/'ledger.json'), complete=len(rows) == 182,
        required=182, passed=sum(r['passed'] for r in rows), all_passed=all(r['passed'] for r in rows),
        rows=rows, recognizer=name, recognizer_revision=revision,
        normalizer_module_sha256=sha(Path(__file__).with_name('context_source_quality.py')),
        wer_threshold=.05, original_peak_threshold=.995, no_audio_changes=True,
        transcript_hints=False, subset_acceptance=False, external_api_calls=0,
        training_approved=False, naturalness_approved=False)
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['all_passed']:
        raise RuntimeError('Whole source cohort failed; preserve every original and no subset approval')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'resumed', 'payload', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('source', 'lease'):
        p.add_argument('--'+name, required=True)
    main(p.parse_args())
