"""Whole32 derived speech audit with waveform-only recognition regions.

Original24kHz files, clocks, frame/word/peak gates and failed audits stay intact.
Only the independent recognizer's temporary16kHz copy skips long quiet gaps.
No authored/provider words or scoring answers reach the remote ASR worker.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

SCHEMA = 'dl-native-reasoning32-teacher-speech-regions-v2'
PLAN_SCHEMA = SCHEMA + '-recognition-plan'
WORKER_SCHEMA = SCHEMA + '-recognition-result'
WEIGHT = 'aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a'
PACKAGES = {'openai-whisper': '20250625', 'whisper-timestamped': '1.15.9'}
POLICY = dict(block_samples_24k=480, rms_threshold=.001, padding_samples_24k=3840,
              merge_gap_samples_24k=24000, original_rate=24000,
              recognizer_rate=16000, beam_size=5, best_of=5, temperature=0.,
              fp16=False, language='en', original_timestamps_restored=True,
              reference_text_prompt_used=False, original_waveforms_changed=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def regions(pcm, rate):
    """Keep EVERY above-threshold20ms block; never consult any transcript."""
    import numpy as np
    if rate != 24000 or pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError('Original finite mono24kHz required')
    active = [i for i in range(0, len(pcm), 480)
              if float(np.sqrt(np.mean(pcm[i:i + 480] ** 2))) > .001]
    spans = []
    for start in active:
        a, b = max(0, start - 3840), min(len(pcm), start + 480 + 3840)
        if spans and a - spans[-1][1] <= 24000:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    if any(not any(a <= i and min(i + 480, len(pcm)) <= b for a, b in spans)
           for i in active):
        raise ValueError('Every detectable original block must remain in ASR copy')
    return [[a / rate, b / rate] for a, b in spans]


def validate_plan(plan):
    rows = plan['recordings']
    if (plan['schema'] != PLAN_SCHEMA or plan['policy'] != POLICY
            or plan['code_sha256'] != sha(Path(__file__))
            or len(rows) != 32 or len({r['id'] for r in rows}) != 32
            or [r['id'] for r in rows] != [f'native-reasoning4-train-{i:03d}' for i in range(32)]
            or any(set(r) != {'id', 'wav', 'wav_sha256', 'samples', 'rate'}
                   or r['wav'] != r['id'] + '/model_out.wav' or r['rate'] != 24000
                   for r in rows)):
        raise ValueError('Exact whole32 word-free recognition plan required')
    return plan


def prepare(source, quality, batch, engine, tokenizer, failed, output):
    from evals import native_reasoning_teacher_quality as old
    import soundfile as sf
    previous = old.verify(source, quality, batch, engine, tokenizer, failed, require_pass=False)
    if previous['passed'] is not False:
        raise ValueError('Preserve and bind actual failed whole32 audit')
    rows = []
    for row in previous['results']:
        name = row['id'] + '/model_out.wav'; info = sf.info(batch / name)
        rows.append(dict(id=row['id'], wav=name, wav_sha256=row['original_rendered_wav_sha256'],
                         samples=info.frames, rate=info.samplerate))
    plan = dict(schema=PLAN_SCHEMA, code_sha256=sha(Path(__file__)), policy=POLICY,
                original_failed_audit_sha256=sha(failed),
                teacher_collection_sha256=sha(batch / 'teacher-collection.json'),
                source_quality_sha256=sha(quality), recordings=rows)
    validate_plan(plan)
    with output.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    return plan


def recognize(plan_path, audio_root, weights, output, threads=8):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import torch
    import whisper_timestamped as whisper
    plan = validate_plan(json.loads(plan_path.read_text()))
    packages = {p: importlib.metadata.version(p) for p in PACKAGES}
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available()
            or packages != PACKAGES or sha(weights) != WEIGHT
            or type(threads) is not int or not 1 <= threads <= 8):
        raise ValueError('Pinned recognizer, CPU-only execution and bounded CPU threads required')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    model = whisper.load_model(str(weights), device='cpu'); rows = []
    for row in plan['recordings']:
        wav = audio_root / row['wav']
        if wav.is_symlink() or sha(wav) != row['wav_sha256']:
            raise ValueError('Original preserved WAV changed')
        pcm, rate = sf.read(wav, dtype='float32')
        if len(pcm) != row['samples'] or rate != row['rate']:
            raise ValueError('Original full waveform geometry changed')
        spans = regions(pcm, rate)
        raw = (whisper.transcribe(model, resample_poly(pcm, 2, 3).astype(np.float32),
                   language='en', vad=[tuple(r) for r in spans], best_of=5, beam_size=5,
                   temperature=0., fp16=False, verbose=None)
               if spans else dict(text='', segments=[], speech_activity=[]))
        name = row['id'] + '-asr.json'; write(output / name, raw)
        if sha(wav) != row['wav_sha256']:
            raise ValueError('Original WAV changed during recognition')
        rows.append(dict(id=row['id'], wav_sha256=row['wav_sha256'],
                         original_clock_regions=spans, asr_file=name, asr_sha256=sha(output / name)))
        write(output / 'progress.json', dict(complete=len(rows), expected=32, training_approved=False))
        print(json.dumps(dict(id=row['id'], complete=len(rows), expected=32)), flush=True)
    value = dict(schema=WORKER_SCHEMA, at_utc=datetime.now(timezone.utc).isoformat(),
                 code_sha256=sha(Path(__file__)), plan_sha256=sha(plan_path), policy=POLICY,
                 asr_weight_sha256=WEIGHT, asr_packages=packages,
                 runtime_packages={p: importlib.metadata.version(p) for p in
                                   ('torch', 'numpy', 'scipy', 'soundfile', 'dtw-python')},
                 asr_device='cpu', threads=threads, complete_cases=32, recordings=rows,
                 training_approved=False)
    write(output / 'recognition.json', value)
    return value


def assess_all(source, quality, batch, engine, tokenizer, failed, plan_path, worker):
    """Recompute every gate from ORIGINAL audio and independently decoded words."""
    from evals import native_reasoning_teacher_quality as old
    from evals import native_reasoning_source_quality as speech
    import soundfile as sf
    previous = old.verify(source, quality, batch, engine, tokenizer, failed, require_pass=False)
    plan = validate_plan(json.loads(plan_path.read_text())); result = json.loads((worker / 'recognition.json').read_text())
    if (previous['passed'] is not False or plan['original_failed_audit_sha256'] != sha(failed)
            or plan['teacher_collection_sha256'] != sha(batch / 'teacher-collection.json')
            or plan['source_quality_sha256'] != sha(quality)
            or result['schema'] != WORKER_SCHEMA or result['code_sha256'] != sha(Path(__file__))
            or result['plan_sha256'] != sha(plan_path) or result['policy'] != POLICY
            or result['asr_weight_sha256'] != WEIGHT or result['asr_packages'] != PACKAGES
            or result['asr_device'] != 'cpu' or type(result['threads']) is not int
            or not 1 <= result['threads'] <= 8 or result['complete_cases'] != 32
            or result['training_approved'] is not False
            or [r['id'] for r in result['recordings']] != [r['id'] for r in plan['recordings']]):
        raise ValueError('Complete independently bound whole32 recognition required')
    rows = []
    for original, recognized, prior in zip(plan['recordings'], result['recordings'], previous['results']):
        identity = original['id']; wav = batch / original['wav']; raw_file = worker / recognized['asr_file']
        if (recognized['asr_file'] != identity + '-asr.json'
                or sha(wav) != original['wav_sha256'] or original['wav_sha256'] != prior['original_rendered_wav_sha256']
                or recognized['wav_sha256'] != sha(wav) or sha(raw_file) != recognized['asr_sha256']):
            raise ValueError('Changed original audio or independent transcript')
        pcm, rate = sf.read(wav, dtype='float32'); spans = regions(pcm, rate)
        raw = json.loads(raw_file.read_text())
        mapped = [dict(start=round(a * 16000) / 16000, end=round(b * 16000) / 16000) for a, b in spans]
        if (recognized['original_clock_regions'] != spans or len(pcm) != original['samples']
                or rate != original['rate'] or raw.get('speech_activity') != mapped):
            raise ValueError('Waveform-only regions/original clock mapping changed')
        text = old.expected_output(batch / identity); verdict = speech.assess(text, raw, pcm, rate)
        report = json.loads((batch / identity / 'report.json').read_text())
        if not text: verdict['reasons'].append('No actual provider speech transcript')
        if report['native_frame_budget']['frame_count_diagnostic_pass'] is not True:
            verdict['reasons'].append('Complete prefix/live/actually sent context spans exceed unchanged1100-frame bound')
        verdict['passed'] = not verdict['reasons']
        rows.append(dict(id=identity, original_rendered_wav_sha256=sha(wav), actual_provider_transcript=text,
                         original_clock_regions=spans, asr_file=recognized['asr_file'],
                         asr_sha256=recognized['asr_sha256'], **verdict))
    return rows


def finalize(source, quality, batch, engine, tokenizer, failed, plan_path, worker):
    rows = assess_all(source, quality, batch, engine, tokenizer, failed, plan_path, worker)
    result = dict(schema=SCHEMA, at_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha(Path(__file__)),
                  original_failed_audit_sha256=sha(failed), recognition_sha256=sha(worker / 'recognition.json'),
                  complete_cases=32, results=rows, passed=all(r['passed'] for r in rows),
                  original_audio_changed=False, passing_subset_accepted=False,
                  semantic_approved=False, training_approved=False)
    target = worker / 'teacher-speech-quality.json'
    if target.exists(): raise FileExistsError('Preserve original derived verdict')
    write(target, result)
    return result


def verify(source, quality, batch, engine, tokenizer, failed, plan_path, worker, *, require_pass=True):
    rows = assess_all(source, quality, batch, engine, tokenizer, failed, plan_path, worker)
    v = json.loads((worker / 'teacher-speech-quality.json').read_text())
    if (v['schema'] != SCHEMA or v['code_sha256'] != sha(Path(__file__))
            or v['original_failed_audit_sha256'] != sha(failed)
            or v['recognition_sha256'] != sha(worker / 'recognition.json')
            or v['complete_cases'] != 32 or v['results'] != rows
            or v['passed'] is not all(r['passed'] for r in rows)
            or any(v[k] is not False for k in ('original_audio_changed', 'passing_subset_accepted',
                                               'semantic_approved', 'training_approved'))
            or (require_pass and v['passed'] is not True)):
        raise ValueError('All32 original-waveform verdicts required; no subset or rewritten approval')
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'finalize'):
        q = sub.add_parser(command)
        for name in ('source', 'quality', 'batch', 'engine', 'tokenizer', 'failed', 'plan'):
            q.add_argument('--' + name, required=True, type=Path)
        if command == 'finalize': q.add_argument('--worker', required=True, type=Path)
    q = sub.add_parser('recognize')
    for name in ('plan', 'audio-root', 'weights', 'output'): q.add_argument('--' + name, required=True, type=Path)
    q.add_argument('--threads', type=int, default=8); a = p.parse_args()
    if a.command == 'recognize':
        recognize(a.plan, a.audio_root, a.weights, a.output, a.threads)
    else:
        args = (a.source, a.quality, a.batch, a.engine, a.tokenizer, a.failed, a.plan)
        if a.command == 'prepare': prepare(*args)
        else:
            result = finalize(*args, a.worker)
            verify(*args, a.worker, require_pass=False)
            print(json.dumps(dict(complete_cases=32, passed=sum(r['passed'] for r in result['results']),
                                  whole_passed=result['passed'], training_approved=False)))
            if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
