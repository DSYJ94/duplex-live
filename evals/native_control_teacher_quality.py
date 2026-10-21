"""Bind independent ASR and native control phase checks to ALL40 raw teachers.

No teacher rewriting, subset approval, source normalization, semantic approval,
student fitting or native qualification. Original stream/WAV bytes stay immutable.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

from data.synthetic.native_control_render import sha256, validate_raw
from evals.native_control_source_quality import ASR_WEIGHT
from evals.native_control_teacher_artifacts import validate_collection
from evals.native_control_teacher_checks import assess


def pins():
    return {name: sha256(Path(__file__).with_name(name)) for name in (
        'native_control_teacher_quality.py', 'native_control_teacher_checks.py',
        'native_control_teacher_artifacts.py', 'native_control_teacher.py',
        'native_control_source_quality.py', 'moshi_speech_audit.py', 'audio_playout.py')}


def speech_record(folder, response):
    import numpy as np
    import soundfile as sf
    pcm, rate = sf.read(folder / 'model_out.wav', dtype='float32')
    transcripts = json.loads((folder / 'transcripts.json').read_text())
    words = [w for s in response.get('segments', []) for w in s.get('words', [])]
    active = [i for i in range(0, len(pcm), 480)
              if float(np.sqrt(np.mean(pcm[i:i + 480] ** 2))) > .001]
    starts_with_voice = bool(words and active and words[0]['start'] >= active[0] / rate - .2)
    speech = dict(model_text=''.join(r['text'] for r in transcripts if r['speaker'] == 'assistant'),
                  asr_text=response.get('text', ''), asr_segments=response.get('segments', []),
                  word_starts_within_recorded_audio=starts_with_voice,
                  expected_text_prompt_used=False, waveform_changed=False)
    return speech, pcm, rate


def decision(raw, source_row, scene, folder, source_checks, response):
    speech, pcm, rate = speech_record(folder, response)
    timeline = json.loads((raw / scene['id'] / 'timeline.actual.json').read_text())
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    session = json.loads((folder / 'report.json').read_text())
    return dict(id=scene['id'], split='train', source_row=source_row, speech=speech,
                checks=assess(scene, timeline, speech, pcm, rate, events, session, source_checks))


def inputs(raw, source_audit, batch):
    manifest = validate_collection(raw, source_audit, batch)
    source, plan = validate_raw(raw)
    checks = json.loads(source_audit.read_text())
    return manifest, source, plan, checks


def audit(raw, source_audit, batch, weights, out, device, threads):
    import numpy as np
    import soundfile as sf
    import torch
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from evals.moshi_speech_audit import asr_window, restore_clock
    manifest, source, plan, checks = inputs(raw, source_audit, batch)
    if sha256(weights) != ASR_WEIGHT:
        raise ValueError('Require the pinned independent Whisper turbo weights')
    if not 1 <= threads <= 8:
        raise ValueError('Use bounded local ASR threads')
    out.mkdir(parents=True, exist_ok=False)
    implementation = pins()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model = whisper.load_model(str(weights), device=device)
    results = []
    for source_row, scene, source_checks in zip(source['cases'], plan['cases'], checks['results']):
        folder = batch / scene['id']
        pcm, rate = sf.read(folder / 'model_out.wav', dtype='float32')
        audio = resample_poly(pcm, 2, 3).astype(np.float32)
        window, head, tail, silent = asr_window(audio, True, True, True)
        response = dict(text='', segments=[]) if silent else whisper.transcribe(
            model, window, language='en', vad=False, beam_size=5, best_of=5,
            temperature=0., fp16=device != 'cpu', verbose=None)
        response = restore_clock(response, head / 16000)
        name = scene['id'] + '.asr.json'
        (out / name).write_text(json.dumps(response, indent=2) + '\n')
        row = decision(raw, source_row, scene, folder, source_checks['clips'], response)
        row.update(asr_file=name, asr_sha256=sha256(out / name),
                   leading_window_s=head / 16000, trailing_window_s=tail / 16000,
                   internal_silence_removed=False, original_clock_restored=True)
        results.append(row)
        (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
        print(json.dumps(dict(id=row['id'], passed=row['checks']['passed'],
                              reasons=row['checks']['reasons'], complete=len(results))), flush=True)
    if implementation != pins():
        raise ValueError('Audit implementation changed during actual transcription')
    report = dict(schema='dl-native-control3-teacher-quality-v1',
                  at_utc=datetime.now(timezone.utc).isoformat(), code_pins=implementation,
                  source_raw_manifest_sha256=sha256(raw / 'raw-source.json'),
                  source_word_audit_sha256=sha256(source_audit),
                  teacher_manifest_sha256=sha256(batch / 'teacher.json'),
                  asr_weight_sha256=ASR_WEIGHT, expected_text_prompt_used=False,
                  asr_packages={p: importlib.metadata.version(p) for p in (
                      'torch', 'numpy', 'scipy', 'openai-whisper', 'whisper-timestamped', 'soundfile')},
                  device=device, threads=threads, precision='float32' if device == 'cpu' else 'float16',
                  all40_complete=True, passed=all(r['checks']['passed'] for r in results),
                  results=results, training_approved=False, original_audio_changed=False,
                  semantic_approval_required=True, native_conversion_and_numerical_required=True,
                  model_behavior_qualified=False)
    (out / 'teacher-quality.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def verify(raw, source_audit, batch, approval):
    _, source, plan, checks = inputs(raw, source_audit, batch)
    report = json.loads(approval.read_text())
    if (report['schema'] != 'dl-native-control3-teacher-quality-v1'
            or report['code_pins'] != pins() or report['passed'] is not True
            or report['all40_complete'] is not True or report['training_approved'] is not False
            or report['asr_weight_sha256'] != ASR_WEIGHT
            or report['source_raw_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or report['source_word_audit_sha256'] != sha256(source_audit)
            or report['teacher_manifest_sha256'] != sha256(batch / 'teacher.json')
            or report['expected_text_prompt_used'] is not False
            or report['original_audio_changed'] is not False
            or report['model_behavior_qualified'] is not False
            or report['semantic_approval_required'] is not True
            or report['native_conversion_and_numerical_required'] is not True
            or [r['id'] for r in report['results']] != [s['id'] for s in plan['cases']]):
        raise ValueError('Require the complete unchanged40 independent teacher quality approval')
    for source_row, scene, source_checks, row in zip(
            source['cases'], plan['cases'], checks['results'], report['results']):
        name = scene['id'] + '.asr.json'
        if (row['asr_file'] != name or row['asr_sha256'] != sha256(approval.parent / name)
                or row['internal_silence_removed'] is not False
                or row['original_clock_restored'] is not True):
            raise ValueError('Independent teacher recognizer evidence changed')
        response = json.loads((approval.parent / name).read_text())
        actual = decision(raw, source_row, scene, batch / scene['id'], source_checks['clips'], response)
        if any(row[k] != value for k, value in actual.items()) or actual['checks']['passed'] is not True:
            raise ValueError('Teacher speech, phase or full source decision changed or failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.verify:
        report = verify(args.raw, args.source_audit, args.batch, args.verify)
    elif args.weights and args.out:
        report = audit(args.raw, args.source_audit, args.batch, args.weights, args.out, args.device, args.threads)
    else:
        parser.error('Use --verify, or --weights and --out')
    print(json.dumps(dict(passed=report['passed'], complete=len(report['results']), training_approved=False)))


if __name__ == '__main__':
    main()
