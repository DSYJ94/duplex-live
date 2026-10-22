"""Complete40 teacher reapproval with explicit protocol transcript boundaries.

The original v1 collector/evaluator and full v2 reapprovals remain immutable.
This v3 evaluator explicitly validates mixed v1/exact-v2 teacher provenance. No audio, ASR word,
timestamp, acceptance threshold or original approval is edited. A failed complete
audit may supply byte-bound recognition evidence, never a passing subset.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path

from data.synthetic.native_control_render import sha256, validate_raw
from evals import native_control_teacher_quality as original
from evals import native_control_teacher_reaudit as parent_quality
from evals.native_control_teacher_exact import validate_collection

SCHEMA = 'dl-native-control3-teacher-quality-v3'
JOIN_POLICY = 'protocol-delta-gap-word-boundary-v1'
GAP_MS = 1000


def transcript_text(transcripts):
    """Keep token deltas; separate lexical chunks across a >=1s protocol gap.

    This uses only the model's original transcript clock, not authored targets or
    ASR text. Within a continuous reply, token fragments remain concatenated.
    """
    rows = [r for r in transcripts if r['speaker'] == 'assistant']
    text, previous, boundaries = '', None, []
    for index, row in enumerate(rows):
        a, b = row['start_ms'], row['end_ms']
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a + b) or not 0 <= a < b
                or (previous and a < previous['start_ms']) or not isinstance(row['text'], str)):
            raise ValueError('Original model transcript needs finite ordered protocol times')
        if (previous and a - previous['end_ms'] >= GAP_MS and text and row['text']
                and text[-1].isalnum() and row['text'][0].isalnum()):
            text += ' '
            boundaries.append(dict(assistant_delta_index=index, previous_end_ms=previous['end_ms'],
                                   next_start_ms=a, gap_ms=a - previous['end_ms']))
        text += row['text']
        previous = row
    return text, dict(policy=JOIN_POLICY, gap_ms=GAP_MS, inserted_whitespace=boundaries,
                      raw_concatenation=''.join(r['text'] for r in rows),
                      original_delta_text_and_timestamps_changed=False, reference_text_used=False)


def pins():
    return dict(original=original.pins(), exact_teacher_sha256=sha256(Path('evals/native_control_teacher_exact.py')),
                parent_reaudit_sha256=sha256(Path('evals/native_control_teacher_reaudit.py')),
                reaudit_sha256=sha256(Path(__file__)))


def decision(raw, source_row, scene, folder, source_checks, response):
    speech, pcm, rate = original.speech_record(folder, response)
    speech['model_text'], speech['transcript_derivation'] = transcript_text(
        json.loads((folder / 'transcripts.json').read_text()))
    timeline = json.loads((raw / scene['id'] / 'timeline.actual.json').read_text())
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    session = json.loads((folder / 'report.json').read_text())
    return dict(id=scene['id'], split='train', source_row=source_row, speech=speech,
                checks=original.assess(scene, timeline, speech, pcm, rate, events, session, source_checks))


def inputs(raw, source_audit, batch):
    manifest = validate_collection(raw, source_audit, batch)
    source, plan = validate_raw(raw)
    return manifest, source, plan, json.loads(source_audit.read_text())


def parent_report(raw, source_audit, batch, quality, seen=None):
    schema = json.loads(quality.read_text())['schema']
    if schema == parent_quality.SCHEMA:
        return parent_quality.verify(raw, source_audit, batch, quality, require_pass=False)
    if schema == SCHEMA:
        return verify(raw, source_audit, batch, quality, require_pass=False, _seen=seen)
    raise ValueError('Unsupported historical teacher approval schema')


def verify(raw, source_audit, batch, approval, *, require_pass=True, _seen=None):
    seen = set() if _seen is None else set(_seen)
    if approval.resolve() in seen:
        raise ValueError('Cyclic teacher recognition lineage')
    seen.add(approval.resolve())
    manifest, source, plan, checks = inputs(raw, source_audit, batch)
    report = json.loads(approval.read_text())
    schema = SCHEMA
    expected_pins = pins()
    if (report['schema'] != schema or report['code_pins'] != expected_pins
            or (require_pass and report['passed'] is not True)
            or type(report['passed']) is not bool or report['all40_complete'] is not True
            or report['training_approved'] is not False or report['asr_weight_sha256'] != original.ASR_WEIGHT
            or report['source_raw_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or report['source_word_audit_sha256'] != sha256(source_audit)
            or report['teacher_manifest_sha256'] != sha256(batch / 'teacher.json')
            or report['expected_text_prompt_used'] is not False or report['original_audio_changed'] is not False
            or report['model_behavior_qualified'] is not False or report['semantic_approval_required'] is not True
            or report['native_conversion_and_numerical_required'] is not True
            or [r['id'] for r in report['results']] != [s['id'] for s in plan['cases']]):
        raise ValueError('Require complete unchanged40 teacher evidence; failed data cannot approve training')
    for source_row, scene, source_checks, row in zip(source['cases'], plan['cases'], checks['results'], report['results']):
        name = scene['id'] + '.asr.json'
        if (row['asr_file'] != name or row['asr_sha256'] != sha256(approval.parent / name)
                or row['internal_silence_removed'] is not False or row['original_clock_restored'] is not True):
            raise ValueError('Independent original teacher recognition evidence changed')
        response = json.loads((approval.parent / name).read_text())
        decide = decision
        actual = decide(raw, source_row, scene, batch / scene['id'], source_checks['clips'], response)
        if any(row[k] != value for k, value in actual.items()) or (require_pass and not actual['checks']['passed']):
            raise ValueError('Original teacher word/phase/quiet decision changed or failed')
    if report['passed'] != all(r['checks']['passed'] for r in report['results']):
        raise ValueError('Whole40 verdict does not match every recomputed decision')
    if report.get('transcript_join_policy') != JOIN_POLICY:
        raise ValueError('Explicit protocol transcript derivation required')
    lineage = report['derivation']
    parent_batch, parent_quality = Path(lineage['parent_batch']), Path(lineage['parent_quality'])
    if (lineage['parent_batch_sha256'] != sha256(parent_batch / 'teacher.json')
            or lineage['parent_quality_sha256'] != sha256(parent_quality)
            or lineage['parent_full40_recomputed'] is not True
            or lineage['thresholds_or_original_audio_changed'] is not False):
        raise ValueError('Original complete recognition lineage changed')
    parent = parent_report(raw, source_audit, parent_batch, parent_quality, seen=seen)
    old_manifest = json.loads((parent_batch / 'teacher.json').read_text())
    expected_reuse = []
    for new_record, old_record, new_row, old_row in zip(manifest['recordings'], old_manifest['recordings'],
                                                       report['results'], parent['results']):
        if new_record['artifacts'] == old_record['artifacts']:
            if (new_row['asr_sha256'] != old_row['asr_sha256']
                    or new_row['leading_window_s'] != old_row['leading_window_s']
                    or new_row['trailing_window_s'] != old_row['trailing_window_s']):
                raise ValueError('Identical original teacher recognition was replaced')
            expected_reuse.append(dict(id=new_record['id'], artifacts=new_record['artifacts'],
                                      asr_sha256=new_row['asr_sha256']))
    if (lineage['byte_bound_reused_records'] != expected_reuse
            or report['reused_recognition_count'] != len(expected_reuse)
            or report['recognition_calls'] != 40 - len(expected_reuse)):
        raise ValueError('Complete40 recognition-call/cache inventory differs')
    return report


def reapprove(raw, source_audit, batch, out, *, reuse_batch, reuse_quality, weights=None, threads=4):
    manifest, source, plan, checks = inputs(raw, source_audit, batch)
    parent = parent_report(raw, source_audit, reuse_batch, reuse_quality)
    old_manifest = json.loads((reuse_batch / 'teacher.json').read_text())
    cached = {r['id']: (record, r) for record, r in zip(old_manifest['recordings'], parent['results'])}
    if weights is not None and sha256(weights) != original.ASR_WEIGHT:
        raise ValueError('Pinned independent turbo weights required')
    packages = {p: importlib.metadata.version(p) for p in (
        'torch', 'numpy', 'scipy', 'openai-whisper', 'whisper-timestamped', 'soundfile')}
    if packages != parent['asr_packages']:
        raise ValueError('Independent recognizer package versions changed')
    if not 1 <= threads <= 8:
        raise ValueError('Use bounded CPU threads')
    out.mkdir(parents=True, exist_ok=False)
    implementation = pins()
    results, reused, calls, model = [], [], 0, None
    for source_row, scene, source_checks, record in zip(source['cases'], plan['cases'], checks['results'], manifest['recordings']):
        old_record, old_row = cached[scene['id']]
        name = scene['id'] + '.asr.json'
        if old_record['artifacts'] == record['artifacts']:
            (out / name).write_bytes((reuse_quality.parent / name).read_bytes())
            head, tail = old_row['leading_window_s'], old_row['trailing_window_s']
            reused.append(dict(id=scene['id'], artifacts=record['artifacts'], asr_sha256=sha256(out / name)))
        else:
            if weights is None:
                raise ValueError('Changed teacher requires new actual independent recognition')
            import numpy as np
            import soundfile as sf
            import torch
            import whisper_timestamped as whisper
            from scipy.signal import resample_poly
            from evals.moshi_speech_audit import asr_window, restore_clock
            if model is None:
                torch.set_num_threads(threads)
                torch.set_num_interop_threads(1)
                model = whisper.load_model(str(weights), device='cpu')
            pcm, rate = sf.read(batch / scene['id'] / 'model_out.wav', dtype='float32')
            window, leading, trailing, silent = asr_window(resample_poly(pcm, 2, 3).astype(np.float32), True, True, True)
            response = dict(text='', segments=[]) if silent else whisper.transcribe(
                model, window, language='en', vad=False, beam_size=5, best_of=5, temperature=0., fp16=False, verbose=None)
            (out / name).write_text(json.dumps(restore_clock(response, leading / 16000), indent=2) + '\n')
            head, tail = leading / 16000, trailing / 16000
            calls += 1
        row = decision(raw, source_row, scene, batch / scene['id'], source_checks['clips'], json.loads((out / name).read_text()))
        row.update(asr_file=name, asr_sha256=sha256(out / name), leading_window_s=head, trailing_window_s=tail,
                   internal_silence_removed=False, original_clock_restored=True)
        results.append(row)
        (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
        print(json.dumps(dict(id=scene['id'], passed=row['checks']['passed'], reasons=row['checks']['reasons'], complete=len(results))), flush=True)
    if implementation != pins():
        raise ValueError('Implementation changed during complete40 reapproval')
    report = dict(parent, schema=SCHEMA, at_utc=datetime.now(timezone.utc).isoformat(), code_pins=implementation,
                  teacher_manifest_sha256=sha256(batch / 'teacher.json'), results=results,
                  passed=all(r['checks']['passed'] for r in results), transcript_join_policy=JOIN_POLICY,
                  recognition_calls=calls, reused_recognition_count=len(reused),
                  derivation=dict(parent_batch=str(reuse_batch), parent_batch_sha256=sha256(reuse_batch / 'teacher.json'),
                                  parent_quality=str(reuse_quality), parent_quality_sha256=sha256(reuse_quality),
                                  parent_schema=parent['schema'], parent_full40_recomputed=True,
                                  byte_bound_reused_records=reused, thresholds_or_original_audio_changed=False))
    target = out / 'teacher-quality.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    verify(raw, source_audit, batch, target, require_pass=False)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'source-audit', 'batch'):
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ('out', 'reuse-batch', 'reuse-quality', 'weights', 'verify'):
        ap.add_argument('--' + name, type=Path)
    args = ap.parse_args()
    if args.verify:
        report = verify(args.raw, args.source_audit, args.batch, args.verify)
    elif args.out and args.reuse_batch and args.reuse_quality:
        report = reapprove(args.raw, args.source_audit, args.batch, args.out, reuse_batch=args.reuse_batch,
                           reuse_quality=args.reuse_quality, weights=args.weights)
    else:
        ap.error('Use --verify or --out with complete parent recognition evidence')
    print(json.dumps(dict(passed=report['passed'], complete=len(report['results']), training_approved=False)))


if __name__ == '__main__':
    main()
