"""Independent whole40 source-word audit for the new native control collection.

Reads raw source and a pinned local Whisper-turbo checkpoint. Never prompts ASR
with reference text, modifies source, accepts a subset, or approves student fitting.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path

from data.synthetic.native_control_render import sha256, validate_raw
from evals.native_control_words import POLICY, VARIANTS, agreement, spoken_words

ASR_WEIGHT = 'aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a'
WER_BAR = .05
PEAK_BAR = .995
SCHEMA = 'dl-native-control3-source-word-audit-v2'


def assess_clip(reference, result, pcm, rate):
    import numpy as np
    reasons = []
    words = [w for segment in result.get('segments', []) for w in segment.get('words', [])]
    duration = len(pcm) / rate
    expected = spoken_words(reference)
    transcript = spoken_words(result.get('text', ''))
    comparison = agreement(reference, result.get('text', ''))
    if not words or comparison['canonical_word_wer'] > WER_BAR:
        reasons.append('Independent source word agreement failed')
    flat = [t for w in words for t in spoken_words(w['text'])]
    # A low aggregate WER must not bless a changed cue, premise or step. Keep
    # exact word-position checks for source labels, not guessed word alignments.
    if flat != expected or flat != transcript:
        reasons.append('Exact authored source word positions do not match measured speech')
    previous = 0.
    for word in words:
        start, end = word['start'], word['end']
        if (type(start) not in (int, float) or type(end) not in (int, float)
                or not math.isfinite(start + end) or not 0 <= start < end <= duration + .08
                or start < previous - .02 or len(spoken_words(word['text'])) != 1):
            reasons.append('Invalid independently measured word interval')
            break
        previous = end
    if (rate != 24000 or pcm.ndim != 1 or not .1 <= duration <= 30
            or not np.isfinite(pcm).all() or not .001 < float(np.abs(pcm).max()) <= PEAK_BAR):
        reasons.append('Empty, invalid or clipped source waveform')
    active = [i for i in range(0, len(pcm), 480)
              if float(np.sqrt(np.mean(pcm[i:i + 480] ** 2))) > .001]
    first_active = active[0] / rate if active else None
    if first_active is None or (words and words[0]['start'] < first_active - .2):
        reasons.append('Word onset predates actual source voice beyond200ms')
    return dict(passed=not reasons, reasons=reasons, raw_wer=comparison['raw_wer'],
                canonical_word_wer=comparison['canonical_word_wer'], word_agreement=comparison,
                authored_text=reference, asr_text=result.get('text', ''), measured_words=words,
                duration_s=duration, peak=float(np.abs(pcm).max()), first_active_s=first_active,
                expected_text_prompt_used=False, waveform_changed=False)


def audit(raw, weights, out, device, threads=4, reuse_raw=None, reuse_approval=None):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import whisper_timestamped as whisper
    import torch
    raw_manifest, plan = validate_raw(raw)
    if bool(reuse_raw) != bool(reuse_approval):
        raise ValueError('Cached recognition requires its complete original source and audit')
    cached, reused = {}, []
    cache_report = None
    if reuse_approval:
        cache_report = verify(reuse_raw, reuse_approval, require_pass=False)
        if cache_report['asr_packages'] != {p: importlib.metadata.version(p)
                                           for p in ('openai-whisper', 'whisper-timestamped')}:
            raise ValueError('Cached independent recognition package versions changed')
        cached = {(r['id'], c['turn_index']): c for r in cache_report['results'] for c in r['clips']}
    if sha256(weights) != ASR_WEIGHT:
        raise ValueError('Exact pinned independent Whisper turbo weights required')
    if out.exists():
        raise ValueError('Use a new immutable source-audit directory')
    if not 1 <= threads <= 8:
        raise ValueError('Use bounded independent ASR threads')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    implementation = sha256(Path(__file__))
    model = whisper.load_model(str(weights), device=device)
    out.mkdir(parents=True)
    rows = []
    for scene in plan['cases']:
        clips = []
        for i, turn in enumerate(scene['turns']):
            path = raw / scene['id'] / 'utterances' / f'{i:02d}_{turn["role"]}.wav'
            pcm, rate = sf.read(path, dtype='float32')
            name = f'{scene["id"]}-{i:02d}-asr.json'
            previous = cached.get((scene['id'], i))
            unchanged = (previous and previous['wav_sha256'] == sha256(path)
                         and previous['receipt_sha256'] == sha256(path.with_suffix('.receipt.json')))
            if unchanged:
                (out / name).write_bytes((reuse_approval.parent / previous['asr_file']).read_bytes())
                reused.append(dict(id=scene['id'], turn_index=i, wav_sha256=previous['wav_sha256'],
                                   receipt_sha256=previous['receipt_sha256'], asr_sha256=previous['asr_sha256']))
            else:
                asr_pcm = resample_poly(pcm, 2, 3).astype(np.float32)
                result = whisper.transcribe(model, asr_pcm, language='en', vad=False,
                                            best_of=5, beam_size=5, temperature=0., fp16=device != 'cpu', verbose=None)
                (out / name).write_text(json.dumps(result, indent=2) + '\n')
            result = json.loads((out / name).read_text())
            verdict = assess_clip(turn['text'], result, pcm, rate)
            clips.append(dict(turn_index=i, role=turn['role'], wav_sha256=sha256(path),
                              receipt_sha256=sha256(path.with_suffix('.receipt.json')),
                              asr_file=name, asr_sha256=sha256(out / name), **verdict))
            (out / 'progress.json').write_text(json.dumps(dict(
                scene=scene['id'], turn=i, completed=sum(len(r['clips']) for r in rows) + len(clips),
                expected=168, current_passed=verdict['passed'], training_approved=False)) + '\n')
        row = dict(id=scene['id'], split='train', passed=all(c['passed'] for c in clips), clips=clips)
        rows.append(row)
        print(json.dumps(dict(id=row['id'], passed=row['passed'], clips=len(clips))), flush=True)
    if implementation != sha256(Path(__file__)):
        raise ValueError('Source audit implementation changed during actual transcription')
    result = dict(schema=SCHEMA, at_utc=datetime.now(timezone.utc).isoformat(),
                  raw_manifest_sha256=sha256(raw / 'raw-source.json'), plan_sha256=raw_manifest['plan_sha256'],
                  code_sha256=sha256(Path(__file__)), renderer_sha256=sha256(Path(__file__).parents[1] / 'data/synthetic/native_control_render.py'),
                  asr_model='whisper-turbo', asr_weight_sha256=ASR_WEIGHT,
                  word_comparison_policy=POLICY, orthographic_variants=VARIANTS.copy(),
                  word_logic_sha256=sha256(Path(__file__).with_name('native_control_words.py')),
                  asr_packages={p: importlib.metadata.version(p) for p in ('openai-whisper', 'whisper-timestamped')},
                  device=device, threads=threads, precision='float32' if device == 'cpu' else 'float16',
                  expected_text_prompt_used=False, wer_bar=WER_BAR, peak_bar=PEAK_BAR,
                  all40_complete=True, complete_clips=sum(len(r['clips']) for r in rows),
                  passed=all(r['passed'] for r in rows), results=rows,
                  recognition_calls=168 - len(reused), reused_clips=len(reused),
                  cache_provenance=(dict(raw=str(reuse_raw), audit=str(reuse_approval),
                                         audit_sha256=sha256(reuse_approval),
                                         raw_manifest_sha256=sha256(reuse_raw / 'raw-source.json'),
                                         reused=reused) if reuse_approval else None),
                  original_waveforms_changed=False, passing_subset_accepted=False,
                  training_approved=False, actual_teacher_collection_completed=False,
                  remaining='Actual teacher collection, independent quality and semantic review, conversion and native numerical approval')
    (out / 'source-word-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify(raw, approval, *, require_pass=True):
    """Recompute the entire report; only explicit repair inspection permits failure.

    All collection/training-facing callers and the CLI require a complete pass.
    require_pass=False validates evidence for diagnosed repairs, never approval.
    """
    import soundfile as sf
    manifest, plan = validate_raw(raw)
    report = json.loads(approval.read_text())
    if (report['schema'] != SCHEMA
            or report['raw_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or report['plan_sha256'] != manifest['plan_sha256']
            or report['code_sha256'] != sha256(Path(__file__))
            or report['renderer_sha256'] != sha256(Path(__file__).parents[1] / 'data/synthetic/native_control_render.py')
            or report['asr_model'] != 'whisper-turbo' or report['asr_weight_sha256'] != ASR_WEIGHT
            or report['word_comparison_policy'] != POLICY or report['orthographic_variants'] != VARIANTS
            or report['word_logic_sha256'] != sha256(Path(__file__).with_name('native_control_words.py'))
            or report['expected_text_prompt_used'] is not False
            or report['wer_bar'] != WER_BAR or report['peak_bar'] != PEAK_BAR
            or report['all40_complete'] is not True or report['complete_clips'] != 168
            or report['original_waveforms_changed'] is not False or report['passing_subset_accepted'] is not False
            or type(report['passed']) is not bool or (require_pass and report['passed'] is not True)
            or report['training_approved'] is not False
            or report['actual_teacher_collection_completed'] is not False
            or [r['id'] for r in report['results']] != [s['id'] for s in plan['cases']]):
        raise ValueError('Missing complete unchanged independent source-word audit')
    for scene, row in zip(plan['cases'], report['results']):
        if (row['split'] != 'train' or len(row['clips']) != len(scene['turns'])
                or row['passed'] != all(c['passed'] for c in row['clips'])
                or (require_pass and row['passed'] is not True)):
            raise ValueError('Incomplete or failed source scene')
        for i, (turn, clip) in enumerate(zip(scene['turns'], row['clips'])):
            path = raw / scene['id'] / 'utterances' / f'{i:02d}_{turn["role"]}.wav'
            asr_name = f'{scene["id"]}-{i:02d}-asr.json'
            response = approval.parent / asr_name
            if (clip['turn_index'] != i or clip['role'] != turn['role']
                    or clip['wav_sha256'] != sha256(path)
                    or clip['receipt_sha256'] != sha256(path.with_suffix('.receipt.json'))
                    or clip['asr_file'] != asr_name or clip['asr_sha256'] != sha256(response)):
                raise ValueError('Source clip or recognizer output lineage changed')
            pcm, rate = sf.read(path, dtype='float32')
            actual = assess_clip(turn['text'], json.loads(response.read_text()), pcm, rate)
            saved = {k: v for k, v in clip.items() if k not in
                     ('turn_index', 'role', 'wav_sha256', 'receipt_sha256', 'asr_file', 'asr_sha256')}
            if actual != saved or (require_pass and actual['passed'] is not True):
                raise ValueError('Source word judgment changed or failed')
    if report['passed'] != all(r['passed'] for r in report['results']):
        raise ValueError('Whole source report disagrees with its complete measured outcomes')
    if report.get('cache_provenance'):
        cache = report['cache_provenance']
        parent_raw, parent_audit = Path(cache['raw']), Path(cache['audit'])
        if (parent_audit.resolve() == approval.resolve()
                or cache['audit_sha256'] != sha256(parent_audit)
                or cache['raw_manifest_sha256'] != sha256(parent_raw / 'raw-source.json')):
            raise ValueError('Cached recognition parent changed')
        parent = verify(parent_raw, parent_audit, require_pass=False)
        old = {(r['id'], c['turn_index']): c for r in parent['results'] for c in r['clips']}
        current = {(r['id'], c['turn_index']): c for r in report['results'] for c in r['clips']}
        keys = [(r['id'], r['turn_index']) for r in cache['reused']]
        if (len(set(keys)) != len(keys) or report['reused_clips'] != len(keys)
                or report['recognition_calls'] != 168 - len(keys)):
            raise ValueError('Cached recognition counts or inventory changed')
        for key, entry in zip(keys, cache['reused']):
            if key not in old or key not in current or any(
                    entry[k] != old[key][k] or entry[k] != current[key][k]
                    for k in ('wav_sha256', 'receipt_sha256', 'asr_sha256')):
                raise ValueError('Cached response does not belong to the identical original waveform/receipt')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--reuse-raw', type=Path)
    parser.add_argument('--reuse-audit', type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.raw, args.verify)
    elif args.weights and args.out:
        result = audit(args.raw, args.weights, args.out, args.device, args.threads, args.reuse_raw, args.reuse_audit)
    else:
        parser.error('Use --verify, or --weights with --out')
    print(json.dumps(dict(passed=result['passed'], complete_clips=result['complete_clips'], training_approved=False)))


if __name__ == '__main__':
    main()
