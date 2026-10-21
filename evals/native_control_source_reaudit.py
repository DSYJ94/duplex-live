"""Reassess ALL168 durable ASR responses without editing words, audio or times.

An observed NumPy-scalar check bug in the first actual audit disagreed with its
own JSON verifier. Preserve that full failed audit; judge every saved response
again in a new version. No new recognition, normalization or relaxed threshold.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from data.synthetic.native_control_render import sha256, validate_raw
from evals.native_control_source_quality import ASR_WEIGHT, PEAK_BAR, WER_BAR, assess_clip, verify


def reassess(raw, parent, out):
    import soundfile as sf
    source, plan = validate_raw(raw)
    report = json.loads(parent.read_text())
    pipeline = json.loads((parent.parent.parent / 'pipeline.json').read_text())
    implementation = subprocess.check_output([
        'git', 'show', pipeline['source_commit'] + ':evals/native_control_source_quality.py'])
    if (pipeline['phase'] != 'stopped_preserve_all_artifacts'
            or pipeline['failed_phase'] != 'verify_source168'
            or report['schema'] != 'dl-native-control3-source-word-audit-v1'
            or report['code_sha256'] != hashlib.sha256(implementation).hexdigest()
            or report['passed'] is not False or report['all40_complete'] is not True
            or report['complete_clips'] != 168 or report['training_approved'] is not False
            or report['actual_teacher_collection_completed'] is not False
            or report['raw_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or report['plan_sha256'] != source['plan_sha256']
            or report['asr_model'] != 'whisper-turbo' or report['asr_weight_sha256'] != ASR_WEIGHT
            or report['expected_text_prompt_used'] is not False
            or report['original_waveforms_changed'] is not False
            or report['passing_subset_accepted'] is not False
            or report['wer_bar'] != WER_BAR or report['peak_bar'] != PEAK_BAR
            or report['renderer_sha256'] != sha256(Path('data/synthetic/native_control_render.py'))
            or [r['id'] for r in report['results']] != [s['id'] for s in plan['cases']]):
        raise ValueError('Require the full immutable failed source audit and stopped parent supervisor')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'parent-source-word-audit.json').write_bytes(parent.read_bytes())
    results = []
    for scene, previous in zip(plan['cases'], report['results']):
        if previous['split'] != 'train' or len(previous['clips']) != len(scene['turns']):
            raise ValueError('Incomplete parent source scene or changed fold')
        clips = []
        for i, (turn, old) in enumerate(zip(scene['turns'], previous['clips'])):
            wav = raw / scene['id'] / 'utterances' / f'{i:02d}_{turn["role"]}.wav'
            name = f'{scene["id"]}-{i:02d}-asr.json'
            response = parent.parent / name
            if (old['turn_index'] != i or old['role'] != turn['role']
                    or old['authored_text'] != turn['text'] or old['asr_file'] != name
                    or old['wav_sha256'] != sha256(wav)
                    or old['receipt_sha256'] != sha256(wav.with_suffix('.receipt.json'))
                    or old['asr_sha256'] != sha256(response)):
                raise ValueError('Parent recognizer output, original words, waveform or receipt changed')
            (out / name).write_bytes(response.read_bytes())
            pcm, rate = sf.read(wav, dtype='float32')
            # Read exactly the durable bytes, as the independent verifier does.
            verdict = assess_clip(turn['text'], json.loads((out / name).read_text()), pcm, rate)
            clips.append(dict(turn_index=i, role=turn['role'], wav_sha256=old['wav_sha256'],
                              receipt_sha256=old['receipt_sha256'], asr_file=name,
                              asr_sha256=old['asr_sha256'], **verdict))
        results.append(dict(id=scene['id'], split='train', passed=all(c['passed'] for c in clips), clips=clips))
    result = dict(report, at_utc=datetime.now(timezone.utc).isoformat(), results=results,
                  passed=all(r['passed'] for r in results),
                  code_sha256=sha256(Path('evals/native_control_source_quality.py')),
                  derivation=dict(schema='dl-full-durable-json-reassessment-v1',
                                  code_sha256=sha256(Path(__file__)),
                                  parent_audit_sha256=sha256(parent),
                                  parent_source_commit=pipeline['source_commit'],
                                  parent_code_sha256=report['code_sha256'],
                                  recognition_calls=0, complete168_recomputed=True,
                                  raw_responses_copied_byte_for_byte=True,
                                  numeric_timestamp_values_changed=False,
                                  source_words_or_audio_changed=False,
                                  spelling_normalization_added=False, thresholds_changed=False))
    target = out / 'source-word-audit.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    if result['passed']:
        verify(raw, target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'parent', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = reassess(args.raw, args.parent, args.out)
    print(json.dumps(dict(passed=report['passed'], clips=report['complete_clips'],
                          failed_scenes=[r['id'] for r in report['results'] if not r['passed']],
                          training_approved=False)))


if __name__ == '__main__':
    main()
