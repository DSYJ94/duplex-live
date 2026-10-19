"""Diagnose the two failed TRAIN sources; render each once into a new directory."""
import argparse
import json
from pathlib import Path
import shutil

from data.synthetic.native_curriculum_v2 import specifications
from data.synthetic.recovery_curriculum import assemble, synthesize
from evals.moshi_model_download import sha256


PLAN = {'native2-train-015': (3, '30eb80cdb035580fbcb5dc3026e0f16472cb09b66351b4e60bb4f63577a519f2'),
        'native2-train-017': (0, 'edb3cc2467b79e5468276dd4760d36a85f562186d1d98f083a1574fc0469feff')}
INSTRUCTIONS = ('Speak naturally in clear conversational English. Read the supplied words exactly once. '
                'Do not repeat, add words, vocalize, or make any other sound after the final word.')


def diagnose(source):
    import numpy as np
    import soundfile as sf
    scenes = json.loads((source / 'specifications.json').read_text())
    if scenes != json.loads(json.dumps(specifications())):
        raise ValueError('Sealed words or folds changed')
    reports = {r['scenario_id']: r for r in map(json.loads, (source / 'filter_report.jsonl').read_text().splitlines())}
    failed = {sid for sid, r in reports.items() if sid.startswith('native2-train-') and r['verdict'] != 'pass'}
    if failed != set(PLAN):
        raise ValueError('This repair is only for the diagnosed 22/24 audit, never a passing subset')
    findings = []
    for sid, (index, expected_hash) in PLAN.items():
        scene = next(s for s in scenes if s['scenario_id'] == sid)
        role, text = scene['utterances'][index]
        path = source / sid / 'utterances' / f'{index:02d}_{role}.wav'
        if sha256(path) != expected_hash:
            raise ValueError('Diagnosed original waveform changed')
        pcm, rate = sf.read(path)
        clip = reports[sid]['asr_detail']['per_utterance'][index]
        if clip['wav_sha256'] != expected_hash or clip['ref'] != text:
            raise ValueError('Independent measured audit does not bind this exact clip')
        if not np.isfinite(pcm).all() or rate != 24000 or pcm.ndim != 1:
            raise ValueError('Unexpected raw source waveform format')
        saturated = np.flatnonzero(np.abs(pcm) >= .999)
        finding = dict(scene=sid, index=index, role=role, text=text, sha256=expected_hash,
                       seconds=len(pcm)/rate, peak=float(np.max(np.abs(pcm))),
                       saturated_samples=len(saturated), independent_hypothesis=clip['hyp'])
        if sid.endswith('015'):
            # Two independently measured words each coincide with actual acoustic energy.
            if text != 'Go.' or clip['hyp'].strip() != 'Go. Go.' or len(clip['words']) != 2:
                raise ValueError('Repeated-word diagnosis no longer matches audit')
            energies = [float(np.sqrt(np.mean(pcm[round(w['start']*rate):round(w['end']*rate)]**2)))
                        for w in clip['words']]
            if min(energies) < .03:
                raise ValueError('No raw acoustic support for both measured Go words')
            finding.update(diagnosis='Two acoustic speech bursts, independently transcribed Go. Go.',
                           measured_word_rms=energies)
        else:
            stem, _ = sf.read(source / sid / 'stem_user.wav')
            mix, _ = sf.read(source / sid / 'mixed.wav')
            # The first utterance starts at the sealed 800ms offset, before other speakers.
            offset = round(.8*rate)
            if len(saturated) != 50 or not np.array_equal(pcm, stem[offset:offset+len(pcm)]) or not np.array_equal(pcm, mix[offset:offset+len(pcm)]):
                raise ValueError('Clipping is not reproduced in the raw first utterance/stem/mix')
            finding.update(diagnosis='Clipped raw API source; identical samples propagate into stem and mix.',
                           saturation_start_s=float(saturated[0]/rate), saturation_end_s=float(saturated[-1]/rate),
                           source_equals_stem_and_mix=True)
        findings.append(finding)
    return scenes, findings


def repair(source, out):
    scenes, findings = diagnose(source)
    if out.exists():
        raise ValueError('A repair must use a new versioned directory')
    protected = {str(p.relative_to(source)): sha256(p) for p in source.rglob('*')
                 if p.is_file() and (p.suffix == '.wav' or p.name.endswith('.receipt.json'))
                 and not any(str(p.relative_to(source)).startswith(sid + '/') for sid in PLAN)}
    held = {p: h for p, h in protected.items() if p.startswith(('native2-validation-', 'native2-test-'))}
    shutil.copytree(source, out, ignore=shutil.ignore_patterns('quality*.json*', 'filter_report.jsonl',
                    'timing_report.jsonl', 'train_manifest*.jsonl'))
    (out / 'generation.previous-v1.json').write_bytes((source / 'generation.json').read_bytes())
    rows = {r['scenario_id']: r for r in map(json.loads, (source / 'manifest.jsonl').read_text().splitlines())}
    repairs = []
    for finding in findings:
        sid, index, role = finding['scene'], finding['index'], finding['role']
        scene = next(s for s in scenes if s['scenario_id'] == sid)
        name = f'{index:02d}_{role}'
        relative = Path(sid) / 'utterances' / (name + '.wav')
        originals = out / 'repair-originals' / sid
        originals.mkdir(parents=True)
        for suffix in ('.wav', '.receipt.json'):
            shutil.copy2(source / sid / 'utterances' / (name + suffix), originals / (name + suffix))
        unchanged = {p.name: sha256(p) for p in (source / sid / 'utterances').iterdir()
                     if p.name not in (name+'.wav', name+'.receipt.json')}
        receipt = synthesize(finding['text'], scene[role+'_voice'], out / relative, instructions=INSTRUCTIONS)
        if not receipt['request_id']:
            raise ValueError('Missing actual speech API receipt')
        for filename, digest in unchanged.items():
            if sha256(out / sid / 'utterances' / filename) != digest:
                raise ValueError('An undiagnosed utterance changed')
        receipts = {i: json.loads((out / sid / 'utterances' / f'{i:02d}_{r}.receipt.json').read_text())
                    for i, (r, _) in enumerate(scene['utterances'])}
        rows[sid] = assemble(scene, out, receipts)
        repairs.append(dict(scene=sid, index=index, text_unchanged=finding['text'],
                            old_sha256=finding['sha256'], new_sha256=receipt['sha256'],
                            actual_request_id=receipt['request_id']))
    for path, digest in protected.items():
        if sha256(out / path) != digest:
            raise ValueError('Unrelated source audio or receipt changed')
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows.values()))
    report = dict(schema='dl-native-v2-train-repair-v2', verified=False, source_batch=str(source),
                  source_quality_sha256=sha256(source / 'quality.jsonl'),
                  source_filter_sha256=sha256(source / 'filter_report.jsonl'),
                  source_specifications_sha256=sha256(source / 'specifications.json'),
                  diagnoses=findings, repairs=repairs, repair_api_calls=len(repairs),
                  heldout_audio_unchanged=held, protected_audio_and_receipts=protected,
                  folds_and_words_unchanged=True, thresholds_unchanged=True,
                  all_24_train_scenes_require_independent_reverification=True)
    (out / 'generation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(repairs=repairs, heldout_files=len(held), training_approved=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.out:
        repair(a.source, a.out)
    else:
        print(json.dumps(diagnose(a.source)[1], indent=2))
        raise SystemExit(1)  # Expected red feedback loop: these diagnosed sources fail.
