"""Repair five failed TRAIN clips verbatim in a new corpus; preserve held-out audio."""
import argparse
import json
from pathlib import Path
import shutil
from data.synthetic.native_curriculum_v2 import specifications
from data.synthetic.recovery_curriculum import assemble, synthesize
from evals.moshi_model_download import sha256


PLAN = {'native2-train-002': 4, 'native2-train-003': 0, 'native2-train-006': 3,
        'native2-train-015': 3, 'native2-train-017': 2}


def repair(source, out):
    scenes = json.loads((source / 'specifications.json').read_text())
    if scenes != json.loads(json.dumps(specifications())):
        raise ValueError('Repair cannot alter sealed words or folds')
    reports = {r['scenario_id']: r for r in (json.loads(line) for line in (source / 'filter_report.jsonl').read_text().splitlines())}
    failed_train = {sid for sid, r in reports.items() if sid.startswith('native2-train-') and r['verdict'] != 'pass'}
    if failed_train != set(PLAN):
        raise ValueError('Repair is limited to the five independently identified failed TRAIN recordings')
    held = {str(p.relative_to(source)): sha256(p) for scene in scenes if scene['split'] != 'train'
        for p in (source / scene['scenario_id']).rglob('*.wav')}
    shutil.copytree(source, out, ignore=shutil.ignore_patterns('quality*.json*', 'filter_report.jsonl',
        'timing_report.jsonl', 'train_manifest*.jsonl'))
    repairs = []
    for sid, index in sorted(PLAN.items()):
        scene = next(s for s in scenes if s['scenario_id'] == sid)
        role, text = scene['utterances'][index]
        clip = f'{sid}/utterances/{index:02d}_{role}.wav'
        before = sha256(source / clip)
        receipt = synthesize(text, scene[role + '_voice'], out / clip)
        repairs.append(dict(scene=sid, index=index, text_unchanged=text, old_sha256=before,
            new_sha256=receipt['sha256'], actual_request_id=receipt['request_id']))
        print(json.dumps(repairs[-1]), flush=True)
    rows = []
    for scene in scenes:
        receipts = {i: json.loads((out / scene['scenario_id'] / 'utterances' / f'{i:02d}_{role}.receipt.json').read_text())
            for i, (role, _) in enumerate(scene['utterances'])}
        rows.append(assemble(scene, out, receipts))
    if any(sha256(out / path) != digest for path, digest in held.items()):
        raise ValueError('Held-out audio changed during TRAIN-only source repair')
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'generation.json').write_text(json.dumps(dict(schema='dl-native-v2-train-repair-v1', verified=False,
        source_batch=str(source), source_quality_sha256=sha256(source / 'quality.jsonl'),
        source_specifications_sha256=sha256(source / 'specifications.json'), repair_api_calls=len(repairs),
        repairs=repairs, heldout_audio_unchanged=held, folds_and_words_unchanged=True,
        all_clips_require_reverification=True), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    repair(a.source, a.out)
