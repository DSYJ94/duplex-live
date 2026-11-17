"""Input-only intake for later native TRAIN failure mining; no model work.

Reuses qualified source recordings and folds. A future collector must extract
only the complete user channel, then append the fixed response window; this
manifest is not an inference, correctness label or new native-training approval.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

RECORD_KEYS = {'id','fold','question','source_wave','source_wave_sha256',
               'source_sidecar_sha256','user_channel','user_frames','response_tail_seconds',
               'required_user_only_extraction'}


def build(manifest, semantic):
    rows = manifest['rows']
    records = semantic['exact_semantic_records']
    if (manifest.get('complete1072') is not True or manifest.get('TRAIN') != 1008
            or manifest.get('DEV') != 64 or len(rows) != 1072 or len(records) != 1072):
        raise ValueError('Complete qualified1072 parents required, never a convenient subset')
    by = {r['id']:r for r in records}
    ids = [r['id'] for r in rows]
    if len(by) != 1072 or len(set(ids)) != 1072 or set(ids) != set(by):
        raise ValueError('Exact distinct source/semantic membership required')
    if Counter(r['fold'] for r in rows) != Counter(TRAIN=1008,DEV=64):
        raise ValueError('Original complete TRAIN/DEV folds required')
    result = []
    for row in rows:
        original = by[row['id']]
        if original['fold'] != row['fold']:
            raise ValueError('No relabelling inspected DEV as TRAIN')
        if row['fold'] != 'TRAIN':
            continue
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('Contained original source path required')
        if any(re.fullmatch('[0-9a-f]{64}',row[k]) is None for k in ('wav_sha256','sidecar_sha256')):
            raise ValueError('Original waveform and sidecar hashes required')
        user = [p for p in row['parents'] if p['id'] == row['id']+'-user']
        if len(user) != 1 or type(user[0]['resampled_frames']) is not int or user[0]['resampled_frames'] < 1:
            raise ValueError('Complete measured user channel length required')
        question = original['question']
        if not isinstance(question,str) or not question.strip():
            raise ValueError('Original nonempty source question required')
        result.append(dict(id=row['id'],fold='TRAIN',question=question,
                           source_wave=path.as_posix(),source_wave_sha256=row['wav_sha256'],
                           source_sidecar_sha256=row['sidecar_sha256'],user_channel=1,
                           user_frames=user[0]['resampled_frames'],response_tail_seconds=60,
                           required_user_only_extraction=True))
    if len(result) != 1008 or any(set(r) != RECORD_KEYS for r in result):
        raise ValueError('Exactly1008 source-input-only TRAIN records required')
    return dict(schema='dl-native-TRAIN-failure-inputs-v1',records=result,required=1008,
                original_DEV_excluded=64,labels_in_model_payload=False,
                teacher_channel_is_not_an_input=True,source_wave_hashes_rechecked=False,
                native_inference_started=False,correctness_labels_created=False,
                new_optimizer_updates=0,external_API_calls=0)


def main(a):
    source = a.source.resolve()
    manifest = source/'manifest.json'
    plan = source/'qualification/plan.json'
    value = build(json.loads(manifest.read_text()),json.loads(plan.read_text()))
    value.update(source_root=str(source),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 semantic_parent_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    with a.out.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k != 'records'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
