"""Reuse exact original TRAIN teaching words as language-source candidates.

No new answers, speech, folds, semantic approval or weight updates are created.
The original qualification/provenance and family-overlap checks remain required.
"""
from collections import Counter
import hashlib
import json


def build(plan_bytes, expected_sha256):
    if hashlib.sha256(plan_bytes).hexdigest() != expected_sha256:
        raise ValueError('Exact independently qualified original source plan required')
    plan = json.loads(plan_bytes)
    records = plan['exact_semantic_records']
    if (len(records) != 1072 or len({r['id'] for r in records}) != 1072
            or Counter(r['fold'] for r in records) != Counter(TRAIN=1008, DEV=64)):
        raise ValueError('Whole original1072 membership and unchanged folds required')
    candidates = []
    for row in records:
        if row['fold'] != 'TRAIN':
            continue
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ('question', 'response')):
            raise ValueError('Exact complete authored question and response required')
        turns = [dict(role='user', text=row['question']),
                 dict(role='assistant', text=row['response'])]
        # Keep the ORIGINAL ID so existing native and historical overlap ledgers
        # can join it; inventing new IDs could conceal a reused family/holdout.
        candidates.append(dict(id=row['id'], turns=turns,
            source_dataset='duplex-original-expanded1072-TRAIN',
            source_revision=expected_sha256,
            source_licence='original-per-record-parent-provenance-required',
            source_group_id=row['id'], source_record_ids=[row['id']],
            content_sha256=hashlib.sha256(json.dumps(turns, sort_keys=True,
                ensure_ascii=False).encode()).hexdigest(),
            original_fold='TRAIN', original_semantic_record=row,
            family_grouping_complete=False, licence_qualified=False,
            admitted_for_training=False))
    return sorted(candidates, key=lambda row: row['id'])
