"""Audit preserved reasoning folds without moving or admitting any example."""
from collections import Counter, defaultdict
import json

from .broad_corpus_overlap import key, tokens
from .reasoning_language_source import build


def audit_records(records):
    if (not records or len({r['id'] for r in records}) != len(records)
            or any(r.get('fold') not in ('TRAIN', 'DEV')
                   or not isinstance(r.get('question'), str) or not r['question'].strip()
                   for r in records)):
        raise ValueError('Distinct original IDs, folds and complete questions required')
    result = {}
    for mode in ('literal', 'numeric_template'):
        groups = defaultdict(list)
        for row in records:
            words = tokens(row['question'], numeric_template=mode == 'numeric_template')
            if not words:
                raise ValueError('Empty normalized question cannot define a family')
            groups[key(words)].append(row)
        cross = []
        for family, rows in sorted(groups.items()):
            if {r['fold'] for r in rows} != {'TRAIN', 'DEV'}:
                continue
            cross.append(dict(input_family_sha256=family,
                TRAIN_ids=sorted(r['id'] for r in rows if r['fold'] == 'TRAIN'),
                DEV_ids=sorted(r['id'] for r in rows if r['fold'] == 'DEV')))
        result[mode] = dict(cross_fold_groups=cross, crossing_groups=len(cross),
            TRAIN_affected=sum(len(g['TRAIN_ids']) for g in cross),
            DEV_affected=sum(len(g['DEV_ids']) for g in cross))
    return dict(original_folds=dict(Counter(r['fold'] for r in records)), methods=result,
        folds_changed=False, examples_removed=False, training_admitted=False,
        semantic_paraphrase_audit_complete=False,
        word_number_normalization_performed=False,
        interpretation='Exact/template overlap audit only. Existing DEV results must be stratified '
            'by familiar-template status; fresh family-disjoint evaluation is required to establish '
            'generalization. Preserve all prior scores and original folds.')


def audit(plan_bytes, expected_sha256):
    build(plan_bytes, expected_sha256)  # Exact whole1072 source/fold/word validation.
    return dict(audit_records(json.loads(plan_bytes)['exact_semantic_records']),
                source_plan_sha256=expected_sha256)
