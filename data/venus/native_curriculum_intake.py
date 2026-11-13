"""Versioned text-source intake for expanded native data; never auto-approve audio."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .reasoning_curriculum import normalized


def build(corpus, source_sha):
    if (corpus.get('schema') != 'dl-reasoning-text-curriculum-v1'
            or corpus.get('official_test_used') is not False
            or corpus.get('native_audio_created') is not False):
        raise ValueError('Exact text-only public-TRAIN/authored source required')
    seen_ids, seen_questions, rows = set(), set(), []
    for old in corpus['rows']:
        identity = old['id']; question = normalized(old['question'])
        if identity in seen_ids or question in seen_questions:
            raise ValueError('Duplicate source ID/question, including cross-fold overlap')
        if old['fold'] not in ('TRAIN', 'DEV') or not old['response'].strip():
            raise ValueError('Preserve original TRAIN/DEV and nonempty target')
        if old['source_kind'] not in ('authored_verified_arithmetic', 'gsm8k_official_train'):
            raise ValueError('Unqualified source kind')
        seen_ids.add(identity); seen_questions.add(question)
        group = ('arithmetic-template-' + str(old['template']) if 'template' in old
                 else 'public-train-row-' + str(old['source_row']))
        rows.append(dict(id=identity, fold=old['fold'], source_group=group,
            domain='maths', source_question=old['question'], source_response=old['response'],
            source_record=old, source_corpus_sha256=source_sha,
            routing_target=None, native_teacher_approved=False, audio_created=False))
    counts = dict(Counter(r['fold'] for r in rows))
    if counts != {'TRAIN': 1008, 'DEV': 64}:
        raise ValueError('All original1008TRAIN/64DEV required, not a subset')
    train_groups = {r['source_group'] for r in rows if r['fold'] == 'TRAIN'}
    dev_groups = {r['source_group'] for r in rows if r['fold'] == 'DEV'}
    return dict(schema='dl-expanded-native-source-intake-v1', rows=rows, counts=counts,
        source_sha256=source_sha, source_words_and_folds_preserved=True,
        shared_template_groups=sorted(train_groups & dev_groups),
        DEV_is_unseen_template_test=False, external_API_calls=0,
        training_ready=False, native_audio_qualified=False,
        coverage=dict(maths_source_rows=len(rows), ordinary_concepts=0, technical_dialogue=0,
                      direct_clarify_delegate_contrasts=0, silence_pause_interrupt=0,
                      actual_worker_lifecycle=0, natural_British_audio=0),
        next_required=['broader independently varied domain and interaction sources',
                       'calibrated direct/clarify/delegate labels, never assumed for all maths',
                       'native teacher/word/timing/audio qualification',
                       'fresh families/speakers and original evaluation exclusions',
                       'retention corpus and declared native loss/clock integration'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); raw = a.source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != a.expected_sha256:
        raise ValueError('Original prepared corpus SHA256 differs')
    value = build(json.loads(raw), digest)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'intake.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: value[k] for k in ('counts', 'source_sha256', 'coverage', 'training_ready')}))


if __name__ == '__main__':
    main()
