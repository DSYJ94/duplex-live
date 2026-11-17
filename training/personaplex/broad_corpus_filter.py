"""Whole-source eligibility, not semantic/audio admission or model training."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import unicodedata


def canonical(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def nodes(prompt):
    stack = [prompt]
    while stack:
        value = stack.pop()
        yield value
        stack.extend(value.get('replies', []))


def label(message, name):
    value = message.get('labels', {}).get(name, {})
    score, count = value.get('value'), value.get('count')
    if (not isinstance(score, (float, int)) or not math.isfinite(score)
            or not 0 <= score <= 1 or type(count) is not int or count < 2):
        return None
    return score


def eligible(message, train_ids):
    if (message.get('message_id') not in train_ids or message.get('lang') != 'en'
            or message.get('review_result') is not True or message.get('deleted') is not False
            or message.get('synthetic') is not False or type(message.get('review_count')) is not int
            or message['review_count'] < 2 or not isinstance(message.get('text'), str)
            or not message['text'].strip()):
        return False
    quality = label(message, 'quality')
    return quality is not None and quality >= .5 and all(label(message, name) == 0 for name in ('pii', 'spam', 'lang_mismatch'))


def candidate(dataset, revision, licence, source_group, source_ids, turns, family):
    if (not turns or len(turns) % 2 or any(t['role'] != ('user' if i % 2 == 0 else 'assistant')
                                         or not isinstance(t['text'], str) or not t['text'].strip()
                                         for i, t in enumerate(turns))):
        raise ValueError('Complete original alternating user/assistant turns required')
    normal = [dict(role=t['role'], text=canonical(t['text'])) for t in turns]
    content_sha = fingerprint(normal)
    return dict(id=dataset.replace('/', '-') + '-' + fingerprint(source_ids)[:24],
                source_dataset=dataset, source_revision=revision, source_licence=licence,
                source_group_id=source_group, source_record_ids=source_ids,
                task_family=family, domain=None, route=None, answerability=None,
                split=None, turns=turns, content_sha256=content_sha,
                prompt_context_sha256=fingerprint(normal[:-1]), target_sha256=fingerprint(normal[-1]),
                requires_length_qualified_rewrite=any(len(t['text'].split()) > 150 for t in turns),
                semantic_qualified=False, native_audio_qualified=False, admitted_for_training=False)


def oasst(tree, train_ids, validation_ids, revision):
    prompt = tree['prompt']
    all_nodes = list(nodes(prompt))
    identities = [n['message_id'] for n in all_nodes]
    if len(set(identities)) != len(identities):
        return None, 'duplicate_message_id'
    if set(identities) & validation_ids:
        return None, 'official_validation_tree'
    if prompt.get('role') != 'prompter' or not eligible(prompt, train_ids):
        return None, 'root_not_eligible'
    path = [prompt]
    while len(path) < 8:
        role = 'assistant' if path[-1]['role'] == 'prompter' else 'prompter'
        options = [r for r in path[-1].get('replies', []) if r.get('role') == role and eligible(r, train_ids)]
        if not options:
            break
        def order(message):
            rank = message.get('rank')
            rank = rank if type(rank) is int and rank >= 0 else 1000000
            return rank, -label(message, 'quality'), message['message_id']
        path.append(min(options, key=order))
    if path[-1]['role'] == 'prompter':
        path.pop()
    if len(path) < 2:
        return None, 'no_complete_reviewed_path'
    turns = [dict(role='user' if n['role'] == 'prompter' else 'assistant', text=n['text']) for n in path]
    result = candidate('OpenAssistant/oasst1', revision, 'Apache-2.0', tree['message_tree_id'],
                       [n['message_id'] for n in path], turns, 'human_conversation_path')
    result.update(official_source_split='train', validation_tree_excluded=True,
                  crowd_review_is_not_factual_certification=True,
                  selected_complete_path_prefix=True)
    return result, None


def dolly(row, index, revision):
    if (not all(isinstance(row.get(k), str) for k in ('instruction', 'context', 'response', 'category'))
            or not row['instruction'].strip() or not row['response'].strip()):
        return None, 'missing_original_fields'
    context = row['context']
    # Explicitly retain the entire supplied context; never truncate premises.
    question = row['instruction'] + ('\n\nSupplied context:\n' + context if context else '')
    group = 'context-' + fingerprint(canonical(context)) if context else 'instruction-' + fingerprint(canonical(row['instruction']))
    result = candidate('databricks/databricks-dolly-15k', revision, 'CC-BY-SA-3.0', group,
                       ['row-%05d' % index], [dict(role='user', text=question), dict(role='assistant', text=row['response'])], row['category'])
    result.update(original_instruction=row['instruction'], original_context=context,
                  official_source_split='unsplit', source_group_precision='context_hash_pending_semantic_clustering')
    return result, None


def run(raw, out):
    import pyarrow
    import pyarrow.parquet as pq
    raw, out = Path(raw), Path(out)
    inventory_path = raw / 'download-inventory.json'
    inventory = json.loads(inventory_path.read_text())
    for row in inventory['sources']:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Every complete original source must retain its pinned bytes')
    assets = {(r['dataset'], Path(r['path']).name): r for r in inventory['sources']}
    def asset(dataset, name):
        return Path(assets[(dataset, name)]['path'])
    def message_ids(path):
        values = pq.read_table(path, columns=['message_id'], use_threads=False)['message_id'].to_pylist()
        if len(values) != len(set(values)) or any(not isinstance(v, str) for v in values):
            raise ValueError('Distinct original official message IDs required')
        return set(values)
    train_ids = message_ids(asset('OpenAssistant/oasst1', 'train-00000-of-00001-b42a775f407cee45.parquet'))
    validation_ids = message_ids(asset('OpenAssistant/oasst1', 'validation-00000-of-00001-134b8fd0c89408b6.parquet'))
    if train_ids & validation_ids:
        raise ValueError('Official source message partitions overlap')
    out.mkdir(exist_ok=False)
    seen, summaries = set(), {}
    for dataset in ('OpenAssistant/oasst1', 'databricks/databricks-dolly-15k'):
        directory = out / dataset.replace('/', '--')
        directory.mkdir()
        counts = Counter()
        source = asset(dataset, '2023-04-12_oasst_ready.trees.jsonl.gz' if dataset.startswith('OpenAssistant') else 'databricks-dolly-15k.jsonl')
        opener = gzip.open if source.name.endswith('.gz') else open
        revision = assets[(dataset, source.name)]['revision']
        with opener(source, 'rt', encoding='utf-8') as incoming, (directory / 'candidates.jsonl').open('x') as output:
            for index, line in enumerate(incoming):
                row = json.loads(line)
                counts['original_records'] += 1
                value, reason = oasst(row, train_ids, validation_ids, revision) if dataset.startswith('OpenAssistant') else dolly(row, index, revision)
                if reason:
                    counts['excluded_' + reason] += 1
                    continue
                if value['content_sha256'] in seen:
                    counts['excluded_exact_duplicate'] += 1
                    continue
                seen.add(value['content_sha256'])
                output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
                counts['eligible_text_candidates'] += 1
                counts['requires_length_qualified_rewrite'] += int(value['requires_length_qualified_rewrite'])
        summaries[dataset] = dict(counts)
    result = dict(status='FIRST_PASS_TEXT_CANDIDATES_NOT_ADMITTED', sources=summaries,
                  source_inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
                  official_oasst_train_message_ids=len(train_ids), official_oasst_validation_message_ids=len(validation_ids),
                  source_licence_partitions_separate=True, pyarrow=pyarrow.__version__,
                  domain_and_routing_labels_pending=True, semantic_family_splits_pending=True,
                  inspected_regression_decontamination_pending=True,
                  semantic_qualification_complete=False, native_audio_created=False,
                  optimizer_updates=0, model_API_calls=0, admitted_for_training=False)
    (out / 'eligibility.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.raw, args.out)
