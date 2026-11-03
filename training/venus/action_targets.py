"""Read-only audit of complete native action targets; no model or optimizer.

Separate measurements expose argument/control errors hidden by scene averages.
Masks refer to original labels, never rewrite words, histories, folds or caches.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import torch


CONTROL_METRICS=('native_listen','native_speak','native_turn_end','native_chunk_end','native_chunk_tts_end')

NAMES = ('<|listen|>', '<|speak|>', '<|turn_eos|>', '<|chunk_eos|>',
         '<|chunk_tts_eos|>', '<delegate>', '</delegate>')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_ids(config):
    added = {value['content']: int(key) for key, value in config['added_tokens_decoder'].items()}
    result = {name: added[name] for name in NAMES}
    if len(set(result.values())) != len(result):
        raise ValueError('Distinct exact native control token identities required')
    return result


def masks(targets, decision_mask, ids):
    """Validate unit framing and complete actions, including multi-unit payloads."""
    values = targets.tolist()
    if targets.ndim != 1 or decision_mask.shape != targets.shape or decision_mask.dtype != torch.bool:
        raise ValueError('Original one-dimensional labelled decision mask required')
    control = {ids[name] for name in NAMES[:5]}
    expected = torch.tensor([value in control for value in values], dtype=torch.bool)
    if not torch.equal(expected, decision_mask.cpu()):
        raise ValueError('Original decision labels differ from pinned native token identities')
    frame = torch.zeros(len(values), dtype=torch.bool)
    action = torch.zeros_like(frame)
    argument = torch.zeros_like(frame)
    spans, opened, in_unit = [], None, False
    for index, value in enumerate(values):
        if value in (ids['<|listen|>'], ids['<|speak|>']):
            if in_unit or (opened is not None and value == ids['<|listen|>']):
                raise ValueError('Native decision interrupts an unfinished unit/action')
            frame[index] = True
            in_unit = value == ids['<|speak|>']
        elif value == ids['<|chunk_eos|>']:
            if not in_unit:
                raise ValueError('Native chunk end without a speaking unit')
            in_unit = False
        elif not in_unit:
            raise ValueError('Label outside a native speaking unit')
        if value == ids['<delegate>']:
            if opened is not None:
                raise ValueError('Nested native action')
            opened = index
        if opened is not None and value not in control:
            action[index] = True
            argument[index] = value not in (ids['<delegate>'], ids['</delegate>'])
        if value == ids['</delegate>']:
            if opened is None or not argument[opened:index].any():
                raise ValueError('Unmatched or empty native action')
            spans.append([opened, index + 1])
            opened = None
    if in_unit or opened is not None or not frame.any():
        raise ValueError('Incomplete native unit/action or no frame decisions')
    return dict(frame_decision=frame, action=action, argument=argument), spans


def metrics(logits, targets, selected, losses=None, *, control_token_ids=None):
    """Teacher-forced token measurements; never call these native task success."""
    if losses is None:
        losses = torch.nn.functional.cross_entropy(logits, targets.to(logits.device), reduction='none')
    predicted = logits.argmax(-1)
    result = {}
    for name, mask in selected.items():
        mask = mask.to(logits.device)
        count = int(mask.sum())
        result[name + '_ce'] = float(losses[mask].mean().detach()) if count else None
        result[name + '_token_accuracy'] = float((predicted[mask] == targets.to(logits.device)[mask]).float().mean()) if count else None
        result[name + '_tokens'] = count
    if control_token_ids is not None:
        target=targets.to(logits.device)
        control=torch.stack([target==control_token_ids[name] for name in NAMES[:5]])
        counts=control.sum(1);denominator=counts.clamp_min(1).float()
        # One small host transfer for the five classes; detached observations
        # do not add a loss term or affect any existing gradient/mask/weight.
        ce=(control*losses.detach().float().unsqueeze(0)).sum(1)/denominator
        accuracy=(control*(predicted==target).unsqueeze(0)).sum(1)/denominator
        values=torch.stack((counts.float(),ce,accuracy),dim=1).detach().cpu().tolist()
        for name,(count,loss,accuracy) in zip(CONTROL_METRICS,values):
            result[name+'_tokens']=int(count)
            result[name+'_ce']=loss if count else None
            result[name+'_token_accuracy']=accuracy if count else None
    return result


def audit(timelines, cache, prefix, tokenizer_path, config_path):
    from tokenizers import Tokenizer
    from .native_cache import schedule
    timelines, cache, prefix = map(Path, (timelines, cache, prefix))
    source = json.loads((timelines / 'manifest.json').read_text())
    native = json.loads((cache / 'manifest.json').read_text())
    suffix = json.loads((prefix / 'manifest.json').read_text())
    if len(native['scenes']) != 268 or native['proof_only'] or suffix['proof_only']:
        raise ValueError('Complete existing 216 TRAIN / 52 DEV caches required')
    if native['source_manifest_sha256'] != digest(timelines / 'manifest.json') or suffix['source_manifest_sha256'] != native['source_manifest_sha256']:
        raise ValueError('Exact approved source manifest required')
    if suffix['source_native_cache_sha256'] != digest(cache / 'manifest.json'):
        raise ValueError('Prefix/native target lineage differs')
    ids = token_ids(json.loads(Path(config_path).read_text()))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for name, ident in ids.items():
        if tokenizer.encode(name, add_special_tokens=False).ids != [ident]:
            raise ValueError('Tokenizer JSON/config native identities differ')
    encoder = SimpleNamespace(encode=lambda text, **kw: tokenizer.encode(text, add_special_tokens=False).ids)
    duplex = SimpleNamespace(tokenizer=encoder, listen_token_id=ids[NAMES[0]], speak_token_id=ids[NAMES[1]],
                             turn_eos_token_id=ids[NAMES[2]], chunk_eos_token_id=ids[NAMES[3]])
    rows, groups = [], defaultdict(lambda: defaultdict(int))
    by_source = {r['id']: r for r in source['scenes'] if r['fold'] in ('TRAIN', 'DEV')}
    by_prefix = {r['id']: r for r in suffix['scenes']}
    if len(by_source) != 268 or len(by_prefix) != 268 or len({r['id'] for r in native['scenes']}) != 268:
        raise ValueError('Complete unique source/cache identities required')
    for row in native['scenes']:
        if row['fold'] not in ('TRAIN', 'DEV'):
            raise ValueError('HOLDOUT is excluded from optimizer diagnostics')
        ident = row['id']
        original, frozen = by_source[ident], by_prefix[ident]
        if any(r['fold'] != row['fold'] or r['family'] != row['family'] for r in (original, frozen)):
            raise ValueError('Fold/family drift')
        path = cache / row['file']
        if digest(path) != row['sha256']:
            raise ValueError('Original cache hash changed: ' + ident)
        ex = torch.load(path, map_location='cpu', weights_only=False)
        if any(ex[key] != row[key] for key in ('id', 'fold', 'family')):
            raise ValueError('Original example identity differs: ' + ident)
        timeline_path = timelines / ident / 'timeline.json'
        timeline = json.loads(timeline_path.read_text())
        timeline_hash = hashlib.sha256(json.dumps(timeline, sort_keys=True).encode()).hexdigest()
        if timeline_hash != ex['timeline_sha256'] or digest(timeline_path) != original['timeline_sha256']:
            raise ValueError('Original teacher timeline differs: ' + ident)
        scheduled = schedule(duplex, timeline)
        expected = [value for frame in range(timeline['frames']) for value in scheduled.get(frame, [duplex.listen_token_id])]
        if ex['targets'].tolist() != expected:
            raise ValueError('Cached labels differ from exact complete teacher schedule: ' + ident)
        selected, spans = masks(ex['targets'], ex['decision_mask'], ids)
        payloads = [tokenizer.decode(ex['targets'][start:end][selected['argument'][start:end]].tolist(), skip_special_tokens=False) for start, end in spans]
        desired = [a['text'] for a in timeline['actions'] if a['kind'] == 'delegate']
        if payloads != desired:
            raise ValueError('Complete action words differ: ' + ident)
        for payload in desired:
            if payload.startswith('lookup '):
                key = payload[len('lookup '):]
                if not key.isalpha() or not any(key in re.findall(r'[a-z]+', turn['text'].lower()) for turn in timeline['user_turns']):
                    raise ValueError('Teacher lookup key not explicit in user words: ' + ident)
            elif payload != 'cancel pending':
                raise ValueError('Unexpected closed-world action: ' + ident)
        item = dict(id=ident, fold=row['fold'], family=row['family'], native_sha256=row['sha256'],
                    prefix_sha256=frozen['sha256'], timeline_canonical_sha256=timeline_hash,
                    timeline_file_sha256=original['timeline_sha256'],
                    target_sha256=hashlib.sha256(json.dumps(expected, separators=(',', ':')).encode()).hexdigest(),
                    labels=len(expected), original_text_tokens=int((~ex['decision_mask']).sum()), action_spans=spans,
                    indices={name: mask.nonzero().flatten().tolist() for name, mask in selected.items()},
                    actions=desired, exact_teacher_schedule=True)
        rows.append(item)
        group = groups[row['fold'] + '/' + row['family']]
        group['scenes'] += 1
        group['actions'] += len(spans)
        for name, mask in selected.items():
            group[name + '_tokens'] += int(mask.sum())
        group['original_text_tokens'] += item['original_text_tokens']
    if sum(r['fold'] == 'TRAIN' for r in rows) != 216 or sum(r['fold'] == 'DEV' for r in rows) != 52:
        raise ValueError('All original folds required; no passing subset')
    return dict(schema='dl-native-action-target-index-v1', complete=True, rows=rows, groups=dict(groups),
                source_manifest_sha256=digest(timelines/'manifest.json'), native_manifest_sha256=digest(cache/'manifest.json'),
                prefix_manifest_sha256=digest(prefix/'manifest.json'), tokenizer_sha256=digest(tokenizer_path),
                tokenizer_config_sha256=digest(config_path), token_ids=ids, optimizer_updates=0,
                words_audio_folds_receipts_changed=False, native_behavior_qualified=False,
                prefix_file_hashes_recomputed_this_audit=False,
                limitation='Prefix hashes bind the previously verified full backup inventory; native target files are independently rehashed here. Correct teacher labels do not establish exposure/generalization, objective sufficiency or native behavior.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('timelines', 'cache', 'prefix', 'tokenizer', 'config', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    result = audit(args.timelines, args.cache, args.prefix, args.tokenizer, args.config)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('complete', 'groups', 'optimizer_updates')}))


if __name__ == '__main__':
    main()
