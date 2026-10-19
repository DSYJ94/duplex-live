"""Reinterleave only repaired text clocks; reuse every original acoustic/reference code.

Uses the pinned native Interleaver on CPU. No Mimi re-encoding, ARC regeneration,
acoustic gain change or optimizer occurs. All 25 cases are required.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from evals.moshi_model_download import sha256
from evals.native_old25_timing_repair import inventory, REPAIRED, ROOT, read, write

SCHEMA = 'dl-native-old25-timing-cache-v1'
VENDOR = ROOT / 'vendor/moshi-finetune-2acc879fe7c48f885a18f6cc9548bccb2674d87b'


def same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def validate_unchanged_payload(old, fresh):
    import torch
    if (old.keys() != fresh.keys() or fresh['codes'].shape != old['codes'].shape
            or fresh['codes'].dtype != old['codes'].dtype
            or not torch.equal(fresh['codes'][:, 1:], old['codes'][:, 1:])
            or any(not same(old[k], fresh[k]) for k in old if k != 'codes')):
        raise RuntimeError('Timing repair changed original acoustics/references/receipts')


def interleaver(model):
    import sentencepiece
    import subprocess
    from evals.personaplex_train import MODEL_HASHES
    commit = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '2acc879fe7c48f885a18f6cc9548bccb2674d87b':
        raise RuntimeError('Pinned native Interleaver changed')
    sys.path.insert(0, str(VENDOR))
    from finetune.data.interleaver import Interleaver
    tokenizer_file = Path(model['path']) / 'tokenizer_spm_32k_3.model'
    if sha256(tokenizer_file) != MODEL_HASHES['tokenizer_spm_32k_3.model']:
        raise RuntimeError('Native vocabulary changed')
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_file))
    return Interleaver(tokenizer, 12.5, text_padding=3, end_of_text_padding=0,
                       zero_padding=-1, keep_main_only=True, keep_and_shift=True, device='cpu'), tokenizer


def text_lane(builder, side, duration, markers):
    from evals.recovery_encode import marker_frame
    text = builder.prepare_item(side['alignments'], duration)
    expected = [marker_frame(e, 12.5) for e in side['timeline']['events'] if e['kind'] == 'marker']
    if expected != markers:
        raise RuntimeError('Timing repair changed a tool-dispatch marker')
    for frame in markers:
        if int(text[0, 0, frame]) not in (0, 3):
            raise RuntimeError('Repaired word collides with original silent control marker')
        text[0, 0, frame] = 4
    return text


def build(data, out):
    import torch
    from evals.recovery_data import validate_conversion
    from evals.recovery_encode import native_word_timing
    conversion = validate_conversion(data)
    items = inventory()
    old_receipts = {r['teacher']['cache']: read(Path(r['teacher']['cache']) / 'cache.json') for r in items}
    parent = next(iter(old_receipts.values()))
    if any(r['model'] != parent['model'] or r['frame_rate'] != 12.5 for r in old_receipts.values()):
        raise RuntimeError('Original native cache codec/reference pinning differs')
    builder, tokenizer = interleaver(parent['model'])
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, original in zip(conversion['accepted'], items):
        sid = case['scenario_id']
        cache = Path(original['teacher']['cache'])
        old_row = next(r for r in old_receipts[str(cache)]['examples'] if r['scenario_id'] == sid)
        old_file = cache / old_row['file']
        old = torch.load(old_file, map_location='cpu', weights_only=True)
        destination = out / (sid + '.pt')
        side = read(Path(case['path']).with_suffix('.json'))
        if sid in REPAIRED:
            old_side = read(Path(original['original_row']['path']).with_suffix('.json'))
            old_text = text_lane(builder, old_side, case['duration'], old['markers'])
            if not torch.equal(old_text, old['codes'][:, :1]):
                raise RuntimeError('Pinned native Interleaver does not reproduce original text lane exactly')
            fresh = dict(old, codes=old['codes'].clone())
            fresh['codes'][:, :1] = text_lane(builder, side, case['duration'], old['markers'])
            torch.save(fresh, destination)
        else:
            shutil.copyfile(old_file, destination)
            fresh = old
        timing = native_word_timing(fresh['codes'][0, 0].tolist(), side['alignments'], tokenizer.encode, 12.5)
        rows.append(dict(old_row, file=destination.name, sha256=sha256(destination), native_word_timing=timing))
    receipt = dict(schema=SCHEMA, conversion=str(data), ingestion_sha256=sha256(data / 'ingestion.json'),
        model=parent['model'], frame_rate=12.5, examples=rows, repaired_ids=sorted(REPAIRED),
        code_sha256=sha256(Path(__file__)), original_acoustics_references_receipts_preserved=True,
        optimizer_steps=0, training_approved=False,
        method='Pinned native text Interleaver only; exact original 16 acoustic lanes and frozen reference tensors')
    write(out / 'cache.json', receipt)
    validate(out, receipt)
    print(json.dumps(dict(cases=len(rows), text_clocks_repaired=4, acoustic_codes_changed=0)), flush=True)


def validate(cache, receipt):
    import torch
    from evals.recovery_data import validate_conversion
    from evals.recovery_encode import native_word_timing
    data = Path(receipt['conversion'])
    conversion = validate_conversion(data)
    items = inventory()
    if (receipt['schema'] != SCHEMA or receipt['code_sha256'] != sha256(Path(__file__))
            or receipt['ingestion_sha256'] != sha256(data / 'ingestion.json')
            or receipt['repaired_ids'] != sorted(REPAIRED) or receipt['frame_rate'] != 12.5
            or [r['scenario_id'] for r in receipt['examples']] != [r['teacher']['id'] for r in items]):
        raise RuntimeError('Repaired native cache lacks the complete fixed 25-history lineage')
    builder, tokenizer = interleaver(receipt['model'])
    for row, case, original in zip(receipt['examples'], conversion['accepted'], items):
        old_cache = Path(original['teacher']['cache'])
        old_receipt = read(old_cache / 'cache.json')
        old_row = next(r for r in old_receipt['examples'] if r['scenario_id'] == row['scenario_id'])
        if old_receipt['model'] != receipt['model']:
            raise RuntimeError('Native codec/reference model changed')
        path = cache / row['file']
        if sha256(path) != row['sha256']:
            raise RuntimeError('Repaired native cache artifact changed')
        old = torch.load(old_cache / old_row['file'], map_location='cpu', weights_only=True)
        fresh = torch.load(path, map_location='cpu', weights_only=True)
        validate_unchanged_payload(old, fresh)
        side = read(Path(case['path']).with_suffix('.json'))
        timing = native_word_timing(fresh['codes'][0, 0].tolist(), side['alignments'], tokenizer.encode, 12.5)
        expected_row = dict(old_row, file=path.name, sha256=sha256(path), native_word_timing=timing)
        if row != expected_row:
            raise RuntimeError('Repaired native cache changed undeclared row metadata')
        if row['scenario_id'] in REPAIRED:
            expected_text = text_lane(builder, side, case['duration'], old['markers'])
            if not torch.equal(expected_text, fresh['codes'][:, :1]):
                raise RuntimeError('Repaired native text differs from measured native interleaving')
        elif sha256(path) != old_row['sha256']:
            raise RuntimeError('An undiagnosed native teacher cache changed')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    build(args.data, args.out)


if __name__ == '__main__':
    main()
