"""Pinned public corpus staging only: no model, synthesis or data admission."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.request

SOURCES = (
    dict(dataset='OpenAssistant/oasst1', revision='fdf72ae0827c1cda404aff25b6603abec9e3399b',
         licence='Apache-2.0', files=('README.md', 'LICENSE',
             '2023-04-12_oasst_ready.trees.jsonl.gz',
             'data/train-00000-of-00001-b42a775f407cee45.parquet',
             'data/validation-00000-of-00001-134b8fd0c89408b6.parquet')),
    dict(dataset='databricks/databricks-dolly-15k', revision='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',
         licence='CC-BY-SA-3.0', files=('README.md', 'databricks-dolly-15k.jsonl')),
)


def inspect_jsonl(path, *, compressed=False, fields=()):
    opener = gzip.open if compressed else open
    count = 0
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict) or not set(fields) <= set(value):
                raise ValueError('Every original JSONL record must retain its schema')
            count += 1
    if not count:
        raise ValueError('Empty source corpus is not a completed download')
    return count


def acquire(root, *, opener=urllib.request.urlopen):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    inventory = []
    for source in SOURCES:
        folder = root / source['dataset'].replace('/', '--') / source['revision']
        folder.mkdir(parents=True)
        for relative in source['files']:
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            url = 'https://huggingface.co/datasets/%s/resolve/%s/%s' % (source['dataset'], source['revision'], relative)
            temporary = target.with_name(target.name + '.partial')
            digest = hashlib.sha256()
            size = 0
            with opener(url, timeout=45) as response, temporary.open('xb') as output:
                linked_etag = response.headers.get('X-Linked-Etag')
                etag = linked_etag or response.headers.get('ETag')
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 150 * 1024 ** 2:
                        raise ValueError('Source file exceeded declared150MiB intake bound')
                    digest.update(chunk)
                    output.write(chunk)
            if not size:
                raise ValueError('Public source download returned no bytes')
            actual = digest.hexdigest()
            # HF LFS linked ETags are SHA256; ordinary Git ETags need not be.
            if linked_etag and len(linked_etag.strip('"')) == 64 and linked_etag.strip('"').lower() != actual:
                raise ValueError('Published source SHA256 differs from downloaded bytes')
            metadata = dict(dataset=source['dataset'], revision=source['revision'],
                            licence=source['licence'], path=str(target), url=url,
                            bytes=size, sha256=actual, source_etag=etag,
                            published_SHA256_verified=bool(linked_etag and len(linked_etag.strip('"')) == 64),
                            official_split='validation' if relative.startswith('data/validation-') else
                                           'train' if relative.startswith('data/train-') else 'unassigned')
            if relative.endswith('.jsonl'):
                metadata['original_records'] = inspect_jsonl(temporary, fields=('instruction', 'context', 'response', 'category'))
            elif relative.endswith('.jsonl.gz'):
                metadata['original_records'] = inspect_jsonl(temporary, compressed=True)
                metadata['official_train_membership_not_yet_joined'] = True
            elif relative.endswith('.parquet'):
                with temporary.open('rb') as stream:
                    head = stream.read(4)
                    stream.seek(-4, 2)
                    tail = stream.read(4)
                if head != b'PAR1' or tail != b'PAR1':
                    raise ValueError('Complete original Parquet source required')
            temporary.rename(target)
            inventory.append(metadata)
            print(json.dumps(dict(event='public_source_staged', dataset=source['dataset'],
                                  file=relative, bytes=size, sha256=actual)), flush=True)
    result = dict(status='PINNED_ORIGINALS_STAGED_NOT_ADMITTED', sources=inventory,
                  elapsed_seconds=time.time() - started, source_licences_kept_separate=True,
                  eligibility_filters_complete=False, decontamination_complete=False,
                  semantic_qualification_complete=False, native_audio_created=False,
                  training_started=False, model_API_calls=0, purchases=0)
    with (root / 'download-inventory.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    acquire(args.out)
