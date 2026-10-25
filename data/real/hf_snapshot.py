"""Acquire pinned, ungated public datasets or model files without credentials.

Streams and verifies each file against HF LFS SHA256 or Git blob SHA1. This
acquires candidates only; it never loads models, converts data, changes folds or
starts training. Inventories select explicit files, never a whole moving repo.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


def digests(path, size):
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(('blob ' + str(size) + '\0').encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024 * 1024), b''):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def verify(row, actual_size, sha256, blob):
    if actual_size != row['size']:
        raise ValueError('Downloaded size differs from pinned inventory')
    if 'lfs' in row:
        if sha256 != row['lfs']['oid']:
            raise ValueError('Downloaded LFS SHA256 differs from upstream')
    elif blob != row['oid']:
        raise ValueError('Downloaded Git blob SHA1 differs from upstream')


def acquire(specs, cache, receipt):
    if (sum(r['size'] for s in specs for r in s['files']) > 21_000_000_000
            or any(s['gated'] is not False for s in specs)
            or any(s.get('repository_type', 'dataset') not in ('dataset', 'model') for s in specs)):
        raise ValueError('Only bounded21GB ungated public acquisition is supported')
    cache = cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    lock = (cache / 'acquisition.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = []
    for spec in specs:
        folder = cache / (spec['repo'].replace('/', '--') + '@' + spec['revision'][:12])
        folder.mkdir(exist_ok=True)
        if len(spec['revision']) != 40 or len({r['path'] for r in spec['files']}) != len(spec['files']):
            raise ValueError('Pinned distinct file inventory required')
        for row in spec['files']:
            relative = PurePosixPath(row['path'])
            if relative.is_absolute() or '..' in relative.parts or '\\' in row['path']:
                raise ValueError('Unsafe public-data path')
            path = folder / row['path']
            if folder.resolve() not in path.resolve().parents:
                raise ValueError('Public-data file escapes its snapshot directory')
            tasks.append((spec, row, path))
    started = time.time()
    records = []
    def download(task):
        spec, row, path = task
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            sha256, blob = digests(path, row['size'])
            verify(row, path.stat().st_size, sha256, blob)
        else:
            prefix = 'datasets/' if spec.get('repository_type', 'dataset') == 'dataset' else ''
            url = ('https://huggingface.co/' + prefix + spec['repo'] + '/resolve/'
                   + spec['revision'] + '/' + quote(row['path']) + '?download=true')
            temporary = path.with_name(path.name + '.part')
            for attempt in range(3):
                try:
                    sha = hashlib.sha256()
                    git = hashlib.sha1(('blob ' + str(row['size']) + '\0').encode())
                    size = 0
                    request = Request(url, headers={'User-Agent':'duplex-live-public-data/1.0'})
                    with urlopen(request, timeout=45) as source, temporary.open('wb') as target:
                        for chunk in iter(lambda:source.read(1024 * 1024), b''):
                            size += len(chunk)
                            if size > row['size']:
                                raise ValueError('Source exceeds pinned file size')
                            target.write(chunk)
                            sha.update(chunk)
                            git.update(chunk)
                    sha256, blob = sha.hexdigest(), git.hexdigest()
                    verify(row, size, sha256, blob)
                    temporary.replace(path)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        return dict(repo=spec['repo'], revision=spec['revision'], path=str(path),
                    bytes=row['size'], sha256=sha256, upstream_hash_verified=True)
    journal = receipt.with_suffix('.jsonl')
    with ThreadPoolExecutor(max_workers=2) as pool, journal.open('a') as stream:
        futures = [pool.submit(download, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if len(records) % 100 == 0 or len(records) == len(tasks):
                print(json.dumps(dict(files=len(records), expected_files=len(tasks),
                    verified_bytes=sum(r['bytes'] for r in records), elapsed_s=time.time()-started)), flush=True)
    result = dict(schema='dl-public-data-acquisition-v1', complete=True,
        files=len(records), bytes=sum(r['bytes'] for r in records), cache=str(cache),
        elapsed_s=time.time()-started, inventories=[dict(repo=s['repo'], revision=s['revision'],
            repository_type=s.get('repository_type', 'dataset'),
            declared_license=s['license'], files=len(s['files']), bytes=sum(r['size'] for r in s['files'])) for s in specs],
        journal_sha256=hashlib.sha256(journal.read_bytes()).hexdigest(),
        all_upstream_hashes_verified=True, owner_conversations_used=False,
        credentials_used=False, training_approved=False, model_adopted=False)
    temporary = receipt.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(receipt)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(json.loads(args.inventory.read_text()), args.cache, args.receipt)), flush=True)
