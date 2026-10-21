"""Persist explicit staged model artifacts with large writes and pinned hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--staging', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--remove-staged', action='store_true')
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    results = []
    for row in manifest['files']:
        name = PurePosixPath(row['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe artifact path')
        source = args.staging / row['path']
        if not source.is_file() or source.is_symlink() or source.stat().st_size != row['bytes']:
            raise RuntimeError('Missing staged artifact: ' + row['path'])
        target = args.destination / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.persist-v1')
        h = hashlib.sha256()
        start = time.monotonic()
        with source.open('rb') as reader, temporary.open('xb') as writer:
            for block in iter(lambda: reader.read(16 * 1024 * 1024), b''):
                h.update(block)
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if h.hexdigest() != row['sha256']:
            raise RuntimeError('Staged artifact SHA256 mismatch: ' + row['path'])
        temporary.replace(target)
        check = hashlib.sha256()
        with target.open('rb') as reader:
            for block in iter(lambda: reader.read(16 * 1024 * 1024), b''):
                check.update(block)
        if check.hexdigest() != row['sha256']:
            raise RuntimeError('Persisted artifact SHA256 mismatch: ' + row['path'])
        if args.remove_staged:
            source.unlink()
        result = {'path': row['path'], 'bytes': row['bytes'], 'sha256': h.hexdigest(),
                  'seconds': round(time.monotonic()-start, 2)}
        print(json.dumps(result), flush=True)
        results.append(result)
    for row in manifest.get('symlinks', []):
        name = PurePosixPath(row['path'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Unsafe archived symlink path')
        source = args.staging / row['path']
        target = args.destination / row['path']
        if not source.is_symlink() or str(source.readlink()) != row['target']:
            raise RuntimeError('Staged archive link differs from inventory')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if str(target.readlink()) != row['target']:
                raise RuntimeError('Existing archive link has a different target')
        elif target.exists():
            raise RuntimeError('Existing archive link path is a regular object')
        else:
            target.symlink_to(row['target'])
    with args.report.open('x') as stream:
        json.dump({'all_staged_hashes_passed': True, 'files': results,
                   'independent_destination_rehash_still_required': True}, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
