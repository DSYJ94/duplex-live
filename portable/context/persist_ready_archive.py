"""Hash and persist completed rsync files without waiting for the whole archive.

Source snapshots are preserved separately from Git-deployed execution checkouts.
The append-only receipt maps original inventory paths to their physical copies.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--staging', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--journal', type=Path, required=True)
    p.add_argument('--until-utc', required=True)
    a = p.parse_args()
    inventory = json.loads(a.manifest.read_text())
    done = {}
    if a.journal.exists():
        for line in a.journal.read_text().splitlines():
            row = json.loads(line)
            done[row['path']] = row
    deadline = datetime.datetime.fromisoformat(a.until_utc).timestamp()
    with a.journal.open('a', buffering=1) as journal:
        while time.time() < deadline:
            progress = 0
            for row in inventory['files']:
                name = PurePosixPath(row['path'])
                if name.is_absolute() or '..' in name.parts:
                    raise ValueError('Unsafe archive path')
                if row['path'] in done:
                    continue
                source = a.staging / row['path']
                if source.is_symlink() or not source.is_file() or source.stat().st_size != row['bytes']:
                    continue
                # Git and original host source metadata are archival, never deployment.
                physical = ('original-source-snapshots-v3/' + row['path']
                            if name.parts[0] in ('vendor', 'worktrees', 'repo') else row['path'])
                target = a.destination / physical
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.persist-v3')
                h = hashlib.sha256()
                with source.open('rb') as reader, temporary.open('wb') as writer:
                    for block in iter(lambda: reader.read(16 * 1024 * 1024), b''):
                        h.update(block)
                        writer.write(block)
                    writer.flush()
                    os.fsync(writer.fileno())
                if h.hexdigest() != row['sha256']:
                    raise RuntimeError('Staged SHA mismatch: ' + row['path'])
                temporary.replace(target)
                if digest(target) != row['sha256']:
                    raise RuntimeError('Destination SHA mismatch: ' + row['path'])
                result = {**row, 'physical_path': str(target), 'destination_rehashed': True}
                journal.write(json.dumps(result) + '\n')
                os.fsync(journal.fileno())
                done[row['path']] = result
                progress += 1
            print(json.dumps({'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              'verified_persistent_files': len(done),
                              'verified_persistent_bytes': sum(r['bytes'] for r in done.values()),
                              'expected_files': len(inventory['files']), 'new_files': progress}), flush=True)
            if len(done) == len(inventory['files']):
                break
            time.sleep(10)


if __name__ == '__main__':
    main()
