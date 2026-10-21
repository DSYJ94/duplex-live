"""Restore verified native archive members to a distinct archival tree.

Git metadata is preserved as historical data. Execution source must be Git-deployed
separately. Derived tar containers can be removed only after every member in that
container has passed destination rehash; original artifacts are never deleted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.native_migration_inventory import digest, safe_name, verify
from portable.context.s3_native_archive import inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--archive-dir', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--remove-derived-tars', action='store_true')
    a = p.parse_args()
    data = inventory(a.manifest, a.manifest_sha256)
    index = json.loads(a.index.read_text())
    assert index['manifest_sha256'] == a.manifest_sha256
    files = {r['path']: r for r in data['files']}
    links = {r['path']: r for r in data['symlinks']}
    listed = [n for r in index['archives'] for n in r['files']]
    listed_links = [n for r in index['archives'] for n in r['symlinks']]
    assert len(listed) == len(set(listed)) and set(listed) == set(files)
    assert len(listed_links) == len(set(listed_links)) and set(listed_links) == set(links)
    a.destination.mkdir(parents=True, exist_ok=False)
    root = a.destination.resolve()
    completed = []
    for spec in index['archives']:
        path = a.archive_dir / safe_name(spec['name'])
        assert path.is_file() and path.stat().st_size == spec['bytes']
        assert digest(path) == spec['sha256']
        seen, seen_links = set(), set()
        with tarfile.open(path, mode='r') as archive:
            for member in archive:
                name = safe_name(member.name)
                assert name not in seen and name not in seen_links
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                assert target.parent.resolve().is_relative_to(root)
                parent = target.parent
                while parent != root:
                    assert not parent.is_symlink()
                    parent = parent.parent
                if member.isfile() and name in spec['files']:
                    row = files[name]
                    assert member.size == row['bytes']
                    h = hashlib.sha256()
                    with archive.extractfile(member) as reader, target.open('xb') as writer:
                        for block in iter(lambda: reader.read(16*1024*1024), b''):
                            writer.write(block)
                            h.update(block)
                        writer.flush()
                        os.fsync(writer.fileno())
                    assert h.hexdigest() == row['sha256']
                    assert digest(target) == row['sha256']
                    target.chmod(member.mode & 0o777)
                    seen.add(name)
                elif member.issym() and name in spec['symlinks']:
                    assert member.linkname == links[name]['target']
                    target.symlink_to(member.linkname)
                    seen_links.add(name)
                else:
                    raise ValueError('Unexpected archived member: ' + name)
        assert seen == set(spec['files']) and seen_links == set(spec['symlinks'])
        completed.append({'archive': spec['name'], 'files': len(seen),
                          'links': len(seen_links), 'all_destination_hashes_passed': True})
        print(json.dumps(completed[-1]), flush=True)
        if a.remove_derived_tars:
            path.unlink()
    result = verify(data, root)
    assert result['all_size_sha256_and_symlinks_match']
    result.update(schema='dl-native-extracted-archive-verification-v1',
                  manifest_sha256=a.manifest_sha256, archive_containers=completed,
                  historical_source_not_deployed=True)
    with a.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'archive_containers'}))


if __name__ == '__main__':
    main()
