"""Restore the complete approved transport on CPU; no provider or model calls.

Source must be Git-deployed separately. Existing destination artifacts and prior
receipts are never overwritten; failed staging remains available for inspection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from evals.native_gcp_destination_verify import INVENTORY_SHA256, verify
from evals.native_gcp_transfer_inventory import load, safe


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def control_archive(archive, root, rows, expected_sha):
    """Only catalogued regular files and their directories may leave this tar."""
    if archive.is_symlink() or digest(archive) != expected_sha:
        raise ValueError('Control archive SHA mismatch')
    files = {row['path']: row for row in rows}
    directories = {str(parent) for name in files for parent in Path(name).parents
                   if str(parent) != '.'}
    root.mkdir(exist_ok=False)
    seen = set()
    with tarfile.open(archive, 'r:gz') as reader:
        for member in reader:
            name = member.name
            while name.startswith('./'):
                name = name[2:]
            if name.rstrip('/') in ('', '.') and member.isdir():
                continue
            name = safe(name.rstrip('/') if member.isdir() else name)
            if member.isdir():
                if name not in directories:
                    raise ValueError('Unexpected control archive directory')
                continue
            if name in seen or name not in files or not member.isfile():
                raise ValueError('Unexpected, duplicate or nonregular control member')
            row = files[name]
            if member.size != row['bytes']:
                raise ValueError('Control member byte count mismatch')
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with reader.extractfile(member) as source, target.open('xb') as writer:
                for block in iter(lambda: source.read(1024*1024), b''):
                    writer.write(block)
                    h.update(block)
            if h.hexdigest() != row['sha256']:
                raise ValueError('Control member SHA mismatch')
            target.chmod(member.mode & 0o777)
            seen.add(name)
    if seen != set(files):
        raise ValueError('Complete control archive required')
    return len(seen)


def materialize_files(transport, root, rows):
    """Make regular hardlinks from an immutable, fully received transport."""
    if root.is_symlink() or transport.is_symlink():
        raise ValueError('Physical materialization roots required')
    root = root.resolve(strict=True)
    transport = transport.resolve(strict=True)
    for row in rows:
        name = safe(row['path'])
        source = transport/'files'/name
        if (source.is_symlink() or not source.is_file()
                or source.stat().st_size != row['bytes']
                or not source.resolve().is_relative_to(transport.resolve())):
            raise ValueError('Transport regular file escaped or changed')
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(root):
            raise ValueError('Destination parent escaped')
        parent = target.parent
        while parent != root:
            if parent.is_symlink():
                raise ValueError('Destination symlink parent')
            parent = parent.parent
        os.link(source, target, follow_symlinks=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    inventory = source/'docs/GCP-NATIVE-TRANSFER-INVENTORY-2026-10-24.json.gz'
    catalog, _ = load(inventory, INVENTORY_SHA256)
    if (args.root.is_symlink() or args.transport.is_symlink()
            or not args.root.is_dir() or not args.transport.is_dir()):
        raise ValueError('Physical existing staging directories required')
    root = args.root.resolve(strict=True)
    transport = args.transport.resolve(strict=True)
    if not transport.is_relative_to(root):
        raise ValueError('Transport must be on the destination persistent filesystem')
    if os.stat(root).st_dev != os.stat(transport).st_dev:
        raise ValueError('Same durable filesystem required')
    args.out.mkdir(exist_ok=False)
    small = [r for r in catalog['files'] if not r['path'].startswith(
        ('native-preservation-v3/', 'source-control40-e658c20/'))]
    materialize_files(transport, root, small)
    print(json.dumps(dict(phase='smaller_files_materialized', files=len(small))), flush=True)
    subprocess.run([sys.executable, '-m', 'portable.context.unpack_native_archive',
        '--manifest', str(root/'native-archive-v3/manifest.json'),
        '--manifest-sha256', '4eac15590dc9def6dfb5e34b2cea92f30d9a067fabe0940e2876b6acdcbe6dce',
        '--index', str(root/'native-archive-v3/index.json'),
        '--archive-dir', str(transport/'carriers/native'),
        '--destination', str(root/'native-preservation-v3'),
        '--report', str(args.out/'native-extracted.json')], check=True, cwd=source)
    control = catalog['archive_carriers']['control40']['containers'][0]
    prefix = 'source-control40-e658c20/'
    rows = [dict(row, path=row['path'][len(prefix):]) for row in catalog['files']
            if row['path'].startswith(prefix)]
    count = control_archive(transport/'carriers/control40'/Path(control['workspace_path']).name,
                            root/prefix, rows, control['sha256'])
    print(json.dumps(dict(phase='control_archive_materialized', files=count)), flush=True)
    report = verify(root, inventory)
    with (args.out/'destination-hashes.json').open('x') as writer:
        json.dump(report, writer, sort_keys=True, indent=2)
        writer.write('\n')
    print(json.dumps({key: report[key] for key in (
        'passed', 'verified_regular_files', 'verified_bytes', 'verified_archival_links')}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
