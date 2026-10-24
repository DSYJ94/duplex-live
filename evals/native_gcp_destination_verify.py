"""Hash every closed-catalog destination entry without following archival links.

Run on the destination after extraction. This does not transfer data, authorize
compute, establish provider billing, or qualify a model. No subset mode exists.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import stat

from evals.native_gcp_transfer_inventory import load, rows, safe


INVENTORY_SHA256 = '94b42bae0152e8150ba7842146f66711b13df3b8ece9125bcb5db92a761196fb'
SCHEMA = 'dl-native-gcp-destination-hashes-v1'
CHUNK_BYTES = 1024 * 1024


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def _parent(root_fd, name):
    """Open each parent relative to the trusted root; refuse symlink traversal."""
    parts = PurePosixPath(safe(name)).parts
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                              dir_fd=current)
            os.close(current)
            current = next_fd
        yield current, parts[-1]
    finally:
        os.close(current)


def _file(root_fd, row):
    with _parent(root_fd, row['path']) as (parent_fd, name):
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('regular_file_required')
        if before.st_size != row['bytes']:
            raise ValueError('byte_size_mismatch')
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=parent_fd)
        with os.fdopen(fd, 'rb') as reader:
            opened = os.fstat(reader.fileno())
            if _identity(before) != _identity(opened):
                raise ValueError('file_changed_before_hash')
            digest = hashlib.sha256()
            count = 0
            while True:
                chunk = reader.read(CHUNK_BYTES)
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
                if count > row['bytes']:
                    raise ValueError('file_grew_during_hash')
            after = os.fstat(reader.fileno())
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (_identity(opened) != _identity(after)
                or _identity(after) != _identity(current)):
            raise ValueError('file_changed_during_hash')
        if count != row['bytes']:
            raise ValueError('byte_size_mismatch')
        if digest.hexdigest() != row['sha256']:
            raise ValueError('sha256_mismatch')
        return count


def _link(root_fd, row):
    with _parent(root_fd, row['path']) as (parent_fd, name):
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISLNK(before.st_mode):
            raise ValueError('archival_symlink_required')
        if os.readlink(name, dir_fd=parent_fd) != row['target']:
            raise ValueError('archival_link_target_mismatch')
        if _identity(before) != _identity(os.stat(name, dir_fd=parent_fd,
                                                follow_symlinks=False)):
            raise ValueError('archival_link_changed')


def _verify_entries(root, files, links):
    """Streaming hash worker; production entry point always pins the full catalog."""
    expected_bytes = rows(files)
    file_paths = {row['path'] for row in files}
    link_paths = [safe(row['path']) for row in links]
    if len(set(link_paths)) != len(links) or file_paths.intersection(link_paths):
        raise ValueError('Duplicate or conflicting archival link')
    root = Path(root).absolute()
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError('Physical destination directory required')
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    failures = []
    verified_files = verified_links = verified_bytes = 0
    try:
        if _identity(info) != _identity(os.fstat(root_fd)):
            raise ValueError('Destination root changed')
        for kind, entries, checker in (('file', files, _file), ('link', links, _link)):
            for row in entries:
                try:
                    size = checker(root_fd, row)
                    if kind == 'file':
                        verified_files += 1
                        verified_bytes += size
                    else:
                        verified_links += 1
                except (OSError, ValueError) as error:
                    # Approved artifact paths only; never include file contents or
                    # arbitrary operating-system exception strings in a receipt.
                    reason = str(error) if isinstance(error, ValueError) else type(error).__name__
                    failures.append(dict(path=row['path'], kind=kind, reason=reason))
        final_root = root.lstat()
        if ((final_root.st_dev, final_root.st_ino)
                != (info.st_dev, info.st_ino) or not stat.S_ISDIR(final_root.st_mode)):
            raise ValueError('Destination root replaced')
    finally:
        os.close(root_fd)
    passed = (not failures and verified_files == len(files)
              and verified_links == len(links) and verified_bytes == expected_bytes)
    return dict(passed=passed, all_catalog_entries_verified=passed,
                expected_regular_files=len(files), verified_regular_files=verified_files,
                expected_bytes=expected_bytes, verified_bytes=verified_bytes,
                expected_archival_links=len(links), verified_archival_links=verified_links,
                failures=failures)


def verify(root, inventory):
    catalog, _ = load(inventory, INVENTORY_SHA256)
    started = datetime.now(timezone.utc).isoformat()
    result = _verify_entries(root, catalog['files'], catalog['archived_symlinks'])
    return dict(schema=SCHEMA, inventory_sha256=INVENTORY_SHA256,
                destination_project=catalog['destination_project'],
                physical_root=str(Path(root).resolve(strict=True)),
                hostname=socket.gethostname(), started_utc=started,
                finished_utc=datetime.now(timezone.utc).isoformat(),
                training_or_compute_authorized=False, model_qualified=False,
                unrelated_destination_files_examined=False, **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True,
                        help='New receipt path; existing files are never overwritten')
    args = parser.parse_args()
    # Refuse an existing receipt before reading tens of gigabytes.
    with args.report.open('x', encoding='utf-8') as writer:
        os.fchmod(writer.fileno(), 0o600)
        report = verify(args.root, args.inventory)
        json.dump(report, writer, sort_keys=True, indent=2)
        writer.write('\n')
        writer.flush()
        os.fsync(writer.fileno())
    print(json.dumps({key: report[key] for key in (
        'schema', 'passed', 'verified_regular_files', 'verified_bytes',
        'verified_archival_links', 'inventory_sha256')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
