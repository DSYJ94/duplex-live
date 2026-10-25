"""Hash closed diagnostic files and preserve only the pinned host-driver links.

Never follow symbolic links. This is artifact preservation, not model approval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


DRIVER_LINKS = {
    'measurement/driver/libcuda.so': '/usr/lib/x86_64-linux-gnu/libcuda.so.580.178.04',
    'measurement/driver/libcuda.so.1': '/usr/lib/x86_64-linux-gnu/libcuda.so.580.178.04',
}


def inventory(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('A real closed diagnostic directory is required')
    rows = []
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            target = os.readlink(path)
            if DRIVER_LINKS.get(name) != target:
                raise ValueError('Unapproved diagnostic link: '+name)
            rows.append(dict(path=name, kind='symlink', target=target))
        elif stat.S_ISREG(before.st_mode):
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = path.lstat()
            identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                      value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
            if identity(before) != identity(after):
                raise ValueError('Diagnostic file changed during hashing: '+name)
            rows.append(dict(path=name, kind='file', bytes=after.st_size, sha256=digest))
        elif not stat.S_ISDIR(before.st_mode):
            raise ValueError('Unexpected diagnostic file type: '+name)
    return rows


def verify(root, expected):
    actual = inventory(root)
    if actual != expected:
        raise ValueError('Complete diagnostic file hashes or link targets differ')
    return dict(regular_files=sum(row['kind'] == 'file' for row in actual),
                regular_bytes=sum(row.get('bytes', 0) for row in actual),
                symlinks=sum(row['kind'] == 'symlink' for row in actual),
                all_regular_hashes_and_link_targets_match=True,
                model_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2))


if __name__ == '__main__':
    main()
