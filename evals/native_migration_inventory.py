"""Build or verify an explicit native-artifact migration inventory, without transfers.

Run from the repository root. Historical receipts remain immutable. The manifest
includes the preservation-ledger union and separately preserved published weights.
It deliberately does not enumerate the whole export or any credential directory.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    forbidden = {'keys', 'credentials', 'owner-conversations', 'session-token',
                 '.aws', '.ssh', '.env', '.hf_token', '.typesafe_key'}
    if not p.parts or p.is_absolute() or '..' in p.parts or '\n' in name or '\r' in name or any(
            x in forbidden or x.startswith('core-') or
            (x.startswith('core.') and x[5:].isdigit()) for x in p.parts):
        raise ValueError('Excluded migration path: ' + name)
    if p.name == 'core':
        raise ValueError('Excluded core dump path: ' + name)
    return name


def build(ledger_path, inventories, legacy_checksums, tree, additional_inventories):
    ledger = json.loads(ledger_path.read_text())
    rows = {}

    def add(name, size, sha):
        safe_name(name)
        row = {'path': name, 'bytes': size, 'sha256': sha}
        if name in rows and rows[name] != row:
            raise ValueError('Conflicting inventory: ' + name)
        rows[name] = row

    for item in ledger['inventories']:
        p = inventories / item['name']
        if digest(p) != item['sha256']:
            raise ValueError('Changed historical inventory: ' + str(p))
        for row in json.loads(p.read_text())['files']:
            add(row['path'], row['bytes'], row['sha256'])
    if (len(rows) != ledger['unique_scoped_artifact_files'] or
            sum(r['bytes'] for r in rows.values()) != ledger['unique_scoped_artifact_bytes']):
        raise ValueError('Historical union totals do not match')
    union_count = len(rows)
    # The original export predates this ledger. It contains the pinned vendor
    # checkout and raw synthetic histories required by the 28 original losses.
    # Its recorded digests, rather than fresh self-approved hashes, are retained.
    legacy_added = 0
    for line in legacy_checksums.read_text().splitlines():
        sha, name = line.split('  ', 1)
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid historical checksum')
        p = tree / safe_name(name)
        if not p.is_file():
            raise ValueError('Missing legacy export file: ' + name)
        legacy_added += name not in rows
        add(name, p.stat().st_size, sha)
    # These large published assets are outside the historical scoped union.
    native = json.loads((inventories / 'native-base-backup-verification.json').read_text())
    prefix = 'runs/cloud-export-20261016/tree/'
    for row in native['assets']:
        if not row['file'].startswith(prefix) or not row['hash_match']:
            raise ValueError('Unapproved native base receipt')
        add(row['file'][len(prefix):], row['bytes'], row['sha256'])
    persona = json.loads((inventories / 'personaplex-base-backup-verification.json').read_text())
    base = ('models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/'
            'fdaf4090a61cb315c138a1faee287ffd6c716309/')
    for row in persona['files']:
        if not row['passed']:
            raise ValueError('Unapproved PersonaPlex asset receipt')
        add(base + row['name'], row['bytes'], row['sha256'])
    additional = []
    for path in additional_inventories:
        inventory = json.loads(path.read_text())
        if inventory.get('missing'):
            raise ValueError('Incomplete additional inventory: ' + str(path))
        for row in inventory['files']:
            add(row['path'], row['bytes'], row['sha256'])
        additional.append({'path': str(path), 'sha256': digest(path)})
    links = json.loads((inventories / 'remote-runtime-links.json').read_text())
    for row in links:
        safe_name(row['path'])
    return {'schema': 'dl-native-migration-inventory-v1',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'preservation_ledger_sha256': digest(ledger_path),
            'historical_union_files': union_count,
            'legacy_export_files_added': legacy_added,
            'legacy_checksums_sha256': digest(legacy_checksums),
            'additional_inventories': additional,
            'files': [rows[k] for k in sorted(rows)], 'symlinks': links,
            'whole_remote_project_backup': False,
            'destination_gpu_restore_or_behavior_qualified': False}


def verify(manifest, tree):
    tree = tree.resolve()
    failed = []
    for row in manifest['files']:
        p = tree / safe_name(row['path'])
        # Reject parent symlinks too: the explicit inventory must stay in tree.
        within_tree = [p]
        parent = p.parent
        while parent != tree:
            within_tree.append(parent)
            parent = parent.parent
        if any(x.is_symlink() for x in within_tree):
            failed.append({'path': row['path'], 'reason': 'symlink_in_regular_path'})
        elif not p.is_file() or p.stat().st_size != row['bytes']:
            failed.append({'path': row['path'], 'reason': 'missing_or_size'})
        elif digest(p) != row['sha256']:
            failed.append({'path': row['path'], 'reason': 'sha256'})
    for row in manifest['symlinks']:
        p = tree / safe_name(row['path'])
        if not p.is_symlink() or str(p.readlink()) != row['target']:
            failed.append({'path': row['path'], 'reason': 'symlink_target'})
    return {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'files': len(manifest['files']),
            'bytes': sum(r['bytes'] for r in manifest['files']),
            'symlinks': len(manifest['symlinks']), 'failed': failed,
            'all_size_sha256_and_symlinks_match': not failed,
            'whole_remote_project_backup': False,
            'destination_gpu_restore_or_behavior_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--ledger', type=Path, default=Path('docs/LOCAL-PRESERVATION-2026-10-19.json'))
    parser.add_argument('--inventories', type=Path, default=Path('runs/local-continuation-20261019'))
    parser.add_argument('--legacy-checksums', type=Path, default=Path('runs/cloud-export-20261016/SHA256SUMS'))
    parser.add_argument('--additional-inventory', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.build:
        manifest = build(args.ledger, args.inventories, args.legacy_checksums, args.tree,
                         args.additional_inventory)
        # New version required; never overwrite an earlier inventory.
        with args.manifest.open('x') as stream:
            json.dump(manifest, stream, indent=2)
            stream.write('\n')
        names = [r['path'] for r in manifest['files']] + [r['path'] for r in manifest['symlinks']]
        args.manifest.with_suffix('.files-from.txt').write_text('\n'.join(names) + '\n')
    else:
        manifest = json.loads(args.manifest.read_text())
    report = verify(manifest, args.tree)
    report['manifest_sha256'] = digest(args.manifest)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
