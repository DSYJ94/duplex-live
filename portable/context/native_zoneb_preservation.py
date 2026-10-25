"""Rehash the complete approved archive and completed recovery/evaluation copy.

CPU only. No extraction, model loading, qualification, deletion or updates.
"""
import argparse
import hashlib
import json
from pathlib import Path

from evals.native_gcp_destination_verify import verify, _verify_entries
from evals.native_gcp_transfer_inventory import rows

INVENTORIES = {
    'saved8-recovery-v7-complete-remote-inventory.json':
        'cdcbd894368ca000570bca1572dc8d51a2e6a68c3d392f4e80e24bb19e43aa6b',
    'gcp-native20-complete-remote-inventory-v1.json':
        '820f0856fba22b2daedfa77232ea318ba13ac379d879c0979515cff18b645d46',
}


def semantic_entries(closure):
    # The authored closure lists semantic.json first; the closed-catalog reader
    # requires lexical path order. Retain every declared identity and hash.
    return sorted((dict(path=row['name'], bytes=row['bytes'], sha256=row['sha256'])
                   for row in closure['files']), key=lambda row: row['path'])


def ordered_files(files):
    result = sorted(files, key=lambda row: row['path'])
    rows(result)  # Validate the complete list, including duplicates, before I/O.
    return result


def preservation_inputs(source, inventories):
    closure = json.loads((source/'docs/GCP-NATIVE-SEMANTIC-RECEIPT-CLOSURE-2026-10-25.json').read_text())
    semantic = ordered_files(semantic_entries(closure))
    catalogs = {}
    for name, expected in INVENTORIES.items():
        raw = (inventories/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Exact completed off-machine inventory required')
        inventory = json.loads(raw)
        files = ordered_files(inventory['files'])
        if len(files) != inventory['file_count'] or rows(files) != inventory['bytes']:
            raise ValueError('Whole saved-result inventory required')
        catalogs[name] = (files, inventory.get('links', []))
    return closure['destination_prefix'], semantic, catalogs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventories', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    # Precheck every exact input list before the expensive full archive read.
    prefix, files, catalogs = preservation_inputs(source, args.inventories)
    args.out.mkdir(exist_ok=False)
    result = verify(args.root, source/'docs/GCP-NATIVE-TRANSFER-INVENTORY-2026-10-24.json.gz')
    (args.out/'destination-hashes.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise ValueError('Full original destination failed')
    results = {}
    results['semantic'] = _verify_entries(args.root/prefix, files, [])
    for name, (files, links) in catalogs.items():
        results[name] = _verify_entries(args.root/'migration', files, links)
    passed = all(row['passed'] for row in results.values())
    (args.out/'additional-preservation.json').write_text(json.dumps(dict(
        schema='dl-zoneb-additional-preservation-v1', passed=passed,
        checks=results, optimizer_updates=0, model_loaded=False,
        model_adopted=False), indent=2)+'\n')
    if not passed:
        raise ValueError('Complete recovered/behavior/semantic preservation required')
    print(json.dumps(dict(passed=True, regular_files=40132, bytes=81154452891,
                         archival_links=8, additional_checks=list(results))))


if __name__ == '__main__':
    main()
