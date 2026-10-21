"""Wait for one exact upload marker, then independently restore the full archive.

Runs only on an explicitly volume-backed staging Pod. No scheduling, model
training, credential handling or quality approval. An existing run is never reset.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--volume', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--latest-start-utc', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    start_limit = datetime.datetime.fromisoformat(a.latest_start_utc).timestamp()
    archive = a.volume/'native-archive-v3'
    marker = archive/'upload-pending-verification.json'
    receipt = {'schema': 'dl-native-archive-staging-watch-v1',
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'pid': __import__('os').getpid(), 'phase': 'waiting_for_complete_upload',
               'training_submitted': False, 'destination_verification_passed': False}
    with a.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    while not marker.exists():
        if time.time() >= start_limit:
            receipt['phase'] = 'start_window_expired_no_verification_claim'
            a.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
            return
        time.sleep(10)
    uploaded = json.loads(marker.read_text())
    assert uploaded['manifest_sha256'] == a.manifest_sha256
    assert uploaded['independent_destination_verification_pending']
    assert uploaded['files'] == 28470 and uploaded['bytes'] == 67764285650
    assert uploaded['symlinks'] == 8 and len(uploaded['archives']) == 35
    receipt['phase'] = 'full_destination_verification_running'
    a.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    source = Path(__file__).resolve().parent
    report = a.volume/'migration/native-full-extracted-verification-v1.json'
    proc = subprocess.run([sys.executable, str(source/'unpack_native_archive.py'),
        '--manifest', str(archive/'manifest.json'), '--manifest-sha256', a.manifest_sha256,
        '--index', str(archive/'index.json'), '--archive-dir', str(archive),
        '--destination', str(a.volume/'native-preservation-v3'), '--report', str(report),
        '--remove-derived-tars'])
    receipt['extraction_exit_code'] = proc.returncode
    if proc.returncode == 0:
        result = json.loads(report.read_text())
        assert result['all_size_sha256_and_symlinks_match']
        assert result['manifest_sha256'] == a.manifest_sha256
        assert result['files'] == 28470 and result['bytes'] == 67764285650
        assert result['symlinks'] == 8
        receipt.update(phase='full_native_archive_verified', destination_verification_passed=True,
                       verification_report=str(report))
    else:
        receipt['phase'] = 'verification_failed_preserve_partial_state_no_restart'
    a.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
