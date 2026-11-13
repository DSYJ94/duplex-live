"""Preserve completed evaluation bytes on independent Lustre, without model work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .matched_archive import sha, independent_filesystems
from .matched_interrupted import DURABLE
from .matched_paths import BASE
from .matched_job import write
from evals.native_migration_inventory import safe_name

JOB = '8366795'
PRODUCER = 'ebd71281829150e855b738dd8ed4e1262e714cb5'
ROOT = DURABLE / 'runs/evaluation-recovery-v2'
DEST = BASE / 'native-moshi-evaluation-backup-v1' / ('job-' + JOB)


def inventory(root):
    rows = []
    for p in sorted(root.rglob('*')):
        name = p.relative_to(root).as_posix()
        safe_name(name)
        if any(part in {'.private', 'owner', 'conversations'} or part.endswith('.env') for part in p.relative_to(root).parts):
            raise ValueError('Private path refused')
        if p.is_symlink():
            # Preserve runtime link text without following it into system paths.
            rows.append(dict(path=name, kind='symlink', target=os.readlink(p), bytes=0))
            continue
        if p.is_file():
            before = p.stat()
            digest = sha(p)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Evaluation source changed during inventory')
            rows.append(dict(path=name, kind='file', bytes=after.st_size, sha256=digest))
    return rows


def archive(resume_stage=False):
    terminal = json.loads((ROOT / ('execution-' + JOB) / 'terminal.json').read_text())
    comparison = json.loads((ROOT / 'comparison.json').read_text())
    if terminal['actual_exit_code'] != 0 or terminal['source'] != PRODUCER or not comparison['complete_six_times32']:
        raise ValueError('Actual complete successful comparison required')
    if comparison['source'] != PRODUCER or comparison['optimizer_updates'] != 0 or len(comparison['reports']) != 6:
        raise ValueError('Exact evaluation producer and all six reports required')
    for report in comparison['reports'].values():
        path = Path(report['path'])
        path.relative_to(ROOT)
        value = json.loads(path.read_text())
        if sha(path) != report['sha256'] or not value['complete'] or len(value['cases']) != 32:
            raise ValueError('All192 report identities required')
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not independent_filesystems(ROOT, DEST.parent):
        raise ValueError('Independent filesystem required')
    rows = inventory(ROOT)
    print(json.dumps(dict(event='source_inventory_complete', files=len(rows), bytes=sum(r['bytes'] for r in rows))), flush=True)
    stage = DEST.with_name('.' + DEST.name + '-stage')
    if resume_stage:
        if not stage.is_dir() or stage.is_symlink():
            raise ValueError('Explicit existing incomplete stage required')
    else:
        stage.mkdir(exist_ok=False)
    if DEST.exists():
        raise ValueError('Existing immutable backup must never be overwritten')
    # Small batches avoid one long many-file IPC transfer; final full hashes
    # remain mandatory. Existing completed files are reused by normal rsync.
    for offset in range(0, len(rows), 2000):
        batch = rows[offset:offset + 2000]
        names = b''.join(r['path'].encode() + b'\0' for r in batch)
        result = subprocess.run(['rsync', '-ar', '--no-owner', '--no-group', '--exclude=session-token',
                                 '--timeout=180', '--from0', '--files-from=-',
                                 str(ROOT) + '/', str(stage) + '/'], input=names)
        if result.returncode:
            raise RuntimeError('Actual rsync failed in batch: ' + str(result.returncode))
        print(json.dumps(dict(event='archive_batch_copied', through=min(offset + 2000, len(rows)),
                              total=len(rows))), flush=True)
    copied = inventory(stage)
    if rows != copied:
        raise ValueError('Complete destination SHA256 inventory differs')
    write(stage / 'archive-inventory.json', dict(files=rows, source=str(ROOT), producer=PRODUCER, job_id=JOB))
    publish(stage, rows, 'fresh_full_source_destination_equality')


def publish(stage, rows, verification):
    receipt = dict(passed=True, verification=verification, source=str(ROOT), destination=str(DEST), job_id=JOB, producer=PRODUCER,
                   files=len(rows), bytes=sum(r['bytes'] for r in rows), independent_filesystem=True,
                   actual_rsync_exit_code=0, every_file_sha256_verified=True, optimizer_updates=0,
                   manifest_sha256=sha(stage / 'archive-inventory.json'), actual_unix=time.time(),
                   backup_program_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip())
    write(stage / 'archive-receipt.json', receipt)
    os.rename(stage, DEST)
    print(json.dumps(receipt), flush=True)


def finalize_verified_stage(expected_manifest_sha):
    stage = DEST.with_name('.' + DEST.name + '-stage')
    manifest = stage / 'archive-inventory.json'
    if DEST.exists() or sha(manifest) != expected_manifest_sha:
        raise ValueError('Exact unpublished post-equality manifest required')
    value = json.loads(manifest.read_text())
    if value['source'] != str(ROOT) or value['producer'] != PRODUCER or value['job_id'] != JOB:
        raise ValueError('Exact completed evaluation lineage required')
    # a6b6a591 writes this manifest only AFTER every source/destination row
    # matches. Its actual publisher then failed on Python3.6 text=True.
    # Recheck the complete destination against that full-source manifest.
    rows = value['files']
    actual = inventory(stage)
    expected_names = {r['path'] for r in rows} | {'archive-inventory.json'}
    import os
    names = set()
    for folder, dirs, files in os.walk(str(stage), followlinks=False):
        for name in files:
            names.add(str((Path(folder) / name).relative_to(stage)))
        for name in dirs:
            path = Path(folder) / name
            if path.is_symlink():
                names.add(str(path.relative_to(stage)))
    if names != expected_names or [r for r in actual if r['path'] != 'archive-inventory.json'] != rows:
        raise ValueError('Full final destination inventory differs')
    publish(stage, rows, 'prior_full_equality_manifest_plus_fresh_full_destination_verification')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-stage', action='store_true')
    parser.add_argument('--finalize-stage-manifest-sha')
    args = parser.parse_args()
    if args.finalize_stage_manifest_sha:
        finalize_verified_stage(args.finalize_stage_manifest_sha)
    else:
        archive(args.resume_stage)
