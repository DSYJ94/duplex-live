"""Cold-verify a completed copy before recovering failed runtime activation."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess

from evals.moshi_model_download import sha256
from portable.context.prepare_control40_replacement import metadata, safe_name, CANONICAL, BASE, LEGACY
from portable.context.prepare_native_layout import checkout, link
from portable.context.native_fresh_development import budget_guard, OWNER_LOCK


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('archive', 'original-inventory', 'workset-inventory', 'source', 'stage',
              'failed-receipt', 'out', 'cost-guard-receipt'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--pod-id', required=True)
    p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    import signal, time
    if a.out.exists() or a.deadline <= time.time():
        raise ValueError('Fresh receipt and bounded deadline required')
    budget_guard(a.cost_guard_receipt, a.pod_id, a.deadline)
    if (subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip() != a.source_commit
            or subprocess.check_output(['git', '-C', str(a.source), 'status', '--porcelain'], text=True).strip()
            or not a.stage.resolve().is_relative_to(Path('/opt')) or a.stage.is_symlink()):
        raise ValueError('Exact clean source and owned ephemeral stage required')
    failed = json.loads(a.failed_receipt.read_text())
    if (failed['phase'] != 'failed' or failed['actual_staging_completed'] is not False
            or failed['source_commit'] != a.source_commit or failed['pod_id'] != a.pod_id
            or failed['stage'] != str(a.stage) or failed['error_type'] != 'CalledProcessError'
            or not all(failed[n]['all_independent_readback_hashes_match'] for n in ('original_runs', 'workset'))):
        raise ValueError('Exact preserved failed activation after completed copies required')
    def expired(*_):
        raise TimeoutError('Bounded activation verification deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, a.deadline - time.time())
    lock = OWNER_LOCK.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('GPU occupied')
    original, workset, models, bundles = metadata(a.archive, a.original_inventory, a.workset_inventory)
    count = total = 0
    for root, rows in ((a.stage, original['files']), (a.source, workset['files']), (a.stage, models)):
        for row in rows:
            path = root / safe_name(row['path'])
            if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                    or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']):
                raise ValueError('Changed cold-verified staged file: ' + str(path))
            count += 1
            total += row['bytes']
    for row in bundles:
        name, commit = row['name'], row['commit']
        target = (a.stage / 'vendor' / (name + '-' + commit) if name == 'context-spanning'
                  else Path('/workspace') / ('control40-vendor-' + name))
        checkout(Path(row['bundle']), target, commit)
        if name != 'context-spanning':
            link(target, a.stage / 'vendor' / (name + '-' + commit))
    for name in ('tmp', 'logs'):
        link(a.archive / name, a.stage / name)
    worktrees = Path('/workspace/duplex-live/worktrees')
    if not worktrees.is_dir():
        raise ValueError('Retained historical checkouts missing')
    link(worktrees, a.stage / 'worktrees')
    link(a.stage / BASE / 'tokenizer_spm_32k_3.model', a.stage / LEGACY)
    for target in CANONICAL:
        link(a.stage, target)
    r = dict(schema='dl-native-replacement-cold-activation-v1', passed=True,
             source_commit=a.source_commit, pod_id=a.pod_id, failed_receipt_sha256=sha256(a.failed_receipt),
             cold_verified_files=count, cold_verified_bytes=total, stage=str(a.stage),
             originals_or_prior_failure_mutated=False, model_or_optimizer_run=False)
    a.out.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
