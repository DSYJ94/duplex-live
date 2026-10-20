"""Durable, single-submission bounded pilot chain; compatible with host Python3.6.

Every transition is recorded before releasing a held job. An uncertain submission
is never retried automatically. This chain cannot extend or adopt a checkpoint.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

BASE = Path('/projects/u6si/duplex-live')
STAGES = ('readiness', 'training', 'native10', 'speech10')


def now():
    return datetime.datetime.utcnow().isoformat() + 'Z'


def call(args, env=None):
    return subprocess.check_output(args, cwd=str(BASE), env=env,
                                   universal_newlines=True, timeout=90)


def save(path, receipt):
    with tempfile.NamedTemporaryFile(mode='w', dir=str(path.parent), delete=False) as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
        temporary = f.name
    os.replace(temporary, str(path))


def validate(receipt, stage, job):
    if stage not in STAGES or receipt['jobs'].get(stage) != job:
        raise ValueError('Unowned stage/job; no action')
    source = os.environ['DL_TASK_COMMIT']
    tree = Path(os.environ['DL_TASK_SOURCE'])
    if (receipt['source'] != source or receipt['executor'] != 'LOCAL'
            or call(['git', '-C', str(tree), 'rev-parse', 'HEAD']).strip() != source
            or call(['git', '-C', str(tree), 'status', '--porcelain']).strip()):
        raise ValueError('Immutable LOCAL source identity required')


def submit(path, receipt, action, job):
    stage, following, script, dependency = {
        'submit-native': ('training', 'native10', 'native-audio49-evaluate.slurm', 'afterok'),
        'submit-speech': ('native10', 'speech10', 'native-audio49-speech.slurm', 'afterany'),
    }[action]
    validate(receipt, stage, job)
    if following in receipt['jobs'] or following in receipt.get('submission_attempts', {}):
        raise ValueError('Transition already recorded; never duplicate a submission')
    if action == 'submit-native':
        from evals.context_native_audio49_training import verify_checkpoint
        identity, report = verify_checkpoint(BASE / 'runs' / ('context-training-proof-' + job))
        receipt['verified_training'] = identity
        assert report['optimizer_steps'] == 8
    active = call(['squeue', '--all', '-u', os.environ['USER'], '-h', '-o', '%i']).split()
    if any(x not in receipt['jobs'].values() for x in active) or len(active) > 1:
        raise ValueError('Another active job owns a slot; no submission')
    receipt.setdefault('submission_attempts', {})[following] = dict(at_utc=now(), status='submitting')
    save(path, receipt)
    env = {k: v for k, v in os.environ.items() if not k.startswith(('SBATCH_', 'SLURM_'))}
    env.update(NATIVE_AUDIO49_CHAIN_RECEIPT=str(path),
               NATIVE_V2_CANDIDATE_JOB=receipt['jobs']['training'])
    if following == 'speech10':
        env['NATIVE_V2_DEVELOPMENT_ARRAY_JOB'] = job
    tree = Path(env['DL_TASK_SOURCE'])
    answer = call(['sbatch', '--hold', '--parsable', '--export=ALL', '--no-requeue',
                   '--kill-on-invalid-dep=yes', '--dependency=' + dependency + ':' + job,
                   str(tree / 'evals' / script)], env).strip()
    new_job = answer.split(';')[0]
    if not new_job.isdigit():
        raise ValueError('Uncertain submission; inspect scheduler, never retry')
    receipt['jobs'][following] = new_job
    receipt['submission_attempts'][following].update(status='held', job=new_job)
    save(path, receipt)
    settings = call(['scontrol', 'show', 'job', new_job])
    receipt.setdefault('held_settings', {})[following] = settings
    save(path, receipt)
    if 'TimeLimit=00:30:00' not in settings or dependency + ':' + job not in settings:
        raise ValueError('Unexpected held job allocation; not released')
    if following == 'native10' and ('Reservation=interactive' not in settings or 'TresPerNode=gres/gpu:2' not in settings):
        raise ValueError('Unexpected GPU reservation; not released')
    call(['scontrol', 'release', new_job])
    receipt.setdefault('released', []).append(new_job)
    receipt['submission_attempts'][following]['status'] = 'released'
    save(path, receipt)
    print(json.dumps(dict(action=action, submitted=new_job, source=receipt['source'])), flush=True)


def roots(receipt, stage):
    job = receipt['jobs'][stage]
    if stage == 'readiness':
        return ['runs/native-audio49-training-preflight-' + job, 'logs/dl-native-audio49-ready-' + job + '.out']
    if stage == 'training':
        return ['runs/context-training-proof-' + job, 'logs/dl-native-audio49-pilot-' + job + '.out']
    if stage == 'speech10':
        return ['runs/native-v2-development-asr-' + job, 'logs/dl-native-audio49-speech10-' + job + '.out']
    return (['runs/native-pilot-session-' + job, 'logs/dl-native-audio49-native10-' + job + '.out']
            + ['runs/native-v2-development-' + job + '_' + str(i) for i in range(10)]
            + ['runs/native-v2-development-case-' + job + '_' + str(i) + '.json' for i in range(10)]
            + ['logs/dl-native-v2-development-' + job + '_' + str(i) + '.out' for i in range(10)])


def excluded(path):
    return any(x == 'session-token' or x.startswith('core-') or x in ('keys', 'credentials', 'owner-conversations')
               for x in path.parts)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preserve(path, receipt, stage, job, code):
    validate(receipt, stage, job)
    receipt.setdefault('stage_exit', {})[stage] = dict(job=job, exit_code=code, at_utc=now(),
        meaning='Wrapper exit status; final Slurm state must be checked separately')
    save(path, receipt)
    target = BASE / 'runs' / ('native-audio49-chain-snapshot-' + receipt['source'][:7]) / stage
    target.mkdir(parents=True, exist_ok=False)
    files = []
    missing = []
    for name in roots(receipt, stage):
        root = BASE / name
        if not root.exists():
            missing.append(name)
            continue
        for item in ([root] if root.is_file() else root.rglob('*')):
            relative = item.relative_to(BASE)
            if item.is_file() and not item.is_symlink() and not excluded(relative):
                files.append(str(relative))
    listing = target / 'files.txt'
    listing.write_text('\n'.join(sorted(files)) + '\n')
    call(['rsync', '-ar', '--exclude=session-token', '--files-from=' + str(listing), str(BASE) + '/', str(target / 'artifacts') + '/'])
    inventory = []
    for name in sorted(files):
        original = BASE / name
        copy = target / 'artifacts' / name
        # Live stdout can still grow while this EXIT trap executes. Preserve it,
        # but claim stable hash equality only for immutable model/data artifacts.
        stable = not name.startswith('logs/')
        hashed = digest(copy)
        if stable and (original.stat().st_size != copy.stat().st_size or digest(original) != hashed):
            raise ValueError('Snapshot mismatch: ' + name)
        inventory.append(dict(path=name, bytes=copy.stat().st_size, sha256=hashed, stable_source_equal=stable))
    inventory_file = target / 'inventory.json'
    inventory_file.write_text(json.dumps(dict(at_utc=now(), stage=stage, files=inventory,
        missing_expected_roots=missing, independent_off_cluster_backup=False), indent=2) + '\n')
    receipt.setdefault('snapshots', {})[stage] = dict(path=str(target), files=len(inventory),
        bytes=sum(x['bytes'] for x in inventory), inventory_sha256=digest(inventory_file),
        independent_off_cluster_backup=False, missing_expected_roots=missing)
    save(path, receipt)
    (target / 'launch-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(snapshot=str(target), files=len(inventory), missing=missing)), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['submit-native', 'submit-speech', 'preserve'])
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--exit-code', type=int)
    args = parser.parse_args()
    with Path(str(args.receipt) + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = json.loads(args.receipt.read_text())
        if args.action == 'preserve':
            if args.stage is None or args.exit_code is None:
                parser.error('Preservation requires a stage and exit code')
            preserve(args.receipt, receipt, args.stage, args.job, args.exit_code)
        else:
            submit(args.receipt, receipt, args.action, args.job)


if __name__ == '__main__':
    main()
