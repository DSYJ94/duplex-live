"""One guarded native TRAIN collection; no submission, retry, training or adoption."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .expanded_guard import AUTH, TRAIN_FAILURE_AUTH, ACCOUNT, KEY, sha, parse_job, stamp, validate_job
from .expanded_evaluation_host import parent_provider
from .expanded_launch import container_command
from .native_feature_host import archive

BASE = Path('/projects/u6ada/duplex-live')
INPUTS = BASE / 'native-TRAIN-user-audio-20261117-v1/inputs.json'
INPUT_SHA = '847084d665c2c848d46bd6ddcab0120ec6da2d5b6100730dc58dd6bca0bb1d25'
PARENT = BASE / 'native-ret-paired-recovery-20261117-v1/job-8545478/host-terminal.json'


def prepare(root, source):
    for job in ('8545478', '8551116'):
        provider = parent_provider(job)
        if provider.get('JobState') != 'COMPLETED' or provider.get('ExitCode') != '0:0':
            raise ValueError('Completed comparison and complete CPU input parent required')
    host = json.loads(PARENT.read_text())
    if host.get('success') is not True or host.get('complete_native_outputs') != 192:
        raise ValueError('Complete preserved paired boundary required')
    for receipt in host['backups'].values():
        if (receipt.get('passed') is not True or receipt.get('every_file_sha256_verified') is not True
                or sha(Path(receipt['destination']) / 'archive-inventory.json') != receipt['inventory_sha256']):
            raise ValueError('Complete independent previous archives required')
    if sha(INPUTS) != INPUT_SHA:
        raise ValueError('Exact complete verified user-only inputs required')
    value = json.loads(INPUTS.read_text())
    if (value.get('complete') is not True or value.get('required') != 1008
            or value.get('original_DEV_excluded') != 64 or len(value['cases']) != 1008
            or len({r['id'] for r in value['cases']}) != 1008
            or any(r.get('source_fold') != 'TRAIN' or r.get('teacher_audio_in_model_input') is not False for r in value['cases'])):
        raise ValueError('All1008 TRAIN-only original cases required')
    identity = host['parent_boundary']['fit']['identity']
    selection = dict(checkpoint=str(BASE / 'native-expanded1072-phase-v1/checkpoints/step-0196.pt'),
                     checkpoint_sha256=identity['checkpoint_sha256'],
                     trainer_source=identity['trainer_source'], step=identity['predecessor_step'])
    if sha(selection['checkpoint']) != selection['checkpoint_sha256']:
        raise ValueError('Exact preserved model checkpoint required')
    root.mkdir(exist_ok=False)
    plan = dict(source=source, input_manifest=str(INPUTS), input_manifest_sha256=INPUT_SHA,
                parent_root=str(BASE / 'native-expanded1072-phase-v1'), selection=selection,
                required_outputs=1008, world_size=4, optimizer_updates=0, external_API_calls=0,
                comparison_parent_sha256=sha(PARENT), teacher_audio_or_answers_provided=False)
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def complete(root, plan):
    input_rows = {r['id']: r for r in json.loads(Path(plan['input_manifest']).read_text())['cases']}
    expected = {key: row['input_sha256'] for key, row in input_rows.items()}
    if len(expected) != 1008:
        raise ValueError('Complete original1008 input identities required')
    actual, processes = {}, []
    for rank in range(4):
        folder = root / ('rank-%02d' % rank)
        terminal = json.loads((folder / 'terminal.json').read_text())
        if (terminal.get('complete') is not True or terminal.get('rank') != rank
                or terminal.get('required') != 252 or len(terminal['cases']) != 252
                or terminal.get('source') != plan['source'] or terminal.get('actual_CUDA') is not True
                or terminal.get('input_manifest_sha256') != INPUT_SHA
                or terminal.get('selection') != plan['selection'] or terminal.get('optimizer_updates') != 0
                or terminal.get('learned_RET_head_installed') is not False):
            raise ValueError('All four exact complete frozen native collectors required')
        processes.append(terminal['process_identity'])
        for row in terminal['cases']:
            ident = row['id']
            if (ident in actual or ident not in expected or row.get('complete') is not True
                    or row.get('source_fold') != 'TRAIN' or row['input_sha256'] != expected[ident]
                    or row.get('every_saved_PCM_frame_replayed_exact') is not True
                    or row.get('teacher_audio_or_answer_provided') is not False
                    or row.get('backend_connected') is not False or row.get('optimizer_updates') != 0):
                raise ValueError('No duplicate, missing, contaminated or incomplete TRAIN case allowed')
            samples = input_rows[ident]['user_frames'] + 60 * 24000
            if (row.get('user_frames') != input_rows[ident]['user_frames']
                    or row.get('response_tail_seconds') != 60
                    or row.get('input_padding_samples') != (-samples) % 1920
                    or row.get('native_frames') != (samples + 1919) // 1920
                    or type(row.get('output_samples')) is not int
                    or not 0 <= row['output_samples'] <= row['native_frames'] * 1920):
                raise ValueError('Every full original input frame must be represented')
            case = folder / ident
            if (sha(case / 'agent-raw.wav') != row['raw_wav_sha256']
                    or sha(case / 'trace.jsonl') != row['trace_sha256']
                    or json.loads((case / 'result.json').read_text()) != row):
                raise ValueError('Every complete original native result must remain exact')
            actual[ident] = row['input_sha256']
    if actual != expected or len({(p['pid_namespace'], p['pid'], p['start_ticks']) for p in processes}) != 4:
        raise ValueError('Complete1008 cohort from four distinct actual native processes required')
    return dict(complete=True, required=1008, actual_outputs=len(actual), all_result_hashes_verified=True,
                process_identities=processes, optimizer_updates=0, correctness_labels_created=False,
                backend_connected=False, product_accepted=False)


def command(work, environment, backup):
    cmd = container_command(work, gpu=True, environment=environment, backup=backup)
    position = cmd.index('--pwd')
    binds = []
    for path in (INPUTS.parent, BASE / 'native-expanded1072-phase-v1'):
        for logical in sorted({str(path), str(path.resolve())}):
            binds += ['--bind', str(path.resolve()) + ':' + logical + ':ro']
    cmd[position:position] = binds
    return cmd


def main(args):
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], universal_newlines=True).strip()
    if current != args.source:
        raise ValueError('Exact immutable committed source required')
    if args.prepare:
        prepare(args.root, args.source)
        return
    plan = json.loads((args.root / 'plan.json').read_text())
    if plan['source'] != args.source or sha(INPUTS) != INPUT_SHA:
        raise ValueError('Pinned complete prepared plan required')
    lock = (args.root / '.controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any(args.root.glob('job-*')):
        raise ValueError('Exactly one collection allocation; no implicit retries')
    if parent_provider('8545478').get('JobState') != 'COMPLETED':
        raise ValueError('Never overlap valid comparison')
    job = os.environ['SLURM_JOB_ID']
    actual = parse_job(subprocess.check_output(['scontrol', 'show', 'job', '-o', job], universal_newlines=True, timeout=30))
    accounting = subprocess.check_output(['sacct', '-A', ACCOUNT, '-S', '2026-10-30', '-X', '-n', '-P', '-o',
        'JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'], universal_newlines=True, timeout=30)
    prior = reserved = 0
    for line in accounting.splitlines():
        fields = line.split('|')
        if len(fields) < 7 or fields[0] == job:
            continue
        ident, name, state, elapsed, allocated, requested, limit = fields[:7]
        tres = dict(v.split('=', 1) for v in (allocated or requested).split(',') if '=' in v)
        count, used = int(tres.get('gres/gpu', 0)), int(elapsed)
        prior += count * used
        if state in ('RUNNING', 'PENDING', 'CONFIGURING'):
            if name.startswith('duplex-') and count:
                raise ValueError('Another valid voice GPU allocation is active')
            if not limit.isdigit():
                raise ValueError('Bounded concurrent reservation required')
            reserved += count * max(0, int(limit) * 60 - used)
    quota = subprocess.check_output(['quota', '-w', '-v'], universal_newlines=True, timeout=20)
    q = next(r.split() for r in quota.splitlines() if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '))
    used, hard = int(q[1]) * 1024, int(q[3]) * 1024
    if hard - used < 12 * 1024 ** 3:
        raise ValueError('Complete independent result archive needs12GiB headroom')
    lease = dict(provider='Slurm', resource_key=KEY, account=ACCOUNT, authorization_sha256=sha(AUTH),
        train_failure_authorization_sha256=sha(TRAIN_FAILURE_AUTH), GPU_count=4,
        job_id=job, node=actual['NodeList'], source=args.source,
        start_unix=stamp(actual['StartTime']), stop_unix=stamp(actual['EndTime']),
        prior_award_GPU_seconds=prior, other_reserved_GPU_seconds=reserved,
        collection_plan_sha256=sha(args.root / 'plan.json'))
    validate_job(actual, lease, time.time())
    folder = args.root / ('job-' + job)
    folder.mkdir(exist_ok=False)
    args.backup.mkdir(parents=True, exist_ok=True)
    path = folder / 'lease.json'
    path.write_text(json.dumps(lease, indent=2) + '\n')
    (folder / 'provider-admission.txt').write_text(accounting + '\n' + quota)
    env = dict(os.environ, DUPLEX_LIVE_GUARD_FILE=str(folder / 'guard.json'), DUPLEX_LIVE_LEASE_FILE=str(path))
    watch = subprocess.Popen(['python3', '-u', '-m', 'training.personaplex.expanded_guard',
                              '--lease', str(path), '--output', str(folder / 'guard.json')], env=env)
    try:
        until = time.time() + 180
        while not (folder / 'guard.json').exists():
            if watch.poll() is not None or time.time() > until:
                raise RuntimeError('Independent provider guard did not become ready')
            time.sleep(2)
        inside = {k: env[k] for k in ('DUPLEX_LIVE_GUARD_FILE', 'DUPLEX_LIVE_LEASE_FILE', 'SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT')}
        cmd = command(Path.cwd(), inside, args.backup)
        cmd += ['torch.distributed.run', '--standalone', '--nproc_per_node=4', '--module',
                'training.personaplex.expanded_gpu', 'training.personaplex.train_failure_rollout',
                '--root', str(args.root), '--plan', str(args.root / 'plan.json'),
                '--lease', str(path), '--source', args.source]
        subprocess.run(cmd, env=env, check=True)
        result = complete(args.root, plan)
        # Archive only immutable output shards, never the live guard/lease.
        backups = {str(rank): archive(args.root / ('rank-%02d' % rank), args.backup / ('job-' + job) / ('rank-%02d' % rank)) for rank in range(4)}
        (folder / 'host-terminal.json').write_text(json.dumps(dict(success=True, source=args.source,
            results=result, backups=backups, product_accepted=False), indent=2) + '\n')
    finally:
        watch.terminate()
        watch.wait(timeout=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'backup'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--prepare', action='store_true')
    main(parser.parse_args())
