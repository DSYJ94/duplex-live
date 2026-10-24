"""Run two fresh zero-update recovery children in one container PID namespace.

Requires an independently live GCP budget guard. Never creates resources, retries
a stage, changes the trainer, or executes an optimizer update. Preserve failures.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from portable.context.native_saved_stage_runtime import execute, guard_receipt
from portable.context.gcp_budget_watchdog import AUTH_SHA

LEASE_KEYS = ('provider', 'resource_id', 'authorization_sha256', 'deadline_unix')


def fresh_guard(path, expected=None):
    value = guard_receipt(lambda: json.loads(Path(path).read_text()))
    if value['provider'] != 'gcp' or value['authorization_sha256'] != AUTH_SHA:
        raise ValueError('Only the exact approved zero-update GCP scope is supported')
    if expected is not None and any(value[key] != expected[key] for key in LEASE_KEYS):
        raise ValueError('A running child cannot change resource, authority or deadline')
    return value


def terminate_child(child):
    """Terminate only the actual process group this controller created."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=10)
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def monitor(child, guard, expected):
    """A stale/changed external lease must stop an in-flight recovery child."""
    try:
        while child.poll() is None:
            fresh_guard(guard, expected)
            time.sleep(1)
        fresh_guard(guard, expected)
        if child.returncode != 0:
            raise RuntimeError('Recovery child did not actually exit zero')
    except BaseException:
        terminate_child(child)
        raise


def child_arguments(args, phase, output, expected, reference=None):
    result = [sys.executable, '-u', '-m', 'portable.context.native_saved_stage_controller',
              '--child', phase, '--source', str(args.source), '--snapshot', str(args.snapshot),
              '--observation', str(args.observation), '--out', str(output),
              '--guard', str(args.guard), '--producer-commit', args.producer_commit,
              '--expected-lease', str(expected)]
    if reference is not None:
        result += ['--reference', str(reference)]
    return result


def run_parent(args):
    from evals.native_saved_stage_proof import package, record_terminal
    from portable.context.native_preflight_takeover import process
    producer = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', '-C', str(producer), 'rev-parse', 'HEAD'], text=True).strip()
    if (commit != args.producer_commit
            or subprocess.check_output(['git', '-C', str(producer), 'status', '--porcelain'], text=True).strip()):
        raise ValueError('Exact clean committed recovery controller required')
    if args.proof is None or args.proof.exists() or args.out.exists():
        raise ValueError('New distinct controller and derived proof outputs required')
    out, proof = args.out.resolve(), args.proof.resolve()
    if out == proof or out in proof.parents or proof in out.parents:
        raise ValueError('Derived proof must be separate from controller outputs')
    lock = (out.parent/'native-gcp-recovery-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = fresh_guard(args.guard)
    out.mkdir()
    expected = out/'authorized-lease.json'
    expected.write_text(json.dumps({key: initial[key] for key in LEASE_KEYS}, indent=2)+'\n')
    state = dict(schema='dl-native-saved-stage-controller-v1', pid=os.getpid(),
                 source_commit=commit, model_source_commit='da389a7031a5504495397d1575952194ffa209f1',
                 original_stage_updates=8, optimizer_updates_this_run=0,
                 initial_guard=initial, children=[], phase='starting', model_adopted=False)

    def record():
        state['at_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = out/'controller.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(out/'controller.json')

    record()
    reference = None
    active_child = None
    def interrupted(number, frame):
        raise SystemExit(128+number)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        for phase in ('finalization', 'independent_restore'):
            fresh_guard(args.guard, initial)
            folder = out/phase
            command = child_arguments(args, phase, folder, expected, reference)
            started = time.time()
            with (out/(phase+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=producer, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                active_child = child
                identity = process(child.pid)
                row = dict(phase=phase, pid=child.pid,
                           birth=None if identity is None else identity['birth'],
                           started_unix=started, actual_exit=None)
                state['children'].append(row); state['phase'] = phase; record()
                try:
                    monitor(child, args.guard, initial)
                finally:
                    row['actual_exit'] = child.poll()
                    row['ended_unix'] = time.time(); record()
            job = out/(phase+'-terminal.json')
            record_terminal(child, folder, args.producer_commit, started, job)
            active_child = None
            if phase == 'finalization':
                reference = folder/'saved-stage-finalization.json'
        fresh_guard(args.guard, initial)
        identity, result = package(out/'finalization', out/'independent_restore',
            out/'finalization-terminal.json', out/'independent_restore-terminal.json',
            args.snapshot, args.observation, proof)
        state.update(phase='recovered_saved8_proof_complete', derived_proof=str(proof),
                     derived_proof_identity=identity, numerical_proof_passed=result['passed'],
                     native_behavior_evaluated=False)
        record()
        return 0
    except BaseException as error:
        if active_child is not None:
            terminate_child(active_child)
        state.update(phase='failed_preserved_no_retry', error_type=type(error).__name__)
        record()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'snapshot', 'observation', 'out', 'guard'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--producer-commit', required=True)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--child', choices=('finalization', 'independent_restore'))
    parser.add_argument('--expected-lease', type=Path)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.child:
        if args.expected_lease is None or ((args.child == 'independent_restore') != (args.reference is not None)):
            raise ValueError('Independent restore requires one completed finalization reference')
        expected = json.loads(args.expected_lease.read_text())
        reference = None if args.reference is None else json.loads(args.reference.read_text())
        execute(args.source, args.snapshot, args.observation, args.out, args.producer_commit,
                lambda: fresh_guard(args.guard, expected), reference=reference)
        return 0
    if args.reference is not None or args.expected_lease is not None:
        raise ValueError('Internal child arguments cannot start a parent run')
    return run_parent(args)


if __name__ == '__main__':
    raise SystemExit(main())
