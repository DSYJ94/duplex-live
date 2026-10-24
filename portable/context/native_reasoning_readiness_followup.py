"""Reuse a completed readiness audit only after its controller's start-window refusal.

No training retry: the original controller must have ended before its first
optimizer stage. Model commands and source stay identical; only output paths
and the controller deadline change, inside the existing same-Pod cost guard.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from portable.context.native_control_preflight import git
from portable.context.native_control_training_followup import digest, training_environment
from portable.context.native_preflight_takeover import process
from portable.context.native_reasoning_pilot import guard_check, stage_seconds


PREFIX = ('whole_workset', 'gradient32', 'combined_readiness')


def eligible(state, live, pid, birth, commit, pod):
    if (state['pid'] != pid or state['source_commit'] != commit or state['pod_id'] != pod
            or live is not None and live['birth'] != birth):
        raise ValueError('Exact original controller identity required')
    if live is not None and live['state'] != 'Z':
        raise ValueError('Original controller must actually be terminal; never duplicate it')
    if (state['phase'] != 'failed_preserved_no_retry'
            or state.get('error_type') != 'TimeoutError'
            or state.get('error') != 'Insufficient bounded headroom for bounded8_training'
            or [r['name'] for r in state['stages']] != list(PREFIX)
            or any(r.get('actual_exit') != 0 for r in state['stages'])
            or state.get('optimizer_update_limit') != 8
            or state.get('passed') is not False):
        raise ValueError('Only a complete exit0 audit followed by pre-training time refusal is reusable')
    numerical = state.get('reused_numerical', {})
    if (numerical.get('actual_exit') != 0
            or any(len(numerical.get(k, '')) != 64 for k in ('controller_sha256', 'numerical_sha256'))):
        raise ValueError('Complete original numerical binding required')


def followup_commands(state, upstream, out):
    old = str(upstream / 'training')
    fit = list(state['planned_commands']['bounded8_training'])
    restore = list(state['planned_commands']['independent_restore'])
    for command in (fit, restore):
        if (command.count('--out') != 1 or command[command.index('--out') + 1] != old
                or command.count('--actual-fit-updates') != 1
                or command[command.index('--actual-fit-updates') + 1] != '8'
                or command.count('--native-audio49-preflight') != 1
                or command[command.index('--native-audio49-preflight') + 1] != str(upstream / 'combined-readiness')):
            raise ValueError('Exact original bounded8 commands and completed readiness required')
        command[command.index('--out') + 1] = str(out / 'training')
    if '--reload' in fit or restore != fit + ['--reload']:
        raise ValueError('Independent restore must differ only by reload')
    strict = [fit[0], '-u', '-c',
        'from pathlib import Path; from evals.native_reasoning_stage import verify_checkpoint; '
        'verify_checkpoint(Path(' + repr(str(out / 'training')) + ')); print("Complete retained and reasoning32 restore verified")']
    return [('bounded8_training', fit), ('independent_restore', restore), ('strict_proof', strict)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'source', 'compiler', 'cost-guard-receipt', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('source-commit', 'controller-commit', 'pod-id', 'deadline-utc', 'expected-fit-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--upstream-pid', type=int, required=True)
    p.add_argument('--upstream-birth', type=int, required=True)
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00')).timestamp()
    original = a.upstream / 'controller.json'
    state = json.loads(original.read_text())
    eligible(state, process(a.upstream_pid), a.upstream_pid, a.upstream_birth, a.source_commit, a.pod_id)
    fit_hash = hashlib.sha256(json.dumps(state['planned_commands']['bounded8_training'],
        separators=(',', ':')).encode()).hexdigest()
    if fit_hash != a.expected_fit_sha256:
        raise ValueError('Original fit must match the separately committed exact recipe')
    def clean():
        if git(a.source, 'rev-parse', 'HEAD') != a.source_commit or git(a.source, 'status', '--porcelain'):
            raise ValueError('Exact unchanged model execution checkout required')
    clean()
    controller = Path(__file__).resolve().parents[2]
    if git(controller, 'rev-parse', 'HEAD') != a.controller_commit or git(controller, 'status', '--porcelain'):
        raise ValueError('Exact clean followup controller checkout required')
    guard_check(json.loads(a.cost_guard_receipt.read_text()), a.pod_id, deadline, time.time())
    stage_seconds('bounded8_training', time.time(), deadline, deadline)
    if (a.upstream / 'training').exists() or a.out.exists():
        raise ValueError('Any prior training output or successor output forbids a retry')
    # Pure complete-set gates run before any GPU process. The unchanged model
    # fit independently rebinds these policies to all original source inputs.
    from evals.native_reasoning_stage import check_readiness
    from evals.native_reasoning_training import validate_gradient
    from evals.native_reasoning_gradient import validate_numerical
    ready_path = a.upstream / 'combined-readiness/reasoning32-combined-readiness.json'
    ready = json.loads(ready_path.read_text())
    check_readiness(ready_path, ready['policy'], a.source_commit)
    gradient_path = a.upstream / 'gradient/gradient.json'
    gradient = json.loads(gradient_path.read_text())
    engine_path = Path(state['planned_commands']['gradient32'][state['planned_commands']['gradient32'].index('--engine-inputs') + 1]) / 'engine-inputs.json'
    if not engine_path.exists():
        raise ValueError('Exact approved Engine manifest required')
    numerical_path = Path(state['planned_commands']['gradient32'][state['planned_commands']['gradient32'].index('--numerical') + 1]) / 'numerical.json'
    engine = json.loads(engine_path.read_text())
    numerical = json.loads(numerical_path.read_text())
    validate_numerical(numerical, engine, ready['policy']['parent'])
    validate_gradient(gradient, engine, numerical, ready['policy']['parent'])
    if (digest(gradient_path) != ready['policy']['gradient_readiness_sha256']
            or digest(numerical_path) != ready['policy']['numerical_sha256']
            or digest(engine_path) != ready['policy']['engine_sha256']
            or digest(numerical_path) != state['reused_numerical']['numerical_sha256']):
        raise ValueError('Exact completed numerical/gradient/Engine hashes must bind joint readiness')
    lock = (a.out.parent / 'native-control40-gpu-owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; refusing duplicate work')
    compiler = json.loads(a.compiler.read_text())
    if (compiler['packages_changed'] is not False or not compiler['actual_compiled_bf16_gpu_kernel_passed']
            or not compiler['actual_triton_driver_compile_passed']):
        raise ValueError('Pinned compiled BF16 runtime required')
    for name in ('header', 'assembler'):
        if digest(compiler[name]) != compiler[name + '_sha256']:
            raise ValueError('Changed CUDA compiler')
    commands = followup_commands(state, a.upstream, a.out)
    receipt = dict(schema='dl-native-reasoning32-readiness-followup-v1', pid=os.getpid(),
        source_commit=a.source_commit, controller_commit=a.controller_commit, pod_id=a.pod_id,
        deadline_utc=a.deadline_utc, phase='ready_before_first_training', passed=False,
        stages=[dict(row, reused_from_controller=str(original)) for row in state['stages']],
        reused_numerical=state['reused_numerical'], optimizer_update_limit=8,
        original_fit_command_sha256=fit_hash,
        reused_readiness=dict(upstream_pid=a.upstream_pid, upstream_birth=a.upstream_birth,
            upstream_controller_sha256=digest(original), readiness_sha256=digest(ready_path),
            gradient_sha256=digest(gradient_path), numerical_sha256=digest(numerical_path),
            engine_sha256=digest(engine_path)), planned_commands=dict(commands),
        behavior_qualified=False, model_adopted=False, extension_approved=False)
    a.out.mkdir(exist_ok=False)
    def save():
        receipt['at_utc'] = datetime.now(timezone.utc).isoformat()
        temp = a.out / 'controller.partial'
        temp.write_text(json.dumps(receipt, indent=2) + '\n')
        temp.replace(a.out / 'controller.json')
    env = training_environment(compiler, os.environ, a.source_commit)
    child = None
    def interrupted(signum, _):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        for name, command in commands:
            clean()
            seconds = stage_seconds(name, time.time(), deadline, deadline)
            child = None
            row = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
            receipt['phase'] = name
            receipt['stages'].append(row)
            save()
            with (a.out / (name + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=a.source, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                    pass_fds=(lock.fileno(),))
                row['pid'] = child.pid
                live = process(child.pid)
                if live: row['birth'] = live['birth']
                save()
                row['actual_exit'] = child.wait(timeout=seconds)
                row['ended_utc'] = datetime.now(timezone.utc).isoformat()
                save()
                if row['actual_exit']: raise RuntimeError(name + ' failed; no retry')
        receipt.update(phase='complete_restored_pilot_only', passed=True)
    except BaseException as error:
        receipt.update(phase='failed_preserved_no_retry', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if child is not None: receipt['stages'][-1]['actual_exit'] = child.returncode
        save()


if __name__ == '__main__': main()
