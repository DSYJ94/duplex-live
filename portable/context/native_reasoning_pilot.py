"""One bounded complete32 audit/retained pilot/restore; no retries or adoption."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from portable.context.native_control_preflight import git
from portable.context.native_control_training_followup import arguments, training_environment, digest
from portable.context.native_preflight_takeover import process

INPUTS = ('inputs', 'version', 'semantic', 'data', 'model', 'codec_source',
          'interleaver_source', 'cache', 'engine_inputs')


def commands(a):
    prefix = [str(a.python), '-u', '-m']
    shared = []
    for name in INPUTS:
        shared += ['--' + name.replace('_', '-'), str(getattr(a, name))]
    shared += ['--assets', str(a.root / 'runs/context-spanning-assets-7454205/manifest.json'),
               '--proof', str(a.root / 'runs/context-training-proof-7510331')]
    numerical = getattr(a, 'approved_numerical', None) or a.out / 'numerical'
    gradient = a.out / 'gradient'
    ready, proof = a.out / 'combined-readiness', a.out / 'training'
    base = arguments(a.root, a.source, a.control_cache, a.control_approval,
                     a.control_approval, a.development_evidence)
    base += ['--native-audio49-prepared', str(a.prepared49),
             '--native-audio49-source-inventory', str(a.original_inventory)]
    for name in INPUTS:
        option = 'engine' if name == 'engine_inputs' else name
        base += ['--reasoning32-' + option.replace('_', '-'), str(getattr(a, name))]
    base += ['--reasoning32-numerical', str(numerical), '--reasoning32-gradient', str(gradient)]
    fit = prefix + ['portable.context.native_training_cached'] + base
    stages = [
        ('whole_workset', prefix + ['evals.native_control_snapshot', '--inventory', str(a.workset_inventory)]),
        ('numerical32', prefix + ['evals.native_reasoning_numerical'] + shared + ['--out', str(numerical)]),
        ('gradient32', prefix + ['evals.native_reasoning_gradient'] + shared +
         ['--numerical', str(numerical), '--out', str(gradient)]),
        ('combined_readiness', fit + ['--out', str(ready), '--native-audio49-preflight-only',
                                    '--native-audio49-stream-gradient-readiness']),
        ('bounded8_training', fit + ['--out', str(proof), '--native-audio49-preflight', str(ready)]),
        ('independent_restore', fit + ['--out', str(proof), '--native-audio49-preflight', str(ready), '--reload']),
        ('strict_proof', [str(a.python), '-u', '-c',
         'from pathlib import Path; from evals.native_reasoning_stage import verify_checkpoint; '
         'verify_checkpoint(Path(' + repr(str(proof)) + ')); print("Complete retained and reasoning32 restore verified")']),
    ]
    return [s for s in stages if s[0] != 'numerical32' or not getattr(a, 'approved_numerical', None)]


def completed_numerical(folder, control, engine, proof):
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.native_reasoning_gradient import numerical_gate
    r = json.loads((control / 'controller.json').read_text())
    if (r['phase'] != 'complete32_numerical_only' or r['complete32_numerical_passed'] is not True
            or r['actual_exit'] != 0 or r['optimizer_updates'] != 0
            or Path(r['command'][r['command'].index('--out') + 1]) != folder):
        raise ValueError('Exact complete numerical32 controller exit0 required before reuse')
    live = process(r['pid'])
    if live is not None and (live['birth'] != r['birth'] or live['state'] != 'Z'):
        raise ValueError('Numerical controller still live or PID reused')
    parent, _ = verify_checkpoint(proof)
    numerical_gate(folder, engine, parent)
    return dict(controller_sha256=digest(control / 'controller.json'),
                numerical_sha256=digest(folder / 'numerical.json'), actual_exit=0)


def numerical_finished(r, live, pid, birth, source_commit):
    if (r['pid'] != pid or r['birth'] != birth or r['source_commit'] != source_commit
            or live is not None and live['birth'] != birth):
        raise ValueError('Exact upstream numerical identity changed')
    if r['phase'] == 'complete32_numerical_only':
        if r['actual_exit'] != 0 or r['complete32_numerical_passed'] is not True:
            raise ValueError('Complete numerical32 must actually exit0')
        return live is None or live['state'] == 'Z'
    if (r['phase'].startswith(('failed', 'interrupted')) or live is None or live['state'] == 'Z'):
        raise ValueError('Upstream numerical terminated without full approval')
    return False


def guard_check(guard, pod, deadline, now):
    if (guard['phase'] != 'armed' or guard['pod_id'] != pod
            or not 0 <= now - datetime.fromisoformat(guard['at_utc']).timestamp() <= 90
            or datetime.fromisoformat(guard['deadline_utc']).timestamp() < deadline + 120):
        raise ValueError('Fresh same-Pod guard must retain at least two minutes after the controller deadline')


def stage_seconds(name, now, deadline, numerical_deadline):
    end = min(deadline, numerical_deadline) if name == 'numerical32' else deadline
    remaining = end - now
    # Measured previous fit was roughly one hour; leave a conservative full
    # hour for independent restore plus a separate backup window outside this controller.
    if remaining <= 0 or name == 'bounded8_training' and remaining < 3 * 3600:
        raise TimeoutError('Insufficient bounded headroom for ' + name)
    return remaining - 3600 if name == 'bounded8_training' else remaining


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUTS, 'source', 'python', 'root', 'control_cache', 'control_approval',
                 'development_evidence', 'prepared49', 'original_inventory', 'workset_inventory',
                 'compiler', 'cost_guard_receipt', 'out'):
        p.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    for name in ('source-commit', 'pod-id', 'deadline-utc', 'numerical-deadline-utc'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--approved-numerical', type=Path)
    p.add_argument('--numerical-control', type=Path)
    p.add_argument('--follow-numerical-pid', type=int)
    p.add_argument('--follow-numerical-birth', type=int)
    p.add_argument('--follow-numerical-source-commit')
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline_utc.replace('Z', '+00:00')).timestamp()
    numerical_deadline = datetime.fromisoformat(a.numerical_deadline_utc.replace('Z', '+00:00')).timestamp()
    if bool(a.approved_numerical) != bool(a.numerical_control):
        raise ValueError('Existing numerical32 report and exact terminal controller are required together')
    follow = (a.follow_numerical_pid, a.follow_numerical_birth, a.follow_numerical_source_commit)
    if any(follow) and (not all(follow) or not a.approved_numerical):
        raise ValueError('Waiting requires explicit numerical output, control and all three identity fields')
    if not ((a.approved_numerical or time.time() + 3600 <= numerical_deadline)
            and numerical_deadline < deadline and time.time() + 3600 < deadline):
        raise ValueError('Complete numerical32 needs at least one hour of bounded headroom')
    def clean():
        if git(a.source, 'rev-parse', 'HEAD') != a.source_commit or git(a.source, 'status', '--porcelain'):
            raise ValueError('Exact clean immutable execution checkout required')
    clean()
    stages = commands(a)
    state = dict(schema='dl-native-reasoning32-pilot-controller-v1', pid=os.getpid(),
                 source_commit=a.source_commit, pod_id=a.pod_id, deadline_utc=a.deadline_utc,
                 numerical_deadline_utc=a.numerical_deadline_utc, planned_commands=dict(stages),
                 phase='prepared_only', stages=[], optimizer_update_limit=8, passed=False,
                 behavior_qualified=False, model_adopted=False, extension_approved=False)
    a.out.mkdir(parents=True, exist_ok=False)
    def save():
        state['at_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = a.out / 'controller.partial'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(a.out / 'controller.json')
    save()
    if a.prepare_only:
        return
    guard_check(json.loads(a.cost_guard_receipt.read_text()), a.pod_id, deadline, time.time())
    if all(follow):
        state.update(phase='waiting_exact_numerical32', followed_numerical_identity=list(follow))
        save()
        try:
            while time.time() < numerical_deadline:
                clean()
                numeric = json.loads((a.numerical_control / 'controller.json').read_text())
                if numerical_finished(numeric, process(a.follow_numerical_pid), *follow):
                    break
                time.sleep(10)
            else:
                raise TimeoutError('No complete terminal numerical32 within bounded start window')
        except BaseException as error:
            state.update(phase='upstream_failed_no_dispatch', error_type=type(error).__name__, error=str(error))
            save()
            raise
    if a.approved_numerical:
        state['reused_numerical'] = completed_numerical(a.approved_numerical, a.numerical_control,
            a.engine_inputs, a.root / 'runs/context-training-proof-7510331')
    compiler = json.loads(a.compiler.read_text())
    if (compiler['packages_changed'] is not False or not compiler['actual_compiled_bf16_gpu_kernel_passed']
            or not compiler['actual_triton_driver_compile_passed']):
        raise ValueError('Actual compiled BF16 runtime proof required')
    for name in ('header', 'assembler'):
        if digest(compiler[name]) != compiler[name + '_sha256']:
            raise ValueError('Changed CUDA compiler')
    lock = (a.out.parent / 'native-control40-gpu-owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; refusing duplicate work')
    env = training_environment(compiler, os.environ, a.source_commit)
    child = None
    def stop():
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    def interrupted(signum, _):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for name, command in stages:
            clean()
            seconds = stage_seconds(name, time.time(), deadline, numerical_deadline)
            row = dict(name=name, started_utc=datetime.now(timezone.utc).isoformat())
            state['phase'] = name
            state['stages'].append(row)
            save()
            with (a.out / (name + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=a.source, env=env, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                    pass_fds=(lock.fileno(),))
                row['pid'] = child.pid
                live = process(child.pid)
                if live is not None:
                    row['birth'] = live['birth']
                save()
                code = child.wait(timeout=seconds)
                row.update(actual_exit=code, ended_utc=datetime.now(timezone.utc).isoformat())
                save()
                if code:
                    raise RuntimeError(name + ' failed; no retry or downstream dispatch')
        state.update(phase='complete_restored_pilot_only', passed=True)
    except BaseException as error:
        state.update(phase='failed_preserved_no_retry', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop()
        if child is not None and state['stages']:
            state['stages'][-1]['actual_exit'] = child.returncode
        save()


if __name__ == '__main__':
    main()
