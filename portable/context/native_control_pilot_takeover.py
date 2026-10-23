"""Preserve an exact running gradient child and finish its original seven-stage pilot.

Only the superseded supervisor is paused. Its child's actual Linux exit status
and complete40 report are required before the original remaining commands run.
GPU execution remains in the original immutable checkout; no model is adopted.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from portable.context.native_control_development_wait import STAGES
from portable.context.native_control_preflight import git
from portable.context.native_control_training_followup import digest, training_environment
from portable.context.native_preflight_takeover import exact, pause, process, resume, retire


def original_gate(c, pid, child, commit, pod):
    if (c['pid'] != pid or c['source_commit'] != commit or c['pod_id'] != pod
            or c['phase'] != STAGES[2] or c['passed'] is not False
            or c['optimizer_update_limit'] != 8 or len(c['stages']) != 3
            or [r['name'] for r in c['stages']] != list(STAGES[:3])
            or any(r.get('exit_code') != 0 for r in c['stages'][:2])
            or c['stages'][2]['pid'] != child or 'exit_code' in c['stages'][2]
            or list(c['planned_commands']) != list(STAGES)
            or any(r['command'] != c['planned_commands'][r['name']] for r in c['stages'])):
        raise ValueError('Exact live gradient stage and original complete bounded chain required')
    return c


def gradient_gate(g, engine, numerical, source):
    if (g['schema'] != 'dl-native-control40-gradient-readiness-v1'
            or g['source_commit'] != source or g['passed'] is not True
            or g['complete40'] is not True or g['all_parameters_and_execution_unchanged'] is not True
            or g['optimizer_updates'] != 0 or g['checkpoint_written'] is not False
            or g['training_approved'] is not False or g['behavior_qualified'] is not False
            or g['original154_and_old49_readiness_still_required'] is not True
            or g['trainable_parameters'] != 1894400 or not 0 < g['peak_allocated_bytes'] < 80_000_000_000
            or g['engine_report_sha256'] != digest(engine/'engine-inputs.json')
            or g['numerical_report_sha256'] != digest(numerical/'numerical.json')):
        raise ValueError('Complete unchanged independent BF16 gradient proof required')
    ids = [r['id'] for r in json.loads((engine/'engine-inputs.json').read_text())['cases']]
    if len(ids) != 40 or len(set(ids)) != 40 or [r['id'] for r in g['cases']] != ids:
        raise ValueError('All exact forty cases required')
    for r in g['cases']:
        if (r['passed'] is not True or r['exact_native_streamed_text_outputs'] is not True
                or r['exact_native_acoustic_outputs'] is not True
                or any(r[n]['all26_gradients_present_and_finite'] is not True
                       or r[n]['frozen_parameter_gradients'] != 0 for n in ('text_gradient','combined_gradient'))
                or r['combined_gradient']['acoustic_text_head_gradient_delta_exactly_zero'] is not True
                or not 0 < r['combined_gradient']['acoustic_temporal_gradient_delta_norm'] < float('inf')):
            raise ValueError('Every case must retain exact outputs and valid gradient ownership')
    return g


def guard_gate(g, pod, deadline):
    now = time.time()
    if (g['phase'] != 'armed' or g['pod_id'] != pod
            or not 0 <= now-datetime.fromisoformat(g['at_utc']).timestamp() <= 90
            or datetime.fromisoformat(g['deadline_utc']).timestamp() < deadline+120):
        raise ValueError('Fresh longer same-Pod cost guard required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('original','compiler','cost-guard-receipt','out'):
        p.add_argument('--'+n, type=Path, required=True)
    for n in ('controller-pid','controller-birth','child-pid','child-birth'):
        p.add_argument('--'+n, type=int, required=True)
    for n in ('source-commit','pod-id','deadline-utc','original-controller-sha256'):
        p.add_argument('--'+n, required=True)
    a = p.parse_args()
    deadline = datetime.fromisoformat(a.deadline_utc).timestamp()
    if not time.time()+120 < deadline < time.time()+18*3600:
        raise ValueError('Finite future pipeline deadline required')
    guard_gate(json.loads(a.cost_guard_receipt.read_text()), a.pod_id, deadline)
    if digest(a.original/'controller.json') != a.original_controller_sha256:
        raise ValueError('Original controller differs from the independently observed snapshot')
    original = original_gate(json.loads((a.original/'controller.json').read_text()),
                             a.controller_pid, a.child_pid, a.source_commit, a.pod_id)
    source = Path(original['execution_source'])
    orchestration = Path(__file__).resolve().parents[2]
    orchestration_commit = git(orchestration, 'rev-parse', 'HEAD')
    def clean():
        for path, commit in ((source, a.source_commit), (orchestration, orchestration_commit)):
            if git(path, 'rev-parse', 'HEAD') != commit or git(path, 'status', '--porcelain'):
                raise ValueError('Execution and orchestration checkouts must stay clean and pinned')
    clean()
    exact(a.controller_pid, a.controller_birth)
    exact(a.child_pid, a.child_birth)
    if original['stages'][2]['birth'] != a.child_birth:
        raise ValueError('Recorded child birth changed')
    cmdline = (Path('/proc')/str(a.child_pid)/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if cmdline != original['stages'][2]['command']:
        raise ValueError('Actual child command differs from approved original')
    compiler = json.loads(a.compiler.read_text())
    if (compiler['packages_changed'] is not False or not compiler['actual_compiled_bf16_gpu_kernel_passed']
            or not compiler['actual_triton_driver_compile_passed']
            or any(digest(compiler[n]) != compiler[n+'_sha256'] for n in ('header','assembler'))):
        raise ValueError('Original compiled BF16 environment required')
    env = training_environment(compiler, os.environ, a.source_commit)
    if env.get('PYTHONPATH') != '/workspace/control40-vendor-moshi-rag/moshi':
        raise ValueError('Original pinned legacy native environment required')
    adoption_lock = (a.original/'pilot-supervisor-takeover.lock').open('a')
    fcntl.flock(adoption_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'original-controller-before-takeover.json').write_text(json.dumps(original, indent=2)+'\n')
    # Original commands retain their exact output paths; expose their proof to
    # the existing evaluation dispatcher without changing any training argument.
    (a.out/'training').symlink_to(a.original/'training', target_is_directory=True)
    state = copy.deepcopy(original)
    state.update(schema='dl-native-control40-pilot-supervisor-takeover-v1', pid=os.getpid(),
                 deadline_utc=a.deadline_utc, orchestration_commit=orchestration_commit,
                 original_controller_pid=a.controller_pid, original_controller_birth=a.controller_birth,
                 original_controller_sha256=digest(a.out/'original-controller-before-takeover.json'),
                 adopted_gradient_pid=a.child_pid, adopted_gradient_birth=a.child_birth,
                 original_deadline_utc=original['deadline_utc'], gradient_restarted=False,
                 phase='preparing_takeover', optimizer_updates=0)
    def save():
        state['at_utc'] = datetime.now(timezone.utc).isoformat()
        temp = a.out/'controller.partial'
        temp.write_text(json.dumps(state, indent=2)+'\n');temp.replace(a.out/'controller.json')
    def interrupted(signum, _): raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted);signal.signal(signal.SIGINT, interrupted)
    paused = False
    child = None
    lease = None
    save()
    try:
        pause(a.controller_pid, a.controller_birth);paused = True
        state.update(phase=STAGES[2], old_supervisor_pause_confirmed=True);save()
        while time.time() < deadline:
            actual = exact(a.child_pid, a.child_birth)
            if actual['state'] == 'Z':
                code = os.waitstatus_to_exitcode(actual['exit_status'])
                state['actual_adopted_gradient_exit_code'] = code;save()
                if code != 0:raise RuntimeError('Original gradient child failed: '+str(code))
                break
            if exact(a.controller_pid, a.controller_birth)['state'] not in ('T','t'):
                raise RuntimeError('Original supervisor resumed unexpectedly')
            try:
                g = json.loads((a.original/'gradient/gradient.json').read_text())
                state['gradient_cases_observed'] = len(g['cases'])
            except (FileNotFoundError, json.JSONDecodeError):pass
            save();time.sleep(5)
        else:
            exact(a.child_pid, a.child_birth)
            os.killpg(a.child_pid, signal.SIGTERM)
            raise TimeoutError('Extended gradient deadline reached')
        clean()
        gradient_gate(json.loads((a.original/'gradient/gradient.json').read_text()),
                      a.original/'engine', a.original/'numerical', a.source_commit)
        stage = state['stages'][2]
        stage.update(exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat(),
                     exit_evidence='actual Linux zombie wait status before original parent retirement')
        save()
        retire(a.controller_pid, a.controller_birth);paused = False
        state.update(old_supervisor_retired_after_child_exit=True);save()
        lease = (a.original.parent/'native-control40-gpu-owner.lock').open('a')
        fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True)
        if any(x.strip().isdigit() for x in gpu.splitlines()):raise RuntimeError('GPU still occupied')
        for name in STAGES[3:]:
            clean()
            if deadline <= time.time():raise TimeoutError('Pilot deadline reached')
            command = original['planned_commands'][name]
            stage = dict(name=name, command=command, started_utc=datetime.now(timezone.utc).isoformat())
            state.update(phase=name);state['stages'].append(stage);save()
            with (a.out/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=source, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                stage.update(pid=child.pid, birth=exact(child.pid, process(child.pid)['birth'])['birth']);save()
                code = child.wait(timeout=max(1, deadline-time.time()))
                stage.update(exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat());save()
                if code != 0:raise RuntimeError(name+' failed: '+str(code))
        state.update(phase='complete_restored_pilot_only', passed=True);save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait()
        state.update(phase='failed', error_type=type(error).__name__, error=str(error));save();raise
    finally:
        if paused:resume(a.controller_pid, a.controller_birth)


if __name__ == '__main__':main()
