"""Retry the interrupted eight-update experiment from its original parent.

Keep the partial candidate and all completed readiness evidence. Only the three
fit/restore/proof output paths change; this cannot resume, extend or adopt it.
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
from portable.context.native_control_pilot_takeover import guard_gate, gradient_gate
from portable.context.native_control_training_followup import digest, training_environment
from portable.context.native_preflight_takeover import process

COMMIT = 'c9efaaac24d983342f7eded46617e6450bc102ce'
POD = '4yx8vahdmjsylx'
REPORTS = {
    'complete-balanced-preflight.json': 'b99575a23cf16365a26a3033aee03c292a9afb665087b85c6a3dc6d82bae5638',
    'native-stream49-gradient-readiness.json': '26ee9b0f55c4331736fe71076f4d3364aa534df598bf11f7b6f112fd1b929acc',
    'control40-combined-readiness.json': '568117e69f18fd7d1282592e726953edbedc80569dc959bdb2db2807459cf352',
}


def retry_commands(previous, out):
    if (previous['source_commit'] != COMMIT or previous['pod_id'] != POD
            or previous['pid'] != 79780 or previous['birth'] != 24607807
            or previous['phase'] != 'bounded8_training' or previous['passed'] is not False
            or previous['optimizer_update_limit'] != 8 or len(previous['stages']) != 5
            or list(previous['planned_commands']) != list(STAGES)
            or [r['name'] for r in previous['stages']] != list(STAGES[:5])
            or any(r.get('exit_code') != 0 for r in previous['stages'][:4])
            or previous['stages'][4]['pid'] != 97250
            or previous['stages'][4]['birth'] != 25672616
            or 'exit_code' in previous['stages'][4]
            or any(r['command'] != previous['planned_commands'][r['name']] for r in previous['stages'])):
        raise ValueError('Exact interrupted pilot with four actual passed predecessors required')
    commands = copy.deepcopy(previous['planned_commands'])
    original = commands[STAGES[4]]
    old = original[original.index('--out')+1]
    if original[original.index('--actual-fit-updates')+1] != '8' or '--resume-training' in original:
        raise ValueError('Only the original eight-update parent-start experiment may be retried')
    for name in STAGES[4:6]:
        command = commands[name]
        if command[command.index('--out')+1] != old:
            raise ValueError('Fit and independent restore must share the original proof')
        command[command.index('--out')+1] = str(out/'training')
    proof = commands[STAGES[6]]
    if len(proof) != 4 or proof[2] != '-c' or proof[3].count(repr(old)) != 1:
        raise ValueError('Exact original strict proof command required')
    proof[3] = proof[3].replace(repr(old), repr(str(out/'training')))
    return commands


def run(a):
    deadline = datetime.fromisoformat(a.deadline_utc).timestamp()
    if not time.time()+120 < deadline < time.time()+12*3600:
        raise ValueError('Finite funded deadline required')
    if digest(a.previous) != a.previous_sha256:
        raise ValueError('Original interrupted controller changed')
    previous = json.loads(a.previous.read_text())
    commands = retry_commands(previous, a.out)
    failure = json.loads(a.failure.read_text())
    if (failure['partial_state_sha256'] != '5128b3c8718db5d89c145f2c1b65bedf7b92c8e2279b1d84bc0221850a44382c'
            or failure['saved_completed_steps'] != 1 or failure['saved_adam_step'] != 521
            or failure['partial_state_hash_verified_on_Mac'] is not True
            or failure['actual_backup_errno'] != 122
            or failure['actual_2GiB_write_after_reclaim_succeeded'] is not True):
        raise ValueError('Preserved partial state and actual quota recovery evidence required')
    guard_gate(json.loads(a.cost_guard.read_text()), POD, deadline)
    for pid in (79780, 97250, 71679, 71818):
        p = process(pid)
        if p is not None and p['state'] != 'Z':
            raise ValueError('Previous execution still live; never duplicate it')
    source = Path(previous['execution_source'])
    orchestration = Path(__file__).resolve().parents[2]
    orchestration_commit = subprocess.check_output(['git', '-C', str(orchestration), 'rev-parse', 'HEAD'], text=True).strip()
    def clean():
        for path, commit in ((source, COMMIT), (orchestration, orchestration_commit)):
            if (subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() != commit
                    or subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()):
                raise ValueError('Execution and orchestration sources must remain clean and pinned')
    clean()
    lease = (a.out.parent/'native-control40-gpu-owner.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(x.strip().isdigit() for x in gpu.splitlines()):
        raise ValueError('GPU occupied; never duplicate work')
    fit = commands[STAGES[4]]
    ready = Path(fit[fit.index('--native-audio49-preflight')+1])
    for name, expected in REPORTS.items():
        if digest(ready/name) != expected or json.loads((ready/name).read_text())['passed'] is not True:
            raise ValueError('Complete original retained readiness changed')
    old_fit = Path(previous['planned_commands'][STAGES[4]][fit.index('--out')+1])
    if digest(old_fit/'training-state.pt') != failure['partial_state_sha256'] or (old_fit/'report.json').exists():
        raise ValueError('Interrupted candidate changed; inspect before any retry')
    gradient = commands[STAGES[2]]
    arg = lambda n: Path(gradient[gradient.index('--'+n)+1])
    gradient_gate(json.loads((arg('out')/'gradient.json').read_text()), arg('engine-inputs'), arg('numerical'), COMMIT)
    compiler = json.loads(a.compiler.read_text())
    if (compiler['packages_changed'] is not False or not compiler['actual_compiled_bf16_gpu_kernel_passed']
            or not compiler['actual_triton_driver_compile_passed']
            or any(digest(compiler[n]) != compiler[n+'_sha256'] for n in ('header', 'assembler'))):
        raise ValueError('Original compiled runtime required')
    env = training_environment(compiler, os.environ, COMMIT)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'interrupted-controller.json').write_bytes(a.previous.read_bytes())
    (a.out/'quota-recovery.json').write_bytes(a.failure.read_bytes())
    state = dict(schema='dl-native-control40-quota-retry-v1', pid=os.getpid(), birth=process(os.getpid())['birth'],
                 source_commit=COMMIT, orchestration_commit=orchestration_commit, execution_source=str(source),
                 pod_id=POD, deadline_utc=a.deadline_utc, phase='prepared', passed=False,
                 stages=copy.deepcopy(previous['stages'][:4]), planned_commands=commands,
                 original_controller_sha256=a.previous_sha256, original_partial_candidate=str(old_fit),
                 partial_candidate_resumed=False, original_parent_restart=True, optimizer_update_limit=8,
                 model_adopted=False, behavior_qualified=False, extension_approved=False,
                 readiness_reused_with_original_actual_exits_and_exact_hashes=True)
    def save():
        state['at_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = a.out/'controller.partial';tmp.write_text(json.dumps(state, indent=2)+'\n');tmp.replace(a.out/'controller.json')
    def interrupt(signum, _):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupt);signal.signal(signal.SIGINT, interrupt)
    child = None
    save()
    try:
        for name in STAGES[4:]:
            clean()
            if time.time() >= deadline:raise TimeoutError('Funded pilot deadline reached')
            row = dict(name=name, command=commands[name], started_utc=datetime.now(timezone.utc).isoformat())
            state['phase'] = name;state['stages'].append(row);save()
            with (a.out/(name+'.log')).open('x') as log:
                child = subprocess.Popen(commands[name], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row.update(pid=child.pid, birth=process(child.pid)['birth']);save()
                code = child.wait(timeout=max(1, deadline-time.time()))
            row.update(exit_code=code, ended_utc=datetime.now(timezone.utc).isoformat());save()
            if code:raise RuntimeError(name+' failed with '+str(code))
        state.update(phase='complete_restored_pilot_only', passed=True);save();return state
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid, signal.SIGKILL);child.wait()
        state.update(phase='failed', error_type=type(error).__name__, error=str(error));save();raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('previous', 'failure', 'compiler', 'cost-guard', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--previous-sha256', required=True);p.add_argument('--deadline-utc', required=True)
    print(json.dumps(run(p.parse_args())))
