"""Run one deadline-bounded, sequential whole40 native preflight; never train.

The controller may live in a separate Git checkout. Every native subprocess runs
from the explicitly pinned execution checkout containing the verified workset.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'python', 'inventory', 'compiler', 'codec-source', 'interleaver-source', 'model', 'assets', 'proof', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00')).timestamp()
    if (deadline <= time.time() + 60 or len(args.source_commit) != 40
            or git(args.source, 'rev-parse', 'HEAD') != args.source_commit
            or git(args.source, 'status', '--porcelain')):
        raise ValueError('Clean exact execution source and future bounded deadline required')
    compiler = json.loads(args.compiler.read_text())
    if (compiler['schema'] != 'dl-native-cuda-compiler-check-v1'
            or compiler['actual_triton_driver_compile_passed'] is not True
            or compiler['actual_compiled_bf16_gpu_kernel_passed'] is not True
            or compiler['packages_changed'] is not False):
        raise ValueError('Actual pinned-runtime compiled GPU kernel must pass first')
    args.out.mkdir(parents=True, exist_ok=False)
    controller_root = Path(__file__).resolve().parents[2]
    state = dict(schema='dl-native-control40-preflight-controller-v1', pid=os.getpid(),
                 controller_commit=git(controller_root, 'rev-parse', 'HEAD'), source_commit=args.source_commit,
                 execution_source=str(args.source), deadline_utc=args.deadline_utc,
                 started_utc=datetime.now(timezone.utc).isoformat(), phase='starting',
                 stages=[], optimizer_updates=0, checkpoint_written=False, training_approved=False, passed=False)
    def save():
        temp = args.out / 'controller.json.partial'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(args.out / 'controller.json')
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', OMP_NUM_THREADS='8')
    prefix = [str(args.python), '-u', '-m']
    shared = ['--cache', str(args.out / 'codes'), '--assets', str(args.assets),
              '--base', str(args.model), '--proof', str(args.proof)]
    stages = [
        ('codec', prefix + ['evals.native_control_cache', '--data', 'runs/native-control3-webrtc-postteacher-20261022-v1/converted',
                           '--model', str(args.model), '--codec-source', str(args.codec_source),
                           '--interleaver-source', str(args.interleaver_source), '--out', str(args.out / 'codes'),
                           '--frozen-workset', str(args.inventory)]),
        ('engine', prefix + ['evals.native_control_engine'] + shared + ['--out', str(args.out / 'engine')]),
        ('numerical', prefix + ['evals.native_control_numerical'] + shared +
         ['--engine-inputs', str(args.out / 'engine'), '--out', str(args.out / 'numerical')]),
    ]
    child = None
    def terminate(*_):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    def interrupted(signum, _):
        terminate()
        state.update(phase='interrupted', signal=signum)
        save()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    for name, command in stages:
        if git(args.source, 'rev-parse', 'HEAD') != args.source_commit or git(args.source, 'status', '--porcelain'):
            state.update(phase='source_changed'); save()
            raise ValueError('Execution checkout changed between stages')
        remaining = deadline - time.time()
        if remaining <= 0:
            state.update(phase='deadline_reached'); save()
            raise TimeoutError('Native preflight deadline reached before next stage')
        stage = dict(name=name, command=command, started_utc=datetime.now(timezone.utc).isoformat())
        state['stages'].append(stage)
        state['phase'] = name
        with (args.out / (name + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=args.source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stage['pid'] = child.pid
            save()
            try:
                code = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                terminate()
                stage.update(timed_out=True, exit_code=child.returncode)
                state['phase'] = 'deadline_reached'; save()
                raise
        stage.update(exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        if code:
            state['phase'] = 'failed_' + name; save()
            raise RuntimeError('Native preflight stage failed: ' + name)
    state.update(phase='complete_numerical_only', passed=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                 gradient_retention_and_behavior_approval_still_required=True)
    save()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
