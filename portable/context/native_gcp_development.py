"""Collect unchanged paired native20 and speech20 under the existing GCP lease.

No resource creation, training, fresh-input access, scoring or adoption. Launch
only after an actual host observation of the recovery service's terminal exit0.
The existing collectors retain their own parity, restore and complete-set gates.
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
import sys
import time
import urllib.request

from portable.context.native_saved_stage_controller import fresh_guard, terminate_child


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_upstream(args, observation, upstream, now):
    if (observation.get('schema') != 'dl-gcp-recovery-terminal-observation-v1'
            or not 0 <= now-observation['observed_unix'] <= 90
            or observation['resource_id'] != args.resource_id
            or observation['unit'] != args.recovery_unit
            or observation['invocation_id'] != args.recovery_invocation
            or type(observation['actual_exit']) is not int or observation['actual_exit'] != 0
            or observation['main_pid'] != 0 or observation['result'] != 'success'
            or observation['active_state'] != 'inactive'
            or observation['controller_sha256'] != sha(args.upstream/'controller.json')
            or observation['proof_report_sha256'] != sha(args.proof/'report.json')):
        raise ValueError('Fresh actual same-resource recovery terminal0 observation required')
    rows = upstream['children']
    if (upstream['phase'] != 'recovered_saved8_proof_complete'
            or upstream['source_commit'] != args.recovery_commit
            or upstream['numerical_proof_passed'] is not True
            or upstream['optimizer_updates_this_run'] != 0 or upstream['original_stage_updates'] != 8
            or Path(upstream['derived_proof']).resolve() != args.proof.resolve()
            or [r['phase'] for r in rows] != ['finalization', 'independent_restore']
            or any(type(r['actual_exit']) is not int or r['actual_exit'] != 0 for r in rows)
            or rows[0]['pid'] == rows[1]['pid']):
        raise ValueError('Complete zero-update recovery and distinct restored proof required')


def native_command(args, paired):
    return [sys.executable, '-u', '-m', 'portable.context.native_development',
            '--source', str(args.source), '--source-commit', args.source_commit,
            '--out', str(paired), '--asr-python', str(args.asr_python),
            '--deadline-utc', args.deadline_utc, '--pod-id', args.resource_id,
            '--gpu-mode', 'serial-one-gpu', '--runtime-profile', 'container-default',
            '--reasoning32-candidate-proof', str(args.proof)]


def speech_command(args, paired, speech, pid, birth):
    return [sys.executable, '-u', '-m', 'portable.context.native_speech_followup',
            '--source', str(args.source), '--paired', str(paired),
            '--python', str(args.asr_python), '--out', str(speech),
            '--controller-pid', str(pid), '--controller-birth', str(birth),
            '--paired-source-commit', args.source_commit, '--deadline-utc', args.deadline_utc]


def collected(args, paired, speech):
    native, audit = read(paired/'execution.json'), read(speech/'controller.json')
    expected = ['baseline520', 'candidate-reasoning32']
    if (native['phase'] != 'complete' or native['complete_paired20'] is not True
            or native['source_commit'] != args.source_commit or native['gpu_mode'] != 'serial-one-gpu'
            or list(native['proofs']) != expected
            or Path(native['proofs']['candidate-reasoning32']).resolve() != args.proof.resolve()
            or audit['phase'] != 'complete_speech20_only' or audit['complete20'] is not True
            or audit['source_commit'] != args.source_commit or audit['paired_roles'] != expected):
        raise ValueError('Complete unchanged pairednative20 and independent speech20 required')
    return dict(native_execution_sha256=sha(paired/'execution.json'),
                speech_controller_sha256=sha(speech/'controller.json'),
                behavior_review_required=True, behavior_approved=False, model_adopted=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'proof', 'asr-python', 'out', 'guard', 'upstream', 'terminal-observation'):
        ap.add_argument('--'+name, type=Path, required=True)
    for name in ('source-commit', 'producer-commit', 'deadline-utc', 'resource-id',
                 'recovery-commit', 'recovery-unit', 'recovery-invocation', 'terminal-observation-sha256'):
        ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    initial = fresh_guard(args.guard)
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00')).timestamp()
    if (args.out.exists() or initial['resource_id'] != args.resource_id
            or not time.time()+2400 <= deadline <= initial['deadline_unix']-120
            or os.environ.get('OPENAI_API_KEY')):
        raise ValueError('Fresh output, complete evaluation window and credential-local lease required')
    producer = Path(__file__).resolve().parents[2]
    destination = args.out.resolve()
    for immutable in (args.source, args.proof, args.upstream):
        root = immutable.resolve()
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError('Collection output must stay separate from source, proof and recovery')
    for root, commit in ((producer, args.producer_commit), (args.source, args.source_commit)):
        if (subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != commit
                or subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()):
            raise ValueError('Exact clean committed evaluation source and producer required')
    if sha(args.terminal_observation) != args.terminal_observation_sha256:
        raise ValueError('Actual recovery terminal observation changed')
    checked_upstream(args, read(args.terminal_observation), read(args.upstream/'controller.json'), time.time())
    from evals.native_saved_stage_proof import verify_checkpoint
    identity, _ = verify_checkpoint(args.proof)
    from evals.recovery_transport import endpoint, PROXY_ENV, RUN_ENV
    if not os.environ.get(PROXY_ENV) or not os.environ.get(RUN_ENV):
        raise ValueError('Existing scoped Mac-only backend transport required')
    url, _, _ = endpoint(os.environ)
    with urllib.request.urlopen(url.replace('/v1/responses', '/health'), timeout=10) as response:
        health = json.load(response)
    if (health['run_id'] != os.environ[RUN_ENV] or health['real_requests_transport'] is not True
            or health['actual_requests'] != 0):
        raise ValueError('Live existing scoped synthetic backend transport required')
    lock = (args.out.parent/'native-control40-gpu-owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('No overlapping GPU model process allowed')
    fresh_guard(args.guard, initial)
    if time.time()+2400 > deadline:
        raise ValueError('Full evaluation window must remain after proof verification')
    args.out.mkdir()
    paired, speech = args.out/'paired-native20', args.out/'speech20'
    state = dict(schema='dl-gcp-native20-speech20-collection-v1', phase='starting',
                 resource_id=args.resource_id, producer_commit=args.producer_commit,
                 source_commit=args.source_commit, pid=os.getpid(), children=[],
                 initial_guard=initial, proof_identity=identity, deadline_utc=args.deadline_utc,
                 optimizer_updates=0, behavior_approved=False, model_adopted=False)
    children = []
    def save():
        state['at_utc'] = datetime.now(timezone.utc).isoformat()
        temp = args.out/'controller.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(args.out/'controller.json')
    def check():
        fresh_guard(args.guard, initial)
        if time.time() >= deadline:
            raise TimeoutError('Complete collection deadline reached')
        if any(c.poll() not in (None, 0) for c in children):
            raise RuntimeError('An owned collection child actually failed')
    def start(name, command):
        with (args.out/(name+'.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=args.source, env={**os.environ,
                'PYTHONDONTWRITEBYTECODE':'1'}, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(lock.fileno(),))
        children.append(child)
        state['children'].append(dict(name=name, pid=child.pid, command=command, actual_exit=None))
        save(); return child
    def interrupted(number, _):
        raise SystemExit(128+number)
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        save(); native = start('native20', native_command(args, paired))
        until = time.monotonic()+120
        while not (paired/'execution.json').is_file():
            check()
            if native.poll() is not None or time.monotonic() >= until:
                raise RuntimeError('Native controller ended or failed initial execution identity')
            time.sleep(.5)
        from portable.context.native_preflight_takeover import process
        handle = process(native.pid); current = read(paired/'execution.json')
        if (handle is None or handle['state'] == 'Z' or current['pid'] != native.pid
                or current['source_commit'] != args.source_commit):
            raise ValueError('Exact live native controller identity required')
        start('speech20', speech_command(args, paired, speech, native.pid, handle['birth']))
        state['phase'] = 'collecting_complete_native20_then_speech20'; save()
        while any(child.poll() is None for child in children):
            check(); time.sleep(1)
        check()
        for child, row in zip(children, state['children']):
            row['actual_exit'] = child.wait()
        state.update(collected(args, paired, speech), phase='complete_collection_only')
        save()
    except BaseException as error:
        for child in children:
            terminate_child(child)
        for child, row in zip(children, state['children']):
            row['actual_exit'] = child.returncode
        state.update(phase='failed_preserved_no_retry', error_type=type(error).__name__)
        save(); raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
