"""Generate the exact complete40 authored source once, preserving failed outputs.

This is source collection, not source/teacher/training approval. No content
rerolls, gain changes, corpus reuse, private conversations or remote credentials.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from data.synthetic.native_control_repair import build, validate
from data.synthetic.native_control_render import load_clip, sha256
from data.synthetic.recovery_curriculum import MODEL, synthesize
from evals.native_control_teacher import collection_lock, local_key


def source_tasks(plan):
    return [dict(case=scene['id'], index=i, role=turn['role'], text=turn['text'],
                 voice=scene['user_voice'] if turn['role'] == 'user' else scene['assistant_voice'])
            for scene in plan['cases'] for i, turn in enumerate(scene['turns'])]


def retained_source(parent, plan_path, tasks):
    """Revalidate original bytes before retaining any raw clip in a repair."""
    ledger = json.loads((parent / 'generation.json').read_text())
    if (ledger['status'] != 'failed_preserved_unapproved' or ledger['training_approved'] is not False
            or ledger['plan_sha256'] != sha256(plan_path)
            or ledger['speech_helper_sha256'] != sha256(Path('data/synthetic/recovery_curriculum.py'))):
        raise ValueError('Require the exact preserved failed source version')
    content = subprocess.check_output(['git', 'show', ledger['source_commit'] + ':data/synthetic/native_control_generate.py'])
    if __import__('hashlib').sha256(content).hexdigest() != ledger['implementation_sha256']:
        raise ValueError('Parent collection lacks immutable source provenance')
    expected = {f'clips/{t["case"]}/{t["index"]:02d}_{t["role"]}.wav': t for t in tasks}
    retained, seen = [], set()
    for row in ledger['clips']:
        name = row['path']
        if name not in expected or name in seen:
            raise ValueError('Duplicate or out-of-plan retained source')
        _, measured = load_clip(parent / name, expected[name])
        if (any(row[k] != v for k, v in expected[name].items())
                or any(row[k] != v for k, v in measured.items())):
            raise ValueError('Retained raw source words, voice, receipt or audio changed')
        retained.append(row)
        seen.add(name)
    # This repair is deliberately narrow: one measured saturated waveform and
    # missing outputs. Never reroll an undiagnosed generated recording.
    failed = 'clips/native-control3-train-034/02_user.wav'
    extra = {str(p.relative_to(parent)) for p in (parent / 'clips').rglob('*.wav')} - seen
    diagnosis = parent / 'failed-waveform-diagnostic.json'
    evidence = json.loads(diagnosis.read_text())
    import numpy as np
    import soundfile as sf
    pcm, _ = sf.read(parent / failed)
    if (extra != {failed} or evidence['wav_sha256'] != sha256(parent / failed)
            or evidence['peak'] != float(np.abs(pcm).max())
            or evidence['samples_above_limit'] != int(np.sum(np.abs(pcm) > .995))
            or evidence['peak'] <= .995 or evidence['samples_above_limit'] <= 0):
        raise ValueError('Only the diagnosed peak failure may replace an existing waveform')
    return retained, dict(parent=str(parent), parent_generation_sha256=sha256(parent / 'generation.json'),
                          diagnosis_sha256=sha256(diagnosis), diagnosed_replacement=failed,
                          original_failed_wav_sha256=evidence['wav_sha256'],
                          retained_clips=len(retained), retained_audio_and_receipts_unchanged=True)


def collect(plan_path, out, key, lock, *, generate=synthesize, repair_source=None):
    plan = json.loads(plan_path.read_text())
    coverage = validate(plan)
    if plan != build():
        raise ValueError('Require the exact committed complete40 source plan')
    tasks = source_tasks(plan)
    if len(tasks) != 168 or sum(len(t['text']) for t in tasks) != 7953:
        raise ValueError('Unexpected authorized request bound')
    with collection_lock(lock):
        if out.exists():
            raise ValueError('Use a NEW version, preserving every failed/partial collection')
        out.mkdir(parents=True)
        (out / 'source-plan.json').write_bytes(plan_path.read_bytes())
        state = dict(schema='dl-native-control3-source-generation-v1',
                     started_utc=datetime.now(timezone.utc).isoformat(),
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                     implementation_sha256=sha256(Path(__file__)),
                     speech_helper_sha256=sha256(Path('data/synthetic/recovery_curriculum.py')),
                     plan_sha256=sha256(plan_path), coverage=coverage, model=MODEL,
                     maximum_clips=168, maximum_characters=7953, concurrency=2,
                     transport_retry_limit=3, content_rerolls=0,
                     owner_conversations_used=False, credentials_exported=False,
                     gain_changed=False, training_approved=False,
                     status='collecting_unapproved', clips=[], artifacts={})
        instructions = {}
        pending = tasks
        if repair_source is not None:
            retained, provenance = retained_source(repair_source, plan_path, tasks)
            for row in retained:
                original = repair_source / row['path']
                for source in (original, original.with_suffix('.receipt.json')):
                    target = out / source.relative_to(repair_source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            state['clips'] = list(retained)
            state['repair_provenance'] = provenance
            copied = {r['path'] for r in retained}
            pending = [t for t in tasks if f'clips/{t["case"]}/{t["index"]:02d}_{t["role"]}.wav' not in copied]
            pending.sort(key=lambda t: (t['case'], t['index']) != ('native-control3-train-034', 2))
            instructions[('native-control3-train-034', 2)] = (
                'Read exactly the supplied single word once, with no repetition or continuation. '
                'End the utterance immediately after that word. Use natural conversational delivery.')
        state['maximum_new_clips'] = len(pending)
        state['maximum_new_characters'] = sum(len(t['text']) for t in pending)

        def persist():
            state['artifacts'] = {str(p.relative_to(out)): sha256(p)
                                  for p in sorted((out / 'clips').rglob('*')) if p.is_file()}
            (out / 'generation.json').write_text(json.dumps(state, indent=2) + '\n')

        def one(task):
            path = out / 'clips' / task['case'] / f'{task["index"]:02d}_{task["role"]}.wav'
            path.parent.mkdir(parents=True, exist_ok=True)
            instruction = instructions.get((task['case'], task['index']))
            if instruction:
                generate(task['text'], task['voice'], path, instructions=instruction)
            else:
                generate(task['text'], task['voice'], path)
            _, receipt = load_clip(path, task)
            return dict(task, path=str(path.relative_to(out)), **receipt)

        previous = os.environ.get('OPENAI_API_KEY')
        os.environ['OPENAI_API_KEY'] = key
        persist()
        failures = []
        try:
            # Probe one actual request before scheduling two concurrently. The
            # helper retries only bounded transient transport failures, never words.
            with ThreadPoolExecutor(max_workers=2) as pool:
                iterator = iter(pending)
                first = next(iterator, None)
                active = {pool.submit(one, first): first} if first else {}
                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = active.pop(future)
                        try:
                            state['clips'].append(future.result())
                        except Exception as exc:
                            http = re.search(r'HTTP (\d{3})', str(exc))
                            failures.append(dict(case=task['case'], index=task['index'],
                                                 error_type=type(exc).__name__,
                                                 http_status=int(http[1]) if http else None))
                    state['failures'] = failures
                    persist()
                    print(json.dumps(dict(completed=len(state['clips']), failed=len(failures),
                                          in_flight=len(active), maximum=168)), flush=True)
                    # Drain already active requests on any failure, without
                    # scheduling new requests or admitting a passing subset.
                    if not failures:
                        while len(active) < 2:
                            task = next(iterator, None)
                            if task is None:
                                break
                            active[pool.submit(one, task)] = task
            state['status'] = ('complete_raw_unapproved' if not failures and len(state['clips']) == 168
                               else 'failed_preserved_unapproved')
        finally:
            if previous is None:
                os.environ.pop('OPENAI_API_KEY', None)
            else:
                os.environ['OPENAI_API_KEY'] = previous
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            if state['status'] == 'collecting_unapproved':
                state['status'] = 'interrupted_preserved_unapproved'
            persist()
        if state['status'] != 'complete_raw_unapproved':
            raise RuntimeError('Source collection failed; preserve complete ledger and partial audio')
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--repair-source', type=Path,
                        help='Retain verified originals and repair only the diagnosed TRAIN peak failure/missing clips')
    parser.add_argument('--lock', type=Path,
                        default=Path('runs/local-migration-20261021/.private/control3-capture.lock'))
    args = parser.parse_args()
    result = collect(args.plan, args.out, local_key(args.env_file), args.lock, repair_source=args.repair_source)
    print(json.dumps(dict(status=result['status'], clips=len(result['clips']), training_approved=False)))


if __name__ == '__main__':
    main()
