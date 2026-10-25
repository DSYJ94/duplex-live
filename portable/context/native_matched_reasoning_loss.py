"""Measure the same complete TRAIN32 at native520 and saved528, without updates.

This is a diagnostic, never a recovery proof, training approval or fresh eval.
The immutable original construction prefix supplies the already approved inputs.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROOF_SHA = '27a6676dd95ef8100e5211b6e24ddca7e9b716c0cbcae4d3e8fb02215d9ccac7'
RELOAD_SHA = '0d29f315d956d273800699706d9892e317a8eab7bc8eef6624eb19a39be59439'
SOURCE = 'da389a7031a5504495397d1575952194ffa209f1'
FAMILIES = {'direct-basic', 'delegate-reasoning', 'cancel-reasoning', 'replace-reasoning'}
FIELDS = {'native_text_calls_sha256', 'assistant_audio_logits_sha256',
          'assistant_audio_targets', 'text_targets', 'audio_mask_frames'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summarize(rows, expected):
    """Reject incomplete endpoints; retain every scalar and exact fingerprint."""
    if (len(expected) != 32 or len(rows) != 32
            or len({r['id'] for r in rows}) != 32
            or {r['id'] for r in rows} != set(expected)
            or Counter(r['family'] for r in rows) != Counter({f: 8 for f in FAMILIES})):
        raise ValueError('Complete distinct32 and all four families required')
    for row in rows:
        if (any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                for k in ('text_ce', 'acoustic_ce'))
                or row['fingerprint'] != expected[row['id']]):
            raise ValueError('Finite complete scalar losses and exact native identities required')
    def means(group):
        return {key: sum(r[key] for r in group)/len(group) for key in ('text_ce', 'acoustic_ce')}
    return dict(overall=means(rows), families={f: means([r for r in rows if r['family'] == f])
                                              for f in sorted(FAMILIES)})


def pinned_proof(path):
    path = Path(path)
    if sha(path/'report.json') != PROOF_SHA or sha(path/'reload-report.json') != RELOAD_SHA:
        raise ValueError('Only the completed exact saved8 recovery may be measured')
    proof = read(path/'report.json')
    if (proof['passed'] is not True or proof['updates_this_run'] != 0
            or proof['original_stage_updates'] != 8 or proof['original_fit_actual_exit'] != -15
            or proof['original_fit_passed'] is not False or proof['model_adopted'] is not False):
        raise ValueError('Unchanged explicit saved8 provenance required')
    return proof


def measure_context(context, inputs, prefix, producer_commit, proof_path, guard):
    import random
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from evals.native_saved_stage_recovery import validate_context, canonical, process_identity
    from evals.context_native_stream_gradient import capture
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss
    from evals.native_control_numerical import target_positions
    from evals.native_reasoning_training import tensors
    from evals.moshi_training_smoke import tensor_hashes
    from portable.context.native_saved_stage_runtime import guard_receipt

    metadata = validate_context(context, inputs)
    proof = pinned_proof(proof_path)
    final = read(Path(proof_path)/'independent-record.json')['evidence']['all_fingerprints']['reasoning32']
    if (prefix['optimizer_branch_included'] is not False
            or prefix['derived_ast_sha256'] != '7907f6de7ddbdaa1a22111829a67cfb1a4a2e5df629482f1693c4676549d15f0'
            or metadata != inputs['prefix']['metadata']):
        raise ValueError('Exact original optimizer-free construction required')
    lm, learned = context['lm'], context['learned']
    bound = context['reasoning_bound']
    policy = metadata['recipe']['reasoning32']
    declared = {r['id']: r for r in policy['teachers']}
    if (len(bound) != 32 or set(declared) != set(final)
            or any(p.grad is not None for p in lm.parameters())):
        raise ValueError('Exact complete32 context with no gradients required')
    frozen = tensor_hashes((n, p) for n, p in lm.named_parameters() if not p.requires_grad)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    out = Path(context['args'].out)
    endpoints = {}
    try:
        for role, artifact, expected in (
                ('native520', context['args'].warm_start_proof/'adapters.safetensors', None),
                ('saved528', inputs['training']/'adapters.safetensors', final)):
            guard_receipt(guard)
            weights = load_file(artifact, device='cuda')
            if set(weights) != set(learned):
                raise ValueError('Exact adapter parameter ownership required')
            expected_digest = (proof['warm_start']['checkpoint_sha256'] if role == 'native520'
                               else proof['checkpoint_sha256'])
            if sha(artifact) != expected_digest:
                raise ValueError('Endpoint adapter changed')
            with torch.no_grad():
                for name, parameter in learned.items():
                    value = weights[name]
                    if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                        raise ValueError('Endpoint adapter layout changed')
                    parameter.copy_(value)
            installed = tensor_hashes(learned.items())
            rows = []
            with torch.no_grad():
                for teacher, path, calls in bound:
                    guard_receipt(guard)
                    if teacher != declared[teacher['id']]:
                        raise ValueError('Approved teacher identity/order changed')
                    x, mask, target, codes, audio_mask = tensors(path)
                    text = capture(lm, x, mask, target, calls)
                    audio = forward(lm, codes)
                    _, counts = target_positions(audio_mask, codes, audio)
                    value, detail = loss(audio, codes, audio_mask[None])
                    selected = audio.logits[:, :8][audio.mask[:, :8] & audio_mask[None, None]].detach()
                    fingerprint = dict(
                        native_text_calls_sha256=hashlib.sha256(json.dumps(text['call_output_hashes'], separators=(',', ':')).encode()).hexdigest(),
                        assistant_audio_logits_sha256=hashlib.sha256(selected.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                        assistant_audio_targets=counts, text_targets=text['valid_targets'],
                        audio_mask_frames=int(audio_mask.sum()))
                    target_counts = policy['target_counts'][teacher['id']]
                    if (text['gradient_calls'] != 0 or detail['user_audio_loss_targets'] != 0
                            or detail['assistant_audio_targets'] != counts
                            or counts != target_counts['audio']
                            or text['valid_targets'] != target_counts['text']
                            or fingerprint['audio_mask_frames'] != target_counts['audio_mask_frames']
                            or set(fingerprint) != FIELDS
                            or (expected is not None and fingerprint != expected[teacher['id']])):
                        raise ValueError('Endpoint targets/final fingerprints differ from complete approved state')
                    row = dict(id=teacher['id'], family=teacher['family'], fingerprint=fingerprint,
                               text_ce=text['balanced_spoken_clock_ce'], acoustic_ce=float(value),
                               gradient_calls=0, per_head_ce=detail['per_head_ce'])
                    rows.append(row)
                    with (out/'loss-rows.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(endpoint=role, **row))+'\n')
                    print(json.dumps(dict(endpoint=role, id=row['id'], text_ce=row['text_ce'], acoustic_ce=row['acoustic_ce'])), flush=True)
                    del x, mask, target, codes, audio_mask, audio, value, selected
            summary = summarize(rows, expected or {r['id']: r['fingerprint'] for r in rows})
            if installed != tensor_hashes(learned.items()):
                raise ValueError('Observation changed endpoint adapter')
            endpoints[role] = dict(adapter_sha256=expected_digest, summary=summary, rows=rows)
            del weights
        if (frozen != tensor_hashes((n, p) for n, p in lm.named_parameters() if not p.requires_grad)
                or any(p.grad is not None for p in lm.parameters())):
            raise ValueError('Observation changed frozen weights or created gradients')
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
    observed_numpy = np.random.get_state()
    if (random.getstate() != rng[0] or not torch.equal(torch.get_rng_state(), rng[2])
            or any(not torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng[3]))
            or observed_numpy[0] != rng[1][0] or not np.array_equal(observed_numpy[1], rng[1][1])
            or observed_numpy[2:] != rng[1][2:]):
        raise ValueError('Observation RNG restoration differs')
    guard_receipt(guard)
    result = dict(schema='dl-native-matched-reasoning32-loss-diagnostic-v1',
                  measurement_complete=True, endpoints=endpoints, source_commit=SOURCE,
                  producer_commit=producer_commit, construction=prefix, process_identity=process_identity(),
                  optimizer_updates=0, backward_calls=0, frozen_weights_unchanged=True,
                  adapter_or_optimizer_checkpoint_written=False, rng_restored=True,
                  original_fit_relabelled=False, behavior_approved=False, model_adopted=False,
                  scope='Complete fixed32 teacher-forced TRAIN loss endpoints; not convergence/generalization or free-running qualification')
    with (out/'matched-reasoning32-loss.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    return canonical(result)


def child(args):
    """Reuse pinned construction only; no original optimizer/recovery tail runs."""
    from portable.context import native_saved_stage_runtime as runtime
    from portable.context.native_saved_stage_controller import fresh_guard
    from evals.native_saved_stage_context import bind
    from evals.native_saved_stage_recovery import verified_inputs
    producer = Path(__file__).resolve().parents[2]
    if (runtime.git(producer, 'rev-parse', 'HEAD') != args.producer_commit
            or runtime.git(producer, 'status', '--porcelain')):
        raise ValueError('Exact clean committed diagnostic producer required')
    initial = fresh_guard(args.guard)
    if initial['resource_id'] != '951004838667287080':
        raise ValueError('Only the existing exact approved GPU may be used')
    pinned_proof(args.proof)
    source = runtime.isolate(args.source)
    inputs = verified_inputs(args.snapshot, args.observation)
    out = args.out.resolve()
    roots = [Path('/projects/u6si/duplex-live/runs').resolve(), source/'runs', inputs['root'], args.proof.resolve()]
    if out.exists() or any(out == root or root in out.parents or out in root.parents for root in roots):
        raise ValueError('New diagnostic output must remain separate from all original inputs/proof')
    import fcntl
    lock = (out.parent.parent/'native-control40-gpu-owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('No overlapping GPU model process allowed')
    command = runtime.arguments(inputs['controller'], out)
    from portable.context.native_training_cached import verify_receipt, CODE, PROFILE
    ready = Path(command[command.index('--native-audio49-preflight')+1])
    verify_receipt(ready/'immutable-input-readiness.json', 'readiness', SOURCE)
    module = importlib.import_module('evals.context_actual_train')
    if Path(module.__file__).resolve() != source/'evals/context_actual_train.py':
        raise ValueError('Original construction resolved outside exact source')
    captured = []
    prefix = None
    def capture(context):
        if captured:
            raise ValueError('Only one immutable diagnostic context allowed')
        captured.append(measure_context(context, inputs, prefix, args.producer_commit,
                                        args.proof, lambda: fresh_guard(args.guard, initial)))
    module.run, prefix = bind(Path(module.__file__).read_bytes(), vars(module), capture)
    sys.argv = ['zero-update-matched32', *command]
    from evals.immutable_file_hashes import verified_reads
    runtime.restore_recorded_precision(importlib.import_module('torch'))
    complete = False
    cache = None
    try:
        with verified_reads(roots[:2]) as cache:
            importlib.import_module('evals.context_training_proof').main()
        if len(captured) != 1:
            raise ValueError('Original construction did not complete exactly one diagnostic')
        fresh_guard(args.guard, initial)
        complete = True
    finally:
        if out.is_dir() and cache is not None:
            value = dict(schema='dl-native-matched32-diagnostic-runtime-v1', measurement_complete=complete,
                         source_commit=SOURCE, producer_commit=args.producer_commit, profile=PROFILE,
                         original_source_files_sha256={name: sha(source/name) for name in CODE},
                         construction=prefix, immutable_reads=cache.receipt(), initial_guard=initial,
                         optimizer_updates=0, model_adopted=False)
            with (out/'diagnostic-runtime.json').open('x') as stream:
                stream.write(json.dumps(value, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'snapshot', 'observation', 'proof', 'guard', 'out'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--producer-commit', required=True)
    ap.add_argument('--child', action='store_true')
    args = ap.parse_args()
    if os.environ.get('OPENAI_API_KEY'):
        raise ValueError('No provider credential or network backend needed')
    if args.child:
        child(args)
        return
    from portable.context.native_saved_stage_controller import fresh_guard, terminate_child
    from evals.native_saved_stage_proof import verify_checkpoint
    from portable.context.native_preflight_takeover import process
    pinned_proof(args.proof)
    identity, _ = verify_checkpoint(args.proof)
    initial = fresh_guard(args.guard)
    if (initial['resource_id'] != '951004838667287080' or args.out.exists()
            or initial['deadline_unix']-time.time() < 3720):
        raise ValueError('Exact existing resource and complete bounded diagnostic window required')
    import fcntl
    lock = (args.out.parent/'native-matched32-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.out.mkdir()
    deadline = min(time.time()+3600, initial['deadline_unix']-120)
    command = [sys.executable, '-u', '-m', 'portable.context.native_matched_reasoning_loss',
               '--child', '--producer-commit', args.producer_commit]
    for name in ('source', 'snapshot', 'observation', 'proof', 'guard'):
        command += ['--'+name, str(getattr(args, name))]
    command += ['--out', str(args.out/'measurement')]
    state = dict(schema='dl-native-matched32-diagnostic-controller-v1', phase='starting',
                 producer_commit=args.producer_commit, proof_identity=identity,
                 optimizer_updates=0, deadline_unix=deadline, model_adopted=False)
    def save():
        temp = args.out/'controller.tmp'
        temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(args.out/'controller.json')
    def interrupted(number, _):
        raise SystemExit(128+number)
    previous = signal.signal(signal.SIGTERM, interrupted)
    worker = None
    try:
        save()
        with (args.out/'measurement.log').open('x') as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(phase='measuring_complete32', child=process(worker.pid), actual_exit=None); save()
            while worker.poll() is None:
                fresh_guard(args.guard, initial)
                if time.time() >= deadline:
                    raise TimeoutError('Original bounded diagnostic window ended')
                time.sleep(1)
        state['actual_exit'] = worker.wait()
        if worker.returncode != 0:
            raise RuntimeError('Matched diagnostic child actually failed')
        fresh_guard(args.guard, initial)
        result = read(args.out/'measurement/matched-reasoning32-loss.json')
        if (result['schema'] != 'dl-native-matched-reasoning32-loss-diagnostic-v1'
                or result['measurement_complete'] is not True or result['optimizer_updates'] != 0
                or result['backward_calls'] != 0 or result['source_commit'] != SOURCE
                or result['producer_commit'] != args.producer_commit
                or set(result['endpoints']) != {'native520', 'saved528'}):
            raise ValueError('Complete zero-update diagnostic result required')
        final = read(args.proof/'independent-record.json')['evidence']['all_fingerprints']['reasoning32']
        for role, endpoint in result['endpoints'].items():
            expected = final if role == 'saved528' else {r['id']: r['fingerprint'] for r in endpoint['rows']}
            if endpoint['summary'] != summarize(endpoint['rows'], expected):
                raise ValueError('Child endpoint summaries differ from complete measured rows')
        state.update(phase='complete_diagnostic_only', result_sha256=sha(args.out/'measurement/matched-reasoning32-loss.json'))
        save()
    except BaseException as error:
        if worker is not None:
            terminate_child(worker)
        state.update(phase='failed_preserved_no_retry', error_type=type(error).__name__,
                     actual_exit=None if worker is None else worker.returncode); save()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
