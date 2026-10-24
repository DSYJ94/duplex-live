"""Integrate reasoning32 while preserving every retained154/native49/control40 gate."""
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

from evals.native_reasoning_source import sha

SCHEMA = 'dl-native-reasoning32-retained154-stream49-control40-proof-v1'
RELOAD_SCHEMA = 'dl-native-reasoning32-retained154-stream49-control40-reload-v1'
FIELDS = ('inputs', 'version', 'semantic', 'data', 'model', 'codec_source', 'interleaver_source',
          'cache', 'engine', 'numerical', 'gradient')
OPTIONS = tuple('reasoning32_' + n for n in FIELDS)


def validate(args):
    supplied = [bool(getattr(args, n, None)) for n in OPTIONS]
    if not any(supplied):
        return False
    from evals.native_control_stage import validate as control_validate
    if (not all(supplied) or not control_validate(args)
            or args.actual_fit_updates != 8 or args.resume_training
            or not args.warm_start_proof or Path(args.warm_start_proof).name != 'context-training-proof-7510331'):
        raise ValueError('Reasoning32 requires complete inputs and retained154/native49/control40 eight-update native520 stage')
    return True


def prepare(args, parent, control_policy, retained_policy):
    from evals.native_reasoning_training import prepare as prepare_objective
    values = {('engine_inputs' if n == 'engine' else n): getattr(args, 'reasoning32_' + n) for n in FIELDS}
    policy, bound = prepare_objective(SimpleNamespace(**values, assets=args.assets, proof=args.warm_start_proof), parent)
    policy['stage_implementation_sha256'] = sha(Path(__file__))
    for name, field, filename in (('cache', 'cache', 'cache.json'), ('engine', 'engine', 'engine-inputs.json'),
                                  ('numerical', 'numerical', 'numerical.json'), ('gradient', 'gradient', 'gradient.json')):
        source = getattr(args, 'reasoning32_' + field) / filename
        target = args.out / ('approved-reasoning32-' + name + '.json')
        if args.reload:
            if sha(target) != sha(source):
                raise ValueError('Changed packaged reasoning32 approval')
        else:
            target.write_bytes(source.read_bytes())
    if not args.native_audio49_preflight_only:
        source = args.native_audio49_preflight / 'reasoning32-combined-readiness.json'
        ready = check_readiness(source, policy)
        control_source = args.native_audio49_preflight / 'control40-combined-readiness.json'
        control = json.loads(control_source.read_text())
        check_bindings(ready, control_policy, retained_policy['native_stream_auxiliary'], control, sha(control_source))
        complete = args.native_audio49_preflight / 'complete-balanced-preflight.json'
        if ready['original154_gradient_source_sha256'] != sha(complete):
            raise ValueError('Changed original154 combined gradient source')
        target = args.out / 'approved-reasoning32-combined-readiness.json'
        if args.reload:
            if sha(target) != sha(source):
                raise ValueError('Changed combined reasoning32 readiness')
        else:
            target.write_bytes(source.read_bytes())
        target = args.out / 'approved-complete-balanced-preflight.json'
        if args.reload:
            if sha(target) != sha(complete):
                raise ValueError('Changed packaged original154 gradient source')
        else:
            target.write_bytes(complete.read_bytes())
    return policy, bound


def check_bindings(ready, control_policy, native49_policy, control, control_digest):
    if (ready['control40_policy'] != control_policy or ready['native49_policy'] != native49_policy
            or ready['control40_combined_readiness_sha256'] != control_digest
            or any(ready[k] != control[k] for k in ('base154_gradient_sha256', 'source_commit',
                'original154_gradient_source_sha256', 'native49_readiness_sha256'))
            or len(control['steps']) != 8
            or any(any(a[k] != b[k] for k in ('step', 'native_stream_teacher_microbatch', 'control40_microbatch'))
                   for a, b in zip(ready['steps'], control['steps']))):
        raise ValueError('Reasoning32 must bind the same retained154/native49/control40 readiness')


def check_readiness(path, policy, source_commit=None):
    from evals.native_reasoning_training import exposure
    from evals.native_control_training import exposure as control_exposure
    from evals.context_native_stream_training import exposure as stream_exposure
    r = json.loads(Path(path).read_text())
    if (r['schema'] != 'dl-native-reasoning32-combined-readiness-v1' or r['passed'] is not True
            or r['policy'] != policy or r['all_parameters_unchanged'] is not True
            or r['optimizer_updates'] != 0 or r['checkpoint_written'] is not False
            or r['training_approved'] is not False or r['original154_components'] != 154
            or r['source_commit'] != (source_commit or os.environ['DL_TASK_COMMIT'])
            or len(r['base154_gradient_sha256']) != 26 or len(r['steps']) != 8
            or r['fixed_parent154_gradient_reused_exactly'] is not True
            or not 0 < r['peak_allocated_bytes'] < 80_000_000_000
            or any(s['frozen_parameter_gradients'] != 0
                   or not 0 < s['combined_gradient_norm'] < float('inf') for s in r['steps'])):
        raise ValueError('Complete retained154/native49/control40/reasoning32 combined readiness required')
    exposure(r['steps'], policy)
    control_exposure(r['steps'], r['control40_policy'])
    stream_exposure(r['steps'], r['native49_policy'])
    return r


def combined_readiness(lm, learned, base_gradients, stream_bound, retained_policy,
                       control_policy, control_bound, policy, bound, args):
    import torch
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_native_stream_training import backward as stream_backward, exposure as stream_exposure
    from evals.native_control_training import backward as control_backward, exposure as control_exposure
    from evals.native_control_stage import write_readiness
    from evals.native_reasoning_training import backward, exposure
    if set(base_gradients) != set(learned) or len(base_gradients) != 26:
        raise ValueError('Actual full154 gradient at the fixed parent required')
    before = tensor_hashes(lm.named_parameters())
    hashes = {n: hashlib.sha256(g.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
              for n, g in base_gradients.items()}
    steps, control_steps = [], []
    def gradient_state():
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):
            raise ValueError('Missing or nonfinite combined native gradient')
        frozen = sum(p.grad is not None for n, p in lm.named_parameters() if n not in learned)
        norm = sum(float(p.grad.detach().double().square().sum()) for p in learned.values()) ** .5
        if frozen or not math.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid gradient ownership or norm')
        return dict(combined_gradient_norm=norm, frozen_parameter_gradients=frozen)
    for index in range(8):
        for n, p in learned.items():
            p.grad = base_gradients[n].detach().clone()
        old = stream_backward(lm, stream_bound, retained_policy['native_stream_auxiliary'], index)
        control = control_backward(lm, control_bound, index)
        retained_state = gradient_state()
        control_steps.append(dict(step=index+1, native_stream_teacher_microbatch=old,
                                  control40_microbatch=control, **retained_state))
        retained_gradients = {n: p.grad.detach().clone() for n, p in learned.items()}
        reasoning = backward(lm, bound, index)
        state = gradient_state()
        delta = {n: p.grad.detach().double() - retained_gradients[n].double() for n, p in learned.items()}
        delta_norm = sum(float(g.square().sum()) for g in delta.values()) ** .5
        dot = sum(float((retained_gradients[n].double() * delta[n]).sum()) for n in learned)
        diagnostic = dict(retained_gradient_norm=retained_state['combined_gradient_norm'],
            incremental_reasoning32_gradient_delta_norm=delta_norm,
            cosine_with_retained_gradient=(dot / (retained_state['combined_gradient_norm'] * delta_norm) if delta_norm else None),
            scope='Fixed-parent accumulated-gradient difference; diagnostic, not an independent loss or learning approval')
        steps.append(dict(step=index+1, native_stream_teacher_microbatch=old, control40_microbatch=control,
                          reasoning32_microbatch=reasoning, **state, gradient_interaction=diagnostic))
        print(json.dumps(dict(reasoning32_combined_partition=index+1, **state, gradient_interaction=diagnostic)), flush=True)
        for p in learned.values():
            p.grad = None
        del retained_gradients, delta
    exposure(steps, policy)
    control_exposure(control_steps, control_policy)
    stream_exposure(control_steps, retained_policy['native_stream_auxiliary'])
    unchanged = before == tensor_hashes(lm.named_parameters())
    peak = torch.cuda.max_memory_allocated()
    control_digest = write_readiness(hashes, control_steps, control_policy, args, unchanged, peak)
    r = dict(schema='dl-native-reasoning32-combined-readiness-v1', policy=policy, steps=steps,
        control40_policy=control_policy, native49_policy=retained_policy['native_stream_auxiliary'],
        passed=unchanged, all_parameters_unchanged=unchanged, optimizer_updates=0,
        checkpoint_written=False, training_approved=False, original154_components=154,
        base154_gradient_sha256=hashes, fixed_parent154_gradient_reused_exactly=True,
        control40_combined_readiness_sha256=control_digest,
        original154_gradient_source_sha256=sha(args.out / 'complete-balanced-preflight.json'),
        native49_readiness_sha256=sha(args.out / 'native-stream49-gradient-readiness.json'),
        source_commit=os.environ['DL_TASK_COMMIT'], peak_allocated_bytes=peak)
    path = args.out / 'reasoning32-combined-readiness.json'
    path.write_text(json.dumps(r, indent=2) + '\n')
    check_readiness(path, policy)
    return control_digest, sha(path)


def verify_checkpoint(proof):
    from evals.context_native_audio49_training import verify_retained_checkpoint
    from evals.native_control_stage import verify_control_layer
    from evals.native_reasoning_training import exposure, validate_gradient
    from evals.native_reasoning_gradient import validate_numerical
    import subprocess
    proof = Path(proof)
    identity, r = verify_retained_checkpoint(proof, SCHEMA, RELOAD_SCHEMA)
    verify_control_layer(proof, identity, r)
    v = json.loads((proof / 'reload-report.json').read_text())
    policy = r['reasoning32']
    exposure(r['steps'], policy)
    ready = check_readiness(proof / 'approved-reasoning32-combined-readiness.json', policy, r['repo_commit'])
    control = json.loads((proof / 'approved-control40-combined-readiness.json').read_text())
    check_bindings(ready, r['control40'], r['assistant_audio_auxiliary']['native_stream_auxiliary'],
                   control, r['control40_combined_readiness_sha256'])
    if (r['combined_recipe']['reasoning32'] != policy
            or ready['control40_policy'] != r['control40']
            or ready['native49_policy'] != r['assistant_audio_auxiliary']['native_stream_auxiliary']
            or ready['control40_combined_readiness_sha256'] != r['control40_combined_readiness_sha256']
            or ready['original154_gradient_source_sha256'] != sha(proof / 'approved-complete-balanced-preflight.json')
            or r['reasoning32_audio_loss_targets'] != policy['assistant_audio_loss_targets_per8']
            or r['reasoning32_text_loss_targets'] != policy['text_loss_targets_per8']
            or r['all_training_assistant_audio_loss_targets'] != r['total_assistant_audio_loss_targets'] + r['reasoning32_audio_loss_targets']
            or v.get('all32_native_reasoning_logits_exact') is not True
            or v['actual_native_reasoning32_logits'] != r['expected_native_reasoning32_logits']
            or set(r['expected_native_reasoning32_logits']) != {t['id'] for t in policy['teachers']}
            or sha(proof / 'approved-reasoning32-combined-readiness.json') != r['reasoning32_combined_readiness_sha256']):
        raise ValueError('Incomplete reasoning32 state, coverage or independent restoration')
    for name, key in (('cache', 'cache_sha256'), ('engine', 'engine_sha256'), ('numerical', 'numerical_sha256'), ('gradient', 'gradient_readiness_sha256')):
        if sha(proof / ('approved-reasoning32-' + name + '.json')) != policy[key]:
            raise ValueError('Changed packaged reasoning32 approval')
    root = Path(__file__).resolve().parents[1]
    for name, key in (('native_reasoning_stage.py', 'stage_implementation_sha256'), ('native_reasoning_training.py', 'implementation_sha256')):
        expected = hashlib.sha256(subprocess.check_output(['git', 'show', r['repo_commit'] + ':evals/' + name], cwd=root)).hexdigest()
        if policy[key] != expected:
            raise ValueError('Changed committed reasoning32 implementation')
    engine = json.loads((proof / 'approved-reasoning32-engine.json').read_text())
    numerical = json.loads((proof / 'approved-reasoning32-numerical.json').read_text())
    gradient = json.loads((proof / 'approved-reasoning32-gradient.json').read_text())
    validate_numerical(numerical, engine, policy['parent'])
    validate_gradient(gradient, engine, numerical, policy['parent'])
    return identity, r
