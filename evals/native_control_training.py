"""Additional control40 objectives for one bounded eight-update repair pilot.

Every original154 objective and the original49 native-call auxiliary must remain.
This module has no optimizer or CLI and cannot by itself authorize fitting.
"""
from collections import Counter
import json
import math
from pathlib import Path

from data.synthetic.native_control_render import sha256
from evals.native_control_cache import FAMILIES


def schedule(records):
    if (len(records) != 40 or len({r['id'] for r in records}) != 40
            or Counter(r['family'] for r in records) != Counter({f: 8 for f in FAMILIES})):
        raise ValueError('All40 distinct approved teachers and all five balanced families required')
    # Store family-major order explicitly; each selected partition has all five
    # families. Never rely on incidental filename/order to balance the pilot.
    ordered = [r for family in sorted(FAMILIES) for r in records if r['family'] == family]
    return [dict(id=r['id'], family=r['family'], weight_per_epoch=1/40,
                 weight_when_selected=8/40) for r in ordered]


def prepare(cache, engine, numerical, gradient, parent):
    from evals.native_control_engine import verify
    from evals.native_control_gradient import numerical_gate
    from evals.native_control_gradient_binding import verify as verify_implementation_binding
    manifest = verify(cache, engine)
    numerical_gate(numerical, engine, parent)
    path = gradient / 'gradient.json'; ready = json.loads(path.read_text())
    implementation_binding = verify_implementation_binding(ready)
    if (ready['schema'] != 'dl-native-control40-gradient-readiness-v1' or ready['passed'] is not True
            or ready['complete40'] is not True or ready['all_parameters_and_execution_unchanged'] is not True
            or ready['trainable_parameters'] != 1894400 or ready['optimizer_updates'] != 0
            or ready['checkpoint_written'] is not False or ready['training_approved'] is not False
            or ready['behavior_qualified'] is not False or ready['peak_allocated_bytes'] >= 80_000_000_000
            or ready['native_execution_profile'] != 'BF16 native recorded-call autograd; compile and CUDA graphs disabled as in retained trainer'
            or ready['parent'] != parent or len(ready['cases']) != 40
            or ready['engine_report_sha256'] != sha256(engine / 'engine-inputs.json')
            or ready['numerical_report_sha256'] != sha256(numerical / 'numerical.json')
            or ready['code_sha256'] != sha256(Path(__file__).with_name('native_control_gradient.py'))
            or [r['id'] for r in ready['cases']] != [r['id'] for r in manifest['cases']]
            or any(r['passed'] is not True or r['exact_native_streamed_text_outputs'] is not True
                   or r['exact_native_acoustic_outputs'] is not True
                   or r['combined_gradient']['acoustic_text_head_gradient_delta_exactly_zero'] is not True
                   or r['combined_gradient']['frozen_parameter_gradients'] != 0
                   or not 0 < r['combined_gradient']['combined_gradient_norm'] < float('inf')
                   for r in ready['cases'])):
        raise ValueError('Independent complete40 BF16 text/acoustic gradient readiness required')
    ordered = schedule(manifest['cases'])
    policy = dict(mode='native520-control40-native-text-assistant-audio-v1', updates=8,
                  parent=parent, teachers=ordered, text_epoch_mean_weight=1., acoustic_epoch_mean_weight=1.,
                  schedule='Stride8 over explicit family-major order; all40 once, all five families per update',
                  original154_objectives_and_original49_native_auxiliary_unchanged=True,
                  original28_histories_old25_new24_kl1_hidden10_required=True,
                  source_words_audio_receipts_folds_and_heldouts_unchanged=True,
                  prefix_span_user_audio_targets=0, native_text_truncation='recorded call boundaries',
                  implementation_sha256=sha256(Path(__file__)),
                  gradient_readiness_sha256=sha256(path), numerical_sha256=sha256(numerical / 'numerical.json'),
                  gradient_implementation_binding=implementation_binding,
                  engine_sha256=sha256(engine / 'engine-inputs.json'), cache_sha256=sha256(cache / 'cache.json'),
                  extends_rejected_joint_or_unqualified_candidate=False, training_approved=False)
    records = {r['id']: r for r in manifest['cases']}
    return policy, [(r, engine / r['id'] / records[r['id']]['file'], records[r['id']]['calls']) for r in ordered]


def backward(lm, bound, step):
    import torch
    from evals.context_native_stream_gradient import capture
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss
    from evals.personaplex_adapters import checkpoint_language_layers
    from evals.native_control_numerical import target_positions
    if type(step) is not int or not 0 <= step < 8 or len(bound) != 40:
        raise ValueError('Only one complete40 bounded eight-update stage')
    rows = []
    for index in range(step, 40, 8):
        declared, path, calls = bound[index]
        saved = torch.load(path, weights_only=True, map_location='cpu')
        inputs, mask, targets, codes = (saved['inputs'].cuda(), saved['mask'].cuda(), saved['targets'].cuda(), saved['codes'][None].cuda())
        text = capture(lm, inputs, mask, targets, calls, backward=True, weight=declared['weight_when_selected'])
        with checkpoint_language_layers(lm):
            output = forward(lm, codes)
            _, counts = target_positions(mask, codes, output)
            acoustic, detail = loss(output, codes, mask[None])
            if detail['assistant_audio_targets'] != counts:
                raise ValueError('Complete native assistant-only target ownership changed')
            (acoustic * declared['weight_when_selected']).backward()
        rows.append(dict(id=declared['id'], family=declared['family'], weight=declared['weight_when_selected'],
                         text_ce=text['balanced_spoken_clock_ce'], acoustic_ce=float(acoustic.detach()),
                         text_targets=text['valid_targets'], acoustic_objective=detail,
                         prefix_span_user_audio_loss_targets=0))
        del saved, inputs, mask, targets, codes, output, acoustic
    return rows


def exposure(steps, policy):
    if len(steps) != 8 or [r['step'] for r in steps] != list(range(1, 9)):
        raise ValueError('Exactly eight complete pilot updates required')
    seen = []
    for index, step in enumerate(steps):
        actual = step['control40_microbatch']; expected = policy['teachers'][index::8]
        if len(actual) != 5 or {r['family'] for r in actual} != FAMILIES:
            raise ValueError('Each update must include every control family')
        for row, declared in zip(actual, expected):
            counts = row['acoustic_objective']['assistant_audio_targets']
            if (any(row[k] != declared[k] for k in ('id', 'family')) or row['weight'] != declared['weight_when_selected']
                    or row['prefix_span_user_audio_loss_targets'] != 0 or row['text_targets'] < 1
                    or row['acoustic_objective']['user_audio_loss_targets'] != 0
                    or len(counts) != 8 or any(type(n) is not int or n < 1 for n in counts)
                    or not math.isfinite(row['text_ce'] + row['acoustic_ce'])):
                raise ValueError('Changed control target ownership, weight, identity or finite loss')
            seen.append(row['id'])
    if len(seen) != 40 or len(set(seen)) != 40:
        raise ValueError('All40 teachers must be trained exactly once; no subset')
    return True


def fingerprints(lm, bound):
    """All40 native text/audio fingerprints for a separate-process state restore."""
    import hashlib
    import torch
    from evals.context_native_stream_gradient import capture
    from evals.context_native_audio49_training import forward
    from evals.native_control_numerical import target_positions
    if len(bound) != 40 or len({r[0]['id'] for r in bound}) != 40:
        raise ValueError('Independent restoration requires complete40')
    result = {}
    with torch.no_grad():
        for declared, path, calls in bound:
            saved = torch.load(path, weights_only=True, map_location='cpu')
            inputs, mask, targets, codes = (saved['inputs'].cuda(), saved['mask'].cuda(), saved['targets'].cuda(), saved['codes'][None].cuda())
            text = capture(lm, inputs, mask, targets, calls)
            output = forward(lm, codes)
            _, counts = target_positions(mask, codes, output)
            keep = output.mask[:, :8] & mask[None, None]
            selected = output.logits[:, :8][keep].detach()
            if not selected.numel() or not torch.isfinite(selected).all():
                raise ValueError('Invalid native control restoration logits')
            result[declared['id']] = dict(
                native_text_calls_sha256=hashlib.sha256(json.dumps(text['call_output_hashes'], separators=(',', ':')).encode()).hexdigest(),
                assistant_audio_logits_sha256=hashlib.sha256(selected.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                assistant_audio_targets=counts, text_targets=text['valid_targets'])
            print(json.dumps(dict(control40_restore_fingerprint=declared['id'], **result[declared['id']])), flush=True)
            del saved, inputs, mask, targets, codes, output, selected
    return result
