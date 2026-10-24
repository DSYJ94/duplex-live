"""Complete32 supplemental objectives; no optimizer or standalone fit approval.

The retained154/native49/control40 objectives must be integrated separately.
Every update covers all four reasoning families; all32 occur once over eight
updates. Text and acoustic masks remain distinct at provided-context boundaries.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from evals.native_reasoning_cache import FAMILIES
from evals.native_reasoning_source import sha


def schedule(records):
    if (len(records) != 32 or len({r['id'] for r in records}) != 32
            or Counter(r['family'] for r in records) != Counter({f: 8 for f in FAMILIES})):
        raise ValueError('All32 distinct teachers and all four balanced families required')
    ordered = [r for family in sorted(FAMILIES)
               for r in sorted(records, key=lambda r: r['id']) if r['family'] == family]
    return [dict(id=r['id'], family=r['family'], weight_per_epoch=1/32,
                 weight_when_selected=8/32) for r in ordered]


def validate_gradient(ready, manifest, numerical, parent):
    if (ready.get('schema') != 'dl-native-reasoning32-gradient-readiness-v1'
            or ready.get('passed') is not True or ready.get('complete32') is not True
            or ready.get('all_parameters_and_execution_unchanged') is not True
            or ready.get('trainable_parameters') != 1894400 or ready.get('optimizer_updates') != 0
            or ready.get('checkpoint_written') is not False or ready.get('training_approved') is not False
            or ready.get('behavior_qualified') is not False
            or not 0 < ready.get('peak_allocated_bytes', float('inf')) < 80_000_000_000
            or ready.get('parent') != parent
            or ready.get('native_execution_profile') != 'BF16 native recorded-call autograd; compile and CUDA graphs disabled as in retained trainer'
            or len(ready.get('cases', [])) != 32
            or [r['id'] for r in ready['cases']] != [r['id'] for r in manifest['cases']]):
        raise ValueError('Independent complete32 BF16 gradient readiness required')
    for row, numeric in zip(ready['cases'], numerical['cases']):
        combined = row['combined_gradient']
        if (row['passed'] is not True or row['exact_native_streamed_text_outputs'] is not True
                or row['exact_native_acoustic_outputs'] is not True
                or combined['acoustic_text_head_gradient_delta_exactly_zero'] is not True
                or combined['all26_gradients_present_and_finite'] is not True
                or combined['frozen_parameter_gradients'] != 0
                or not 0 < combined['combined_gradient_norm'] < float('inf')
                or not 0 < combined['acoustic_temporal_gradient_delta_norm'] < float('inf')
                or any(row[k] != numeric[k] for k in ('id', 'real_text_targets',
                    'real_acoustic_mask_frames', 'pre_span_acoustic_loss_exclusions'))
                or row['acoustic_objective']['assistant_audio_targets'] != numeric['assistant_audio_target_counts']
                or row['acoustic_objective']['user_audio_loss_targets'] != 0):
            raise ValueError('Every32 finite gradient and exact span target count required')
    return ready


def prepare(a, parent):
    from types import SimpleNamespace
    from evals.native_reasoning_engine import verify
    from evals.native_reasoning_gradient import numerical_gate
    from evals.native_control_gradient import implementations
    manifest = verify(SimpleNamespace(**{**vars(a), 'out': a.engine_inputs}))
    numerical = numerical_gate(a.numerical, a.engine_inputs, parent)
    ready = json.loads((a.gradient / 'gradient.json').read_text())
    validate_gradient(ready, manifest, numerical, parent)
    if (ready['engine_report_sha256'] != sha(a.engine_inputs / 'engine-inputs.json')
            or ready['numerical_report_sha256'] != sha(a.numerical / 'numerical.json')
            or ready['code_sha256'] != sha(Path(__file__).with_name('native_reasoning_gradient.py'))
            or ready['retained_gradient_implementation_pins'] != implementations()):
        raise ValueError('Changed source or independent approval binding')
    teachers = schedule(manifest['cases'])
    counts = {r['id']: dict(text=r['real_text_targets'], audio=r['assistant_audio_target_counts'],
        audio_mask_frames=r['real_acoustic_mask_frames'],
        pre_span_exclusions=r['pre_span_acoustic_loss_exclusions']) for r in numerical['cases']}
    policy = dict(mode='native520-retained154-native49-control40-reasoning32-v1', updates=8,
        parent=parent, teachers=teachers, target_counts=counts,
        text_loss_targets_per8=sum(r['text'] for r in counts.values()),
        assistant_audio_loss_targets_per8=sum(sum(r['audio']) for r in counts.values()),
        text_epoch_mean_weight=1., acoustic_epoch_mean_weight=1.,
        schedule='Stride8 over sorted family-major teachers: all32 once, all four families each update',
        original154_native49_control40_objectives_required=True,
        original28_histories_old25_new24_kl1_hidden10_required=True,
        source_words_audio_receipts_folds_and_heldouts_unchanged=True,
        prefix_span_user_audio_targets=0, native_text_truncation='recorded call boundaries',
        acoustic_mask_policy='Exact independently verified pinned upstream context-span audio mask',
        implementation_sha256=sha(Path(__file__)),
        engine_sha256=sha(a.engine_inputs / 'engine-inputs.json'),
        numerical_sha256=sha(a.numerical / 'numerical.json'),
        gradient_readiness_sha256=sha(a.gradient / 'gradient.json'),
        cache_sha256=sha(a.cache / 'cache.json'),
        extends_rejected_or_unqualified_candidate=False, training_approved=False)
    records = {r['id']: r for r in manifest['cases']}
    bound = [(r, a.engine_inputs / r['id'] / records[r['id']]['file'], records[r['id']]['calls']) for r in teachers]
    return policy, bound


def tensors(path):
    import torch
    saved = torch.load(path, weights_only=True, map_location='cpu')
    inputs, mask, targets, audio_mask = (saved[k].cuda() for k in ('inputs', 'mask', 'targets', 'audio_mask'))
    if (audio_mask & ~mask).any():
        raise ValueError('Audio objective includes non-real frames')
    return inputs, mask, targets, saved['codes'][None].cuda(), audio_mask


def backward(lm, bound, step):
    from evals.context_native_stream_gradient import capture
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss
    from evals.personaplex_adapters import checkpoint_language_layers
    from evals.native_control_numerical import target_positions
    if type(step) is not int or not 0 <= step < 8 or len(bound) != 32:
        raise ValueError('Only complete32 over one bounded eight-update stage')
    rows = []
    for index in range(step, 32, 8):
        declared, path, calls = bound[index]
        inputs, mask, targets, codes, audio_mask = tensors(path)
        text = capture(lm, inputs, mask, targets, calls, backward=True, weight=declared['weight_when_selected'])
        with checkpoint_language_layers(lm):
            output = forward(lm, codes)
            _, counts = target_positions(audio_mask, codes, output)
            acoustic, detail = loss(output, codes, audio_mask[None])
            if detail['assistant_audio_targets'] != counts:
                raise ValueError('Changed native assistant target ownership')
            (acoustic * declared['weight_when_selected']).backward()
        rows.append(dict(id=declared['id'], family=declared['family'], weight=declared['weight_when_selected'],
            text_ce=text['balanced_spoken_clock_ce'], acoustic_ce=float(acoustic.detach()),
            text_targets=text['valid_targets'], audio_mask_frames=int(audio_mask.sum()),
            acoustic_objective=detail, prefix_span_user_audio_loss_targets=0))
        del inputs, mask, targets, codes, audio_mask, output, acoustic
    return rows


def exposure(steps, policy):
    declared = policy['teachers']
    if (schedule(declared) != declared or len(steps) != 8
            or [r['step'] for r in steps] != list(range(1, 9))
            or set(policy['target_counts']) != {r['id'] for r in declared}):
        raise ValueError('Exact complete32 policy and eight updates required')
    seen = []
    for index, step in enumerate(steps):
        rows = step['reasoning32_microbatch']
        if len(rows) != 4 or {r['family'] for r in rows} != FAMILIES:
            raise ValueError('Every update must contain all four reasoning families')
        for row, expected in zip(rows, declared[index::8]):
            counts = policy['target_counts'][expected['id']]
            if (any(row[k] != expected[k] for k in ('id', 'family'))
                    or row['weight'] != expected['weight_when_selected']
                    or row['text_targets'] != counts['text']
                    or row['audio_mask_frames'] != counts['audio_mask_frames']
                    or row['acoustic_objective']['assistant_audio_targets'] != counts['audio']
                    or row['acoustic_objective']['user_audio_loss_targets'] != 0
                    or row['prefix_span_user_audio_loss_targets'] != 0
                    or not math.isfinite(row['text_ce']) or not math.isfinite(row['acoustic_ce'])):
                raise ValueError('Changed exact reasoning32 identity, weight, targets or loss')
            seen.append(row['id'])
    if (len(set(seen)) != 32
            or sum(r['text'] for r in policy['target_counts'].values()) != policy['text_loss_targets_per8']
            or sum(sum(r['audio']) for r in policy['target_counts'].values()) != policy['assistant_audio_loss_targets_per8']):
        raise ValueError('Complete32 totals changed')
    return True


def fingerprints(lm, bound):
    import torch
    from evals.context_native_stream_gradient import capture
    from evals.context_native_audio49_training import forward
    from evals.native_control_numerical import target_positions
    if len(bound) != 32 or len({r[0]['id'] for r in bound}) != 32:
        raise ValueError('Independent restoration requires every32 teacher')
    result = {}
    with torch.no_grad():
        for declared, path, calls in bound:
            inputs, mask, targets, codes, audio_mask = tensors(path)
            text = capture(lm, inputs, mask, targets, calls)
            output = forward(lm, codes)
            _, counts = target_positions(audio_mask, codes, output)
            selected = output.logits[:, :8][output.mask[:, :8] & audio_mask[None, None]].detach()
            if not selected.numel() or not torch.isfinite(selected).all():
                raise ValueError('Invalid native reasoning restoration logits')
            result[declared['id']] = dict(
                native_text_calls_sha256=hashlib.sha256(json.dumps(text['call_output_hashes'], separators=(',', ':')).encode()).hexdigest(),
                assistant_audio_logits_sha256=hashlib.sha256(selected.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                assistant_audio_targets=counts, text_targets=text['valid_targets'], audio_mask_frames=int(audio_mask.sum()))
            print(json.dumps(dict(reasoning32_restore_fingerprint=declared['id'], **result[declared['id']])), flush=True)
            del inputs, mask, targets, codes, audio_mask, output, selected
    return result
