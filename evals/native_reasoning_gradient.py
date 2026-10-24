"""Complete32 BF16 gradient readiness with the pinned context-span audio mask.

No optimizer or checkpoint is created. Numerical32 must already be complete;
this cannot authorize fitting or replace retained154/native49/control40 gates.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from evals.native_reasoning_source import sha


def validate_numerical(report, manifest, parent):
    """Pure validation, also used by offline adversarial approval tests."""
    rows = report.get('cases', [])
    expected = manifest.get('cases', [])
    if (report.get('schema') != 'dl-native-reasoning32-numerical-v1'
            or report.get('complete32') is not True or report.get('passed') is not True
            or report.get('all_model_parameters_unchanged') is not True
            or report.get('threshold') != .001 or report.get('causal_prefix_error_required') != 0.
            or report.get('optimizer_updates') != 0 or report.get('new_checkpoint_written') is not False
            or report.get('training_approved') is not False or report.get('behavior_qualified') is not False
            or not 0 < report.get('peak_allocated_bytes', float('inf')) < 80_000_000_000
            or report.get('parent') != parent
            or manifest.get('complete32') is not True or manifest.get('passed') is not True
            or manifest.get('all_model_parameters_unchanged') is not True
            or len(rows) != 32 or len(expected) != 32
            or len({r['id'] for r in rows}) != 32
            or [r['id'] for r in rows] != [r['id'] for r in expected]):
        raise ValueError('Complete unchanged32 numerical approval and exact parent required')
    for row, source in zip(rows, expected):
        comparisons = row.get('comparisons', {})
        if (row.get('passed') is not True or row.get('causal_prefix_maximum_error') != 0.
                or not 0 <= row.get('native_training_text_alignment_max_error', float('inf')) <= .001
                or row.get('native_depth_parity', {}).get('passed') is not True
                or set(comparisons) != {'hidden', 'text_logits'}
                or any(not 0 <= v.get('maximum_absolute_error', float('inf')) <= .001 for v in comparisons.values())
                or row['real_text_targets'] != source['ownership']['text_targets']
                or row['pre_span_acoustic_loss_exclusions'] != source['ownership']['actual_delivered_contexts']
                or row['real_acoustic_mask_frames'] != row['real_text_targets'] - row['pre_span_acoustic_loss_exclusions']
                or len(row['assistant_audio_target_counts']) != 8
                or any(type(n) is not int or n <= 0 for n in row['assistant_audio_target_counts'])
                or not math.isfinite(row['text_ce']) or not math.isfinite(row['acoustic_ce'])):
            raise ValueError('Every32 numerical comparison and exact span target ownership required')
    return report


def numerical_gate(folder, engine, parent):
    from evals.native_control_gradient import reference_dependencies
    report = json.loads((folder / 'numerical.json').read_text())
    manifest = json.loads((engine / 'engine-inputs.json').read_text())
    validate_numerical(report, manifest, parent)
    pins = {'code_sha256': 'native_reasoning_numerical.py',
            'reference_code_sha256': 'native_control_reference.py',
            'original_numerical_helpers_sha256': 'native_control_numerical.py'}
    if (report['engine_report_sha256'] != sha(engine / 'engine-inputs.json')
            or any(report[key] != sha(Path(__file__).with_name(name)) for key, name in pins.items())):
        raise ValueError('Changed numerical source or Engine binding')
    reference_dependencies(report['source_commit'])
    return report


def run(lm, a, parent, adapter, numerical):
    import torch
    from evals.native_control_gradient import gradient_summary, output_hashes, implementations
    from evals.personaplex_adapters import adapter_parameters, checkpoint_language_layers
    from evals.context_native_stream_gradient import capture
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss
    from evals.native_control_numerical import target_positions
    from evals.context_execution import snapshot
    from evals.moshi_training_smoke import tensor_hashes
    learned = adapter_parameters(lm)
    if len(learned) != 26 or sum(p.numel() for p in learned.values()) != 1894400:
        raise ValueError('Exact existing adapter required')
    for p in learned.values():
        p.requires_grad_(True)
    if {n for n, p in lm.named_parameters() if p.requires_grad} != set(learned):
        raise ValueError('Changed trainable ownership')
    execution = snapshot(lm)
    if 'torch.bfloat16' not in execution['parameter_dtypes']:
        raise ValueError('Actual native BF16 base required')
    before = tensor_hashes(lm.named_parameters())
    torch.cuda.reset_peak_memory_stats()
    manifest = json.loads((a.engine_inputs / 'engine-inputs.json').read_text())
    result = dict(schema='dl-native-reasoning32-gradient-readiness-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), parent=parent, adapter=adapter,
        execution=execution, cases=[], passed=False, complete32=False,
        engine_report_sha256=sha(a.engine_inputs / 'engine-inputs.json'),
        numerical_report_sha256=sha(a.numerical / 'numerical.json'), code_sha256=sha(Path(__file__)),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        retained_gradient_implementation_pins=implementations(), trainable_parameters=1894400,
        optimizer_updates=0, checkpoint_written=False, training_approved=False, behavior_qualified=False,
        original154_native49_control40_readiness_still_required=True,
        native_execution_profile='BF16 native recorded-call autograd; compile and CUDA graphs disabled as in retained trainer',
        stream_autograd_truncation='Recorded native call boundaries; exact cache values; no cache recomputation',
        acoustic_mask_policy='Exact independently verified pinned upstream context-span audio mask')
    for record, numeric in zip(manifest['cases'], numerical['cases']):
        saved = torch.load(a.engine_inputs / record['id'] / record['file'], weights_only=True, map_location='cpu')
        inputs, mask, targets, audio_mask = (saved[k].cuda() for k in ('inputs', 'mask', 'targets', 'audio_mask'))
        codes = saved['codes'][None].cuda()
        if ((audio_mask & ~mask).any() or int(mask.sum()) != numeric['real_text_targets']
                or int(audio_mask.sum()) != numeric['real_acoustic_mask_frames']):
            raise ValueError('Changed approved text/acoustic masks')
        for p in learned.values():
            p.grad = None
        reference = capture(lm, inputs, mask, targets, record['calls'])
        actual = capture(lm, inputs, mask, targets, record['calls'], backward=True)
        if (reference['call_output_hashes'] != actual['call_output_hashes']
                or reference['balanced_spoken_clock_ce'] != actual['balanced_spoken_clock_ce']):
            raise ValueError('Gradient execution changed exact native streamed text outputs')
        text_gradient = gradient_summary(lm, learned)
        previous = {n: p.grad.detach().clone() for n, p in learned.items()}
        with torch.no_grad():
            output = forward(lm, codes)
            expected_hashes = output_hashes(output)
            _, counts = target_positions(audio_mask, codes, output)
            reference_loss, reference_detail = loss(output, codes, audio_mask[None])
            expected_ce = float(reference_loss)
            del output, reference_loss
        if counts != numeric['assistant_audio_target_counts']:
            raise ValueError('Acoustic targets differ from independent numerical approval')
        with checkpoint_language_layers(lm):
            output = forward(lm, codes)
            if output_hashes(output) != expected_hashes:
                raise ValueError('Gradient execution changed exact native depth/text outputs')
            acoustic, detail = loss(output, codes, audio_mask[None])
            if (float(acoustic.detach()) != expected_ce or detail != reference_detail
                    or detail['assistant_audio_targets'] != counts):
                raise ValueError('BF16 acoustic loss or approved span target ownership changed')
            acoustic.backward()
        combined = gradient_summary(lm, learned, previous)
        result['cases'].append(dict(id=record['id'], family=record['family'],
            exact_native_streamed_text_outputs=True, exact_native_acoustic_outputs=True,
            text_ce=actual['balanced_spoken_clock_ce'], acoustic_ce=expected_ce,
            real_text_targets=int(mask.sum()), real_acoustic_mask_frames=int(audio_mask.sum()),
            pre_span_acoustic_loss_exclusions=numeric['pre_span_acoustic_loss_exclusions'],
            acoustic_objective=detail, native_calls=actual['native_call_count'],
            text_gradient=text_gradient, combined_gradient=combined, passed=True))
        (a.out / 'gradient.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(id=record['id'], complete=len(result['cases']), expected=32)), flush=True)
        for p in learned.values():
            p.grad = None
        del saved, inputs, mask, targets, audio_mask, codes, previous, output, acoustic
    unchanged = before == tensor_hashes(lm.named_parameters()) and execution == snapshot(lm)
    peak = torch.cuda.max_memory_allocated()
    result.update(complete32=len(result['cases']) == 32,
        all_parameters_and_execution_unchanged=unchanged, peak_allocated_bytes=peak)
    result['passed'] = result['complete32'] and unchanged and peak < 80_000_000_000
    (a.out / 'gradient.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise ValueError('Complete32 BF16 gradient readiness failed')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('inputs', 'version', 'semantic', 'data', 'model', 'codec-source', 'interleaver-source',
              'cache', 'assets', 'proof', 'engine-inputs', 'numerical', 'out'):
        p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args()
    from types import SimpleNamespace
    from evals import native_reasoning_engine as engine
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.native_control_numerical import load_model
    engine.verify(SimpleNamespace(**{**vars(a), 'out': a.engine_inputs}))
    parent, _ = verify_checkpoint(a.proof)
    numerical = numerical_gate(a.numerical, a.engine_inputs, parent)
    a.out.mkdir(parents=True, exist_ok=False)
    lm, loaded_parent, adapter = load_model(SimpleNamespace(**{**vars(a), 'base': a.model}))
    if loaded_parent != parent:
        raise ValueError('Loaded parent differs from preflight')
    from contextspan.moshi.utils.compile import no_compile, no_cuda_graph
    with no_compile(), no_cuda_graph():
        return run(lm, a, parent, adapter, numerical)


if __name__ == '__main__':
    main()
