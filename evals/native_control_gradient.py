"""Complete40 BF16 gradient readiness; no optimizer, checkpoint or adoption.

Native recorded-call text gradients retain the exact streaming forward values.
Assistant acoustic gradients use the independently checked native depth layout.
This is additional readiness, not a replacement for the complete154/old49 gates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from data.synthetic.native_control_render import sha256


def implementations():
    names = ('native_control_gradient.py', 'native_control_numerical.py', 'native_control_engine.py',
             'context_native_stream_gradient.py', 'context_native_audio49_training.py',
             'context_native_audio49_loss.py', 'context_adapter_load.py', 'personaplex_adapters.py',
             'personaplex_depth.py', 'context_execution.py')
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def numerical_gate(folder, engine, parent):
    path = folder / 'numerical.json'
    report = json.loads(path.read_text())
    current = json.loads((engine / 'engine-inputs.json').read_text())
    if (report['schema'] != 'dl-native-control40-numerical-v1'
            or report['complete40'] is not True or report['passed'] is not True
            or report['all_model_parameters_unchanged'] is not True
            or report['threshold'] != .001 or report['causal_prefix_error_required'] != 0.
            or report['optimizer_updates'] != 0 or report['new_checkpoint_written'] is not False
            or report['training_approved'] is not False or report['behavior_qualified'] is not False
            or report['peak_allocated_bytes'] >= 80_000_000_000
            or report['engine_report_sha256'] != sha256(engine / 'engine-inputs.json')
            or report['code_sha256'] != sha256(Path(__file__).with_name('native_control_numerical.py'))
            or report['reference_code_sha256'] != sha256(Path(__file__).with_name('native_control_reference.py'))
            or report['parent'] != parent or len(report['cases']) != 40
            or [r['id'] for r in report['cases']] != [r['id'] for r in current['cases']]
            or any(r['passed'] is not True or r['causal_prefix_maximum_error'] != 0.
                   or r['native_training_text_alignment_max_error'] > .001
                   or not r['native_depth_parity']['passed']
                   or any(c['maximum_absolute_error'] > .001 for c in r['comparisons'].values())
                   for r in report['cases'])):
        raise ValueError('Independent unchanged complete40 native numerical approval required')
    return report


def output_hashes(output):
    values = (output.logits, output.mask, output.text_logits, output.text_mask)
    return [hashlib.sha256(t.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()
            for t in values]


def gradient_summary(lm, learned, previous=None):
    import torch
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):
        raise ValueError('Missing or nonfinite native adapter gradient')
    if any(p.grad is not None for n, p in lm.named_parameters() if n not in learned):
        raise ValueError('Frozen parameter acquired a gradient')
    norm = sum(float(p.grad.detach().double().square().sum()) for p in learned.values()) ** .5
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Dead native gradient')
    result = dict(combined_gradient_norm=norm, all26_gradients_present_and_finite=True,
                  frozen_parameter_gradients=0)
    if previous is not None:
        text = [n for n in learned if n.startswith('text_linear.')]
        if len(text) != 2 or any(not torch.equal(learned[n].grad, previous[n]) for n in text):
            raise ValueError('Assistant acoustic loss changed the text-head gradient')
        delta = sum(float((p.grad.detach().double() - previous[n].double()).square().sum())
                    for n, p in learned.items() if n not in text) ** .5
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError('Assistant acoustic objective has no finite temporal gradient')
        result.update(acoustic_temporal_gradient_delta_norm=delta, acoustic_text_head_gradient_delta_exactly_zero=True)
    return result


def run(lm, engine, numerical, out, parent, adapter):
    import torch
    from evals.personaplex_adapters import adapter_parameters, checkpoint_language_layers
    from evals.context_native_stream_gradient import capture
    from evals.context_native_audio49_training import forward
    from evals.context_native_audio49_loss import loss
    from evals.native_control_numerical import target_positions
    from evals.context_execution import snapshot
    from evals.moshi_training_smoke import tensor_hashes
    learned = adapter_parameters(lm)
    if len(learned) != 26 or sum(p.numel() for p in learned.values()) != 1894400:
        raise ValueError('Exact existing 1,894,400-parameter adapter required')
    for p in learned.values():
        p.requires_grad_(True)
    if {n for n, p in lm.named_parameters() if p.requires_grad} != set(learned):
        raise ValueError('Changed trainable ownership')
    execution = snapshot(lm)
    if 'torch.bfloat16' not in execution['parameter_dtypes']:
        raise ValueError('Actual native BF16 base required')
    before = tensor_hashes(lm.named_parameters())
    torch.cuda.reset_peak_memory_stats()
    manifest = json.loads((engine / 'engine-inputs.json').read_text())
    result = dict(schema='dl-native-control40-gradient-readiness-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  parent=parent, adapter=adapter, execution=execution, cases=[], passed=False, complete40=False,
                  engine_report_sha256=sha256(engine / 'engine-inputs.json'), numerical_report_sha256=sha256(numerical / 'numerical.json'),
                  code_sha256=sha256(Path(__file__)), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  implementation_pins=implementations(),
                  trainable_parameters=1894400, optimizer_updates=0, checkpoint_written=False,
                  native_execution_profile='BF16 native recorded-call autograd; compile and CUDA graphs disabled as in retained trainer',
                  training_approved=False, behavior_qualified=False, original154_and_old49_readiness_still_required=True,
                  stream_autograd_truncation='Recorded native call boundaries; exact cache values; no cache recomputation')
    for record in manifest['cases']:
        saved = torch.load(engine / record['id'] / record['file'], weights_only=True, map_location='cpu')
        inputs, mask, targets, codes = (saved['inputs'].cuda(), saved['mask'].cuda(), saved['targets'].cuda(), saved['codes'][None].cuda())
        for p in learned.values(): p.grad = None
        reference = capture(lm, inputs, mask, targets, record['calls'])
        actual = capture(lm, inputs, mask, targets, record['calls'], backward=True)
        if reference['call_output_hashes'] != actual['call_output_hashes'] or reference['balanced_spoken_clock_ce'] != actual['balanced_spoken_clock_ce']:
            raise ValueError('Gradient execution changed exact native streamed text outputs')
        text_gradient = gradient_summary(lm, learned)
        previous = {n: p.grad.detach().clone() for n, p in learned.items()}
        with torch.no_grad():
            output = forward(lm, codes)
            expected_hashes = output_hashes(output)
            _, counts = target_positions(mask, codes, output)
            reference_loss, reference_detail = loss(output, codes, mask[None])
            expected_ce = float(reference_loss)
            del output, reference_loss
        with checkpoint_language_layers(lm):
            output = forward(lm, codes)
            if output_hashes(output) != expected_hashes:
                raise ValueError('Gradient execution changed exact native depth/text outputs')
            acoustic, detail = loss(output, codes, mask[None])
            if float(acoustic.detach()) != expected_ce or detail != reference_detail or detail['assistant_audio_targets'] != counts:
                raise ValueError('BF16 acoustic loss or all-frame target ownership changed')
            acoustic.backward()
        combined = gradient_summary(lm, learned, previous)
        row = dict(id=record['id'], family=record['family'], exact_native_streamed_text_outputs=True,
                   exact_native_acoustic_outputs=True, text_ce=actual['balanced_spoken_clock_ce'],
                   acoustic_ce=expected_ce, acoustic_objective=detail, native_calls=actual['native_call_count'],
                   text_gradient=text_gradient, combined_gradient=combined, passed=True)
        result['cases'].append(row)
        (out / 'gradient.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        for p in learned.values(): p.grad = None
        del saved, inputs, mask, targets, codes, previous, output, acoustic
    unchanged = before == tensor_hashes(lm.named_parameters()) and execution == snapshot(lm)
    peak = torch.cuda.max_memory_allocated()
    result.update(complete40=len(result['cases']) == 40 and len({r['id'] for r in result['cases']}) == 40,
                  all_parameters_and_execution_unchanged=unchanged, peak_allocated_bytes=peak)
    result['passed'] = result['complete40'] and unchanged and peak < 80_000_000_000 and all(r['passed'] for r in result['cases'])
    (out / 'gradient.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise ValueError('Complete40 BF16 gradient readiness failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'engine-inputs', 'numerical', 'assets', 'base', 'proof', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    from evals.native_control_engine import verify
    from evals.context_rollout_conditioning import verify_checkpoint
    verify(args.cache, args.engine_inputs)
    parent, _ = verify_checkpoint(args.proof)
    numerical_gate(args.numerical, args.engine_inputs, parent)
    args.out.mkdir(parents=True, exist_ok=False)
    from evals.native_control_numerical import load_model
    lm, parent, adapter = load_model(args)
    from contextspan.moshi.utils.compile import no_compile, no_cuda_graph
    with no_compile(), no_cuda_graph():
        run(lm, args.engine_inputs, args.numerical, args.out, parent, adapter)


if __name__ == '__main__':
    main()
