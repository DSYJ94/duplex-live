"""Fixed contextual pilot: nine TRAIN-only proof updates, then fresh324 candidate.

The shell invokes restoration in a second process. No saved in-memory candidate
is scored. Original pretrained parameters and all heldout fixtures stay frozen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

PLAN = Path('evals/fixtures/paired-context-adapter-plan-20261018.json')
DATA_SHA = '62cc4ae1a0bc1f35ec5cd973b9cb80b8150316efeb56f016af0e7596e42bcbab'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def assert_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            raise ValueError('Restored tensor differs')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('Restored dictionary inventory differs')
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('Restored sequence differs')
        for x, y in zip(a, b):
            assert_equal(x, y)
    elif a != b:
        raise ValueError('Restored value differs')


def encode(row, tokenizer):
    prompt = tokenizer.apply_chat_template(row['student_messages'], tools=row['student_tools'], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False) + row['student_assistant_prefill']
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    if prefix != tokenizer.encode(prompt) or not row['target_native_text'].startswith(row['student_assistant_prefill']):
        raise ValueError('Native prefix alignment differs')
    suffix = row['target_native_text'][len(row['student_assistant_prefill']):]
    target = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prefix or not target or len(prefix + target) > 4096:
        raise ValueError('Training token bounds exceeded')
    return dict(id=row['id'], kind=row['kind'], inputs=(prefix + target)[:-1], targets=target, prefix_tokens=len(prefix))


def logits(model, example):
    import torch
    ids = torch.tensor([example['inputs']], device=model.device)
    result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=len(example['targets'])).logits[0]
    if result.shape != (len(example['targets']), model.config.text_config.vocab_size):
        raise ValueError('Causal target alignment differs')
    return result


def logit_hashes(model, examples):
    import torch
    result = {}
    with torch.no_grad():
        for ex in examples:
            value = logits(model, ex).detach().contiguous()
            result[ex['id']] = dict(shape=list(value.shape), dtype=str(value.dtype),
                                    sha256=hashlib.sha256(value.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
    return result


def optimizer_for(adapters):
    import torch
    return torch.optim.AdamW(list(adapters.values()), lr=2e-5, weight_decay=0.)


def validate_preflight(folder, assets_sha, implementation):
    from training.control.context_adapter import sha
    folder = Path(folder)
    trained = json.loads((folder / 'training-proof.json').read_text())
    restored = json.loads((folder / 'restore-proof.json').read_text())
    if (trained['phase'] != 'preflight' or trained['optimizer_steps'] != 9 or not restored['fresh_process_restore_passed']
            or trained['pid'] == restored['pid'] or trained['base_assets_sha256'] != assets_sha
            or trained['plan_sha256'] != sha(PLAN) or trained['implementation_sha256'] != implementation
            or restored['training_proof_sha256'] != sha(folder / 'training-proof.json')
            or restored['training_state_sha256'] != sha(folder / 'candidate-adapter/training-state.pt')
            or restored['adapter_weights_sha256'] != sha(folder / 'candidate-adapter/adapter.safetensors')
            or not trained['all_adapter_tensors_received_nonzero_gradients'] or not trained['all_source_hashes_and_versions_unchanged']
            or not trained['zero_adapter_logits_and_rng_exact'] or trained['peak_allocated_bytes'] >= 80_000_000_000):
        raise ValueError('Actual nine-update and independent restore proofs required')
    return dict(training_proof_sha256=sha(folder / 'training-proof.json'), restore_proof_sha256=sha(folder / 'restore-proof.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'assets', 'kernel-assets', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--phase', choices=('preflight', 'full'), required=True)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--restore-only', action='store_true')
    a = p.parse_args()
    from training.control.paired_data import verify
    from training.control.context_adapter import sha
    rows, heldout, manifest = verify(a.data)
    if sha(a.data / 'manifest.json') != DATA_SHA or len(rows) != 108 or len(heldout) != 54:
        raise ValueError('Requires the complete approved paired corpus')
    plan = json.loads(PLAN.read_text())
    if plan['teacher_manifest_sha256'] != DATA_SHA or plan['fixed_full_budget_updates'] != 324:
        raise ValueError('Changed predeclared recipe')
    implementation = {f: sha(f) for f in ('training/control/context_adapter.py', 'training/control/train_context.py', 'training/control/context_evaluate.py')}
    assets_sha = sha(a.assets)
    preflight_proof = None
    if a.phase == 'full':
        if a.preflight is None:
            raise ValueError('Full run requires completed preflight')
        preflight_proof = validate_preflight(a.preflight, assets_sha, implementation)
    if not a.restore_only:
        a.out.mkdir(parents=True, exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor = Qwen35ResponseExecutor(a.assets, kernel_assets=a.kernel_assets, pending_speech_protocol='explicit-actions', compact_replies=True)
    warm_long_prefill(executor)
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from training.control.context_adapter import install, save, restore
    from training.control.head_adapter import parameter_hashes
    from training.control.context_evaluate import evaluate, retention
    torch.manual_seed(42424242)
    torch.cuda.manual_seed_all(42424242)
    rng = random.Random(42424242)
    model = executor.model
    source = dict(model.named_parameters())
    versions = {n: p._version for n, p in source.items()}
    before = parameter_hashes(source)
    examples = [encode(r, executor.tokenizer) for r in rows]
    from training.control.paired_curriculum import KINDS
    probes = [next(e for e in examples if e['kind'] == kind) for kind in KINDS]
    settings = dict(phase=a.phase, seed=42424242, learning_rate=2e-5, weight_decay=0., gradient_clip=1., batch_size=1,
                    epochs=3 if a.phase == 'full' else None, steps=324 if a.phase == 'full' else 9,
                    data_manifest_sha256=DATA_SHA, base_assets_sha256=assets_sha, plan_sha256=sha(PLAN), implementation_sha256=implementation,
                    source_commit=os.environ.get('DL_TASK_COMMIT'), preflight_proof=preflight_proof)
    with torch.cuda.device(model.device), sdpa_kernel(SDPBackend.MATH):
        if a.restore_only:
            training = json.loads((a.out / 'training-proof.json').read_text())
            if training['pid'] == os.getpid() or training['settings'] != settings:
                raise ValueError('Restore must be an independent identical-source process')
            if json.loads((a.out / 'source-parameters-before.json').read_text()) != before:
                raise ValueError('Fresh base inventory differs')
            adapters = install(model)
            meta = restore(a.out / 'candidate-adapter', adapters, assets_sha)
            state_meta = json.loads((a.out / 'candidate-adapter/training-state.json').read_text())
            state_path = a.out / 'candidate-adapter/training-state.pt'
            if (sha(state_path) != state_meta['state_sha256'] or state_meta['adapter_weights_sha256'] != meta['weights_sha256']
                    or state_meta['settings_sha256'] != sha(a.out / 'settings.json') or meta['provenance']['settings_sha256'] != sha(a.out / 'settings.json')):
                raise ValueError('Unbound optimizer or settings checkpoint')
            state = torch.load(state_path, map_location='cpu', weights_only=True)
            optimizer = optimizer_for(adapters)
            optimizer.load_state_dict(state['optimizer'])
            assert_equal(state['optimizer'], cpu_tree(optimizer.state_dict()))
            torch.set_rng_state(state['torch_rng_state'])
            torch.cuda.set_rng_state_all(state['cuda_rng_states'])
            rng.setstate(state['python_rng_state'])
            assert_equal(state['torch_rng_state'], torch.get_rng_state())
            assert_equal(state['cuda_rng_states'], torch.cuda.get_rng_state_all())
            assert_equal(state['python_rng_state'], rng.getstate())
            actual = logit_hashes(model, probes)
            if actual != training['restoration_logit_hashes'] or state['steps'] != settings['steps']:
                raise ValueError('Fresh restored logits or step inventory differs')
            if before != parameter_hashes(source) or versions != {n: p._version for n, p in source.items()}:
                raise ValueError('Restoration altered pretrained source')
            proof = dict(fresh_process_restore_passed=True, pid=os.getpid(), training_pid=training['pid'],
                         training_proof_sha256=sha(a.out / 'training-proof.json'), training_state_sha256=sha(state_path),
                         adapter_weights_sha256=meta['weights_sha256'], optimizer_state_exact=True, rng_exact=True, logits_exact=True)
            write(a.out / 'restore-proof.json', proof)
            if a.phase == 'full':
                summary = evaluate(executor, heldout, a.out / 'restored-adapted')
                retained = retention(a.out / 'baseline', a.out / 'restored-adapted')
                write(a.out / 'report.json', dict(summary=summary, retention=retained, restore=proof,
                      full_worker_generations_completed=True, heldout_optimizer_rows=0, heldout_checkpoint_selection=False,
                      production_adopted=False, native_duplex_qualified=False, speech_qualified=False))
            print(json.dumps(proof), flush=True)
            return
        write(a.out / 'settings.json', settings)
        write(a.out / 'source-load.json', executor.proof)
        write(a.out / 'source-parameters-before.json', before)
        write(a.out / 'feature-inventory.json', [dict(id=e['id'], kind=e['kind'], prompt_tokens=e['prefix_tokens'], target_tokens=len(e['targets'])) for e in examples])
        if a.phase == 'full':
            evaluate(executor, heldout, a.out / 'baseline')
        zero_before = logit_hashes(model, probes)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        adapters = install(model)
        assert_equal(cpu_rng, torch.get_rng_state())
        assert_equal(cuda_rng, torch.cuda.get_rng_state_all())
        if zero_before != logit_hashes(model, probes):
            raise ValueError('Zero adapters changed actual model logits')
        write(a.out / 'zero-adapter-logits.json', zero_before)
        optimizer = optimizer_for(adapters)
        grad_seen = {n: 0. for n in adapters}
        torch.cuda.reset_peak_memory_stats(model.device)
        order = list(probes)
        if a.phase == 'full':
            order = []
            for _ in range(3):
                epoch = list(examples)
                rng.shuffle(epoch)
                order.extend(epoch)
        with (a.out / 'training.jsonl').open('w') as log:
            for step, example in enumerate(order, 1):
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                values = logits(model, example)
                target = torch.tensor(example['targets'], dtype=torch.long, device=model.device)
                loss = torch.nn.functional.cross_entropy(values.float(), target)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite context loss')
                loss.backward()
                for name, param in adapters.items():
                    if param.grad is None or not torch.isfinite(param.grad).all():
                        raise ValueError('Missing/nonfinite context gradient: ' + name)
                    grad_seen[name] = max(grad_seen[name], float(param.grad.abs().max()))
                if any(p.grad is not None for p in source.values()):
                    raise ValueError('A source parameter received gradients')
                norm = torch.nn.utils.clip_grad_norm_(list(adapters.values()), 1., error_if_nonfinite=True)
                optimizer.step()
                entry = dict(step=step, id=example['id'], kind=example['kind'], loss=float(loss.detach()), gradient_norm=float(norm), seconds=time.monotonic()-started)
                log.write(json.dumps(entry) + '\n')
                log.flush()
                print(json.dumps(entry), flush=True)
                del values, loss, target
        if len(order) != settings['steps'] or not all(v > 0 for v in grad_seen.values()):
            raise ValueError('Incomplete updates or adapter tensors without nonzero gradients')
        if before != parameter_hashes(source) or versions != {n: p._version for n, p in source.items()}:
            raise ValueError('Frozen source hashes/versions changed')
        peak = torch.cuda.max_memory_allocated(model.device)
        if peak >= 80_000_000_000:
            raise ValueError('Predeclared80GB peak allocation exceeded')
        hashes = logit_hashes(model, probes)
        meta = save(a.out / 'candidate-adapter', adapters, assets_sha, dict(data_manifest_sha256=DATA_SHA, settings_sha256=sha(a.out / 'settings.json'), optimizer_steps=len(order)))
        state = dict(optimizer=cpu_tree(optimizer.state_dict()), torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all(), python_rng_state=rng.getstate(), steps=len(order))
        state_path = a.out / 'candidate-adapter/training-state.pt'
        torch.save(state, state_path)
        write(a.out / 'candidate-adapter/training-state.json', dict(state_sha256=sha(state_path), adapter_weights_sha256=meta['weights_sha256'], settings_sha256=sha(a.out / 'settings.json')))
        proof = dict(phase=a.phase, pid=os.getpid(), settings=settings, optimizer_steps=len(order),
                     base_assets_sha256=assets_sha, implementation_sha256=implementation, plan_sha256=sha(PLAN),
                     zero_adapter_logits_and_rng_exact=True, all_source_hashes_and_versions_unchanged=True,
                     all_adapter_tensors_received_nonzero_gradients=True, gradient_maxima=grad_seen,
                     peak_allocated_bytes=peak, restoration_logit_hashes=hashes, production_adopted=False)
        write(a.out / 'training-proof.json', proof)
        print(json.dumps(dict(training_completed=True, optimizer_steps=len(order), peak_allocated_bytes=peak, independent_restore_pending=True)), flush=True)


if __name__ == '__main__':
    main()
