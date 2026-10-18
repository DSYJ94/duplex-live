"""Bounded native assistant-text/audio audit, eight updates, independent restore.

This is a new experiment, not an extension of the old text-only proof schema.
No held-out or owner recording is read. The previous520 native adapter supplies
initial weights; optimizer, update count and objective explicitly restart.
"""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evals.moshi_model_download import sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def implementations():
    names = ('context_native_joint.py', 'context_native_joint_data.py',
        'context_native_joint_loss.py', 'context_native_teacher_data.py',
        'context_teacher_native_approval.py', 'context_training_data.py',
        'context_observed_teacher.py', 'context_training_parity.py',
        'personaplex_adapters.py', 'personaplex_depth.py', 'context_training_state.py',
        'context_adapter_load.py')
    return {n: sha256(Path(__file__).with_name(n)) for n in names}


def verify_inference(folder):
    """Bind native inference to this exact experiment and independent restore."""
    import math
    folder = Path(folder)
    read = lambda n: json.loads((folder / n).read_text())
    r, t, restored = read('report.json'), read('training-proof.json'), read('reload-report.json')
    data, settings, audit = read('data.json'), read('settings.json'), read('approved-joint-audit.json')
    if (r['schema'] != 'dl-native-assistant-joint-proof-v1' or not r['passed'] or not t['passed']
            or r['optimizer_steps'] != 8 or t['optimizer_steps'] != 8 or not r['new_optimizer']
            or r['previous_native_updates'] != 520 or r['assistant_audio_heads'] != 8
            or r['implementation_sha256'] != implementations()
            or settings != t['settings'] or settings != audit['settings']
            or r['checkpoint_sha256'] != t['adapter_sha256']
            or r['checkpoint_sha256'] != sha256(folder / 'adapters.safetensors')
            or r['training_state_sha256'] != t['training_state_sha256']
            or r['training_state_sha256'] != sha256(folder / 'training-state.pt')
            or r['data_sha256'] != settings['data_sha256']
            or r['data_sha256'] != sha256(folder / 'data.json')
            or r['joint_audit_sha256'] != sha256(folder / 'approved-joint-audit.json')
            or r['training_proof_sha256'] != sha256(folder / 'training-proof.json')
            or restored['training_proof_sha256'] != r['training_proof_sha256']
            or restored['checkpoint_sha256'] != r['checkpoint_sha256']
            or restored['data_sha256'] != r['data_sha256']
            or restored['pid'] == restored['training_pid']
            or not all(restored[k] for k in ('passed', 'fresh_process', 'optimizer_exact',
                'rng_restored', 'logits_exact', 'source_unchanged'))
            or not t['source_unchanged'] or not all(math.isfinite(x) and x > 0 for x in t['gradient_maxima'].values())
            or t['peak_allocated_bytes'] >= 80_000_000_000
            or not audit['passed'] or audit['optimizer_steps'] != 0 or len(audit['cases']) != 4
            or not audit['all25_input_target_masks_verified'] or not data['all25_native_inputs_exact']
            or len(data['cases']) != 25 or data['heldout_labels'] or data['owner_data_used']
            or any(r[k] or data[k] for k in ('user_audio_loss_targets', 'prefix_loss_targets', 'supplied_span_loss_targets'))
            or any(x['user_audio_loss_targets'] or x['prefix_loss_targets'] or x['supplied_span_loss_targets']
                or x['assistant_audio_targets_per_head'] < 1 for x in data['cases'])):
        raise ValueError('Unverified native assistant-only joint inference artifact')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'base', 'parent', 'approval', 'scope', 'numerical', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--cache', type=Path, action='append', required=True)
    p.add_argument('--phase', choices=('audit', 'train', 'restore'), required=True)
    p.add_argument('--audit', type=Path)
    a = p.parse_args()
    if a.phase != 'restore':
        a.out.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors.torch import load_file, save_file
    from evals.context_spanning_download import SOURCE_COMMIT, MODEL_REVISION, WEIGHT_SHA256
    from evals.personaplex_train import MODEL_HASHES
    from evals.context_spanning_probe import audited_loads
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.context_execution import math_attention
    from evals.context_training_parity import precise_native, depth_parity
    from evals.personaplex_depth import native_depth_context
    from evals.personaplex_adapters import install_adapters, adapter_parameters, checkpoint_language_layers
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_native_joint_data import load as load_data
    from evals.context_native_joint_loss import loss
    from evals.context_training_state import save as save_state, restore as restore_state
    from training.control.train_context import assert_equal, cpu_tree

    torch.backends.cudnn.enabled = False
    math_attention()
    assets = json.loads(a.assets.read_text())
    source, weights = Path(assets['source']), Path(assets['path'])
    if (assets['source_commit'] != SOURCE_COMMIT or assets['revision'] != MODEL_REVISION
            or subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT
            or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)):
        raise ValueError('Changed or dirty pinned native source')
    if assets['files']['context_spanning_7b.pt']['sha256'] != WEIGHT_SHA256:
        raise ValueError('Changed published native checkpoint identity')
    for n, entry in assets['files'].items():
        if sha256(weights / n) != entry['sha256']:
            raise ValueError('Changed native assets')
    for n, expected in MODEL_HASHES.items():
        if sha256(a.base / n) != expected:
            raise ValueError('Changed PersonaPlex base assets')
    parent, parent_report = verify_checkpoint(a.parent)
    os.environ.update(CS_BASE_DIR=str(a.base), CS_WEIGHTS_DIR=str(weights), CS_VOICES_DIR=str(weights / 'voices'))
    driver = a.out / ('driver-restore' if a.phase == 'restore' else 'driver')
    os.environ['TRITON_LIBCUDA_PATH'] = subprocess.check_output(
        [sys.executable, '-m', 'evals.cuda_linker_path', '--out', str(driver)], text=True).strip()
    sys.path.insert(0, str(source))
    from contextspan.model.weights import load_model
    checkpoint = weights / 'context_spanning_7b.pt'
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):
        raise ValueError('Unsafe native checkpoint globals')
    loads = []
    with audited_loads(loads):
        lm, mimi, spm = load_model(str(checkpoint), 'cuda')
    final = [x for x in loads if x['module_identity'] == id(lm)][-1]
    if not final['strict'] or any(x['missing'] or x['unexpected'] for x in loads) or lm.dep_q != 16:
        raise ValueError('Incomplete pinned native checkpoint load')
    lm.eval().requires_grad_(False)
    del mimi
    torch.cuda.empty_cache()
    voice = torch.load(a.parent / 'voice.pt', map_location='cpu', weights_only=True)['codes']
    data, batches = load_data(a, spm, voice)
    data_text = json.dumps(data, indent=2) + '\n'
    data_sha = hashlib.sha256(data_text.encode()).hexdigest()
    settings = dict(schema='dl-native-assistant-joint-settings-v1', source_commit=os.environ['DL_TASK_COMMIT'],
        implementations=implementations(), assets_sha256=sha256(a.assets), data_sha256=data_sha,
        parent=parent, new_optimizer=True, previous_native_updates=520, updates=8,
        learning_rate=1e-5, weight_decay=0., clip_norm=1., audio_weight=1., seed=42424242,
        last_layers=4, rank=8, scaling=2., precision='fp32/math/no-cudnn', user_audio_loss_targets=0)
    if a.phase == 'restore':
        if (json.loads((a.out / 'settings.json').read_text()) != settings
                or (a.out / 'data.json').read_text() != data_text):
            raise ValueError('Independent restore source, objective or data differs')
    else:
        (a.out / 'data.json').write_text(data_text)
        write(a.out / 'settings.json', settings)
        shutil.copyfile(a.parent / 'voice.pt', a.out / 'voice.pt')
        write(a.out / 'source-load.json', dict(loads=loads, final_model_strict=True, all_missing_unexpected_empty=True))
    kinds = ('private', 'simple', 'clarify', 'cancel')
    probes = [next(x for x in batches if x['record']['teacher']['kind'] == k) for k in kinds]

    def on_gpu(row):
        return {k: row[k].cuda() for k in ('codes', 'text_mask', 'audio_mask')}

    def forward(batch):
        with native_depth_context(lm):
            return lm.forward_train(batch['codes'])

    def fingerprints():
        result = {}
        with torch.no_grad():
            for row in probes:
                batch = on_gpu(row)
                out = forward(batch)
                # Complete-vocabulary samples over ten fixed real frames in every task kind.
                index = torch.where(batch['text_mask'][0])[0][:10]
                for key, tensor, valid in [('text', out.text_logits[:, :, index], out.text_mask[:, :, index]),
                        ('audio', out.logits[:, :8, index], out.mask[:, :8, index])]:
                    selected = tensor[valid]
                    if not torch.isfinite(selected).all():
                        raise ValueError('Nonfinite valid native restoration probe')
                    payload = selected.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    result[row['record']['teacher']['id'] + ':' + key] = hashlib.sha256(payload).hexdigest()
                del batch, out
        return result

    with precise_native(lm):
        frozen = dict(lm.named_parameters())
        frozen_before = tensor_hashes(frozen.items())
        versions = {n: x._version for n, x in frozen.items()}
        torch.manual_seed(42424242)
        torch.cuda.manual_seed_all(42424242)
        random.seed(42424242)
        zero_before = fingerprints() if a.phase == 'audit' else None
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            modules = install_adapters(lm, last_layers=4)
        learned = adapter_parameters(lm)
        if set(n for n, x in lm.named_parameters() if x.requires_grad) != set(learned):
            raise ValueError('Unexpected trainable parameter ownership')
        if a.phase == 'audit' and fingerprints() != zero_before:
            raise ValueError('Zero native adapters changed aligned logits')
        warm = load_file(a.parent / 'adapters.safetensors', device='cuda')
        if set(warm) != set(learned) or modules != parent_report['adapter_modules']:
            raise ValueError('Warm-start native adapter coverage differs')
        with torch.no_grad():
            for n, x in learned.items():
                if warm[n].shape != x.shape or not torch.isfinite(warm[n]).all():
                    raise ValueError('Invalid warm-start tensor')
                x.copy_(warm[n])

        if a.phase == 'audit':
            checks = []
            with torch.no_grad():
                for row in probes:
                    batch = on_gpu(row)
                    out = forward(batch)
                    _, z = lm.forward_codes(row['delayed_inputs'].cuda())
                    keep = batch['text_mask'][0] & out.text_mask[0, 0]
                    error = float((z[0, 0, keep] - out.text_logits[0, 0, keep]).abs().max())
                    if error > .001:
                        raise ValueError('Joint forward shifted native text predictions')
                    value, detail = loss(out, batch['codes'], batch['text_mask'], batch['audio_mask'])
                    index = int(torch.where(batch['audio_mask'][0])[0][0])
                    parity = depth_parity(lm, batch['codes'], [index])
                    checks.append(dict(id=row['record']['teacher']['id'], text_alignment_max_error=error,
                        finite_joint_loss=float(value), objective=detail, native_depth_parity=parity))
                    del out, z, batch
            if tensor_hashes(frozen.items()) != frozen_before:
                raise ValueError('Audit changed frozen source parameters')
            write(a.out / 'audit.json', dict(schema='dl-native-assistant-joint-audit-v1', passed=True,
                settings=settings, cases=checks, all25_input_target_masks_verified=True,
                zero_adapter_logits_exact=True, source_parameters_unchanged=True,
                optimizer_steps=0, pid=os.getpid(), native_qualified=False))
            print(json.dumps(dict(audit_passed=True, cases=4, approved_recordings=25)), flush=True)
            return

        if a.audit is None:
            raise ValueError('Real optimization requires an independent native audit')
        audit = json.loads((a.audit / 'audit.json').read_text())
        if (not audit['passed'] or audit['settings'] != settings or audit['optimizer_steps'] != 0
                or not audit['source_parameters_unchanged'] or not audit['all25_input_target_masks_verified']):
            raise ValueError('Missing matching native joint audit')
        optimizer = torch.optim.AdamW(list(learned.values()), lr=1e-5, weight_decay=0.)
        metadata = dict(settings=settings, audit_sha256=sha256(a.audit / 'audit.json'), frozen_hashes=frozen_before)
        if a.phase == 'train':
            shutil.copyfile(a.audit / 'audit.json', a.out / 'approved-joint-audit.json')
        elif sha256(a.out / 'approved-joint-audit.json') != metadata['audit_sha256']:
            raise ValueError('Packaged native joint audit changed')
        if a.phase == 'restore':
            proof = json.loads((a.out / 'training-proof.json').read_text())
            if proof['pid'] == os.getpid() or proof['training_state_sha256'] != sha256(a.out / 'training-state.pt'):
                raise ValueError('Restore must be independent with unchanged saved state')
            state = restore_state(a.out / 'training-state.pt', learned, optimizer, metadata)
            if proof['adapter_sha256'] != sha256(a.out / 'adapters.safetensors'):
                raise ValueError('Saved native inference adapter changed')
            assert_equal(state['adapters'], load_file(a.out / 'adapters.safetensors', device='cpu'))
            assert_equal(state['optimizer'], cpu_tree(optimizer.state_dict()))
            assert_equal(state['torch_cpu_rng'], torch.get_rng_state())
            assert_equal(state['torch_cuda_rng'], torch.cuda.get_rng_state_all())
            assert_equal(state['python_rng'], random.getstate())
            import numpy as np
            np_state = np.random.get_state()
            assert_equal(state['numpy_rng'], dict(name=np_state[0], keys=np_state[1].tolist(),
                position=np_state[2], has_gauss=np_state[3], cached_gauss=np_state[4]))
            if state['completed_steps'] != 8 or fingerprints() != proof['expected_logits']:
                raise ValueError('Native restored predictions or update count differ')
            if tensor_hashes(frozen.items()) != frozen_before:
                raise ValueError('Restore changed pretrained parameters')
            restored = dict(schema='dl-native-assistant-joint-restore-v1', passed=True,
                fresh_process=True, pid=os.getpid(), training_pid=proof['pid'], optimizer_exact=True,
                rng_restored=True, logits_exact=True, source_unchanged=True,
                training_proof_sha256=sha256(a.out / 'training-proof.json'),
                checkpoint_sha256=sha256(a.out / 'adapters.safetensors'),
                data_sha256=sha256(a.out / 'data.json'), native_qualified=False)
            write(a.out / 'restore-proof.json', restored)
            write(a.out / 'reload-report.json', restored)
            write(a.out / 'report.json', dict(schema='dl-native-assistant-joint-proof-v1', passed=True,
                source_commit=SOURCE_COMMIT, repo_commit=settings['source_commit'],
                checkpoint_sha256=restored['checkpoint_sha256'], data_sha256=restored['data_sha256'],
                voice_sha256=sha256(a.out / 'voice.pt'), voice_reference_sha256=parent_report['voice_reference_sha256'],
                frozen_weights_unchanged=True, base_audio_parameters_frozen=True,
                training_state_sha256=proof['training_state_sha256'],
                training_proof_sha256=restored['training_proof_sha256'],
                joint_audit_sha256=metadata['audit_sha256'], implementation_sha256=implementations(),
                adapter_modules=modules, rank=8, scaling=2., last_layers=4, optimizer_steps=8,
                previous_native_updates=520, new_optimizer=True, assistant_audio_heads=8,
                user_audio_loss_targets=0, prefix_loss_targets=0, supplied_span_loss_targets=0,
                native_protocol=parent_report['native_protocol'], quality_qualified=False,
                production_adopted=False))
            print(json.dumps(dict(fresh_native_restore_passed=True)), flush=True)
            return

        seen = {n: 0. for n in learned}
        torch.cuda.reset_peak_memory_stats()
        order = []
        for cycle in range(2):
            for kind in kinds:
                group = [x for x in batches if x['record']['teacher']['kind'] == kind]
                order.append(group[cycle % len(group)])
        with (a.out / 'training.jsonl').open('w') as log:
            for step, row in enumerate(order, 1):
                start = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                batch = on_gpu(row)
                with checkpoint_language_layers(lm):
                    out = forward(batch)
                    value, detail = loss(out, batch['codes'], batch['text_mask'], batch['audio_mask'])
                    value.backward()
                for n, x in learned.items():
                    if x.grad is None or not torch.isfinite(x.grad).all():
                        raise ValueError('Missing or nonfinite native adapter gradient')
                    seen[n] = max(seen[n], float(x.grad.abs().max()))
                if any(x.grad is not None for x in frozen.values()):
                    raise ValueError('Frozen pretrained parameter received gradients')
                norm = torch.nn.utils.clip_grad_norm_(list(learned.values()), 1., error_if_nonfinite=True)
                optimizer.step()
                save_state(a.out / 'training-state.pt', learned, optimizer, step, metadata)
                entry = dict(step=step, id=row['record']['teacher']['id'], loss=float(value.detach()),
                    gradient_norm=float(norm), seconds=time.monotonic() - start, objective=detail)
                log.write(json.dumps(entry) + '\n')
                log.flush()
                print(json.dumps(entry), flush=True)
                del batch, out, value
        if (not all(x > 0 for x in seen.values()) or tensor_hashes(frozen.items()) != frozen_before
                or versions != {n: x._version for n, x in frozen.items()}):
            raise ValueError('Unused adapter or changed source weights')
        peak = torch.cuda.max_memory_allocated()
        if peak >= 80_000_000_000:
            raise ValueError('Native pilot exceeded declared80GB allocation')
        save_file({n: x.detach().cpu().contiguous() for n, x in learned.items()}, a.out / 'adapters.safetensors')
        write(a.out / 'training-proof.json', dict(schema='dl-native-assistant-joint-training-v1',
            passed=True, pid=os.getpid(), settings=settings, optimizer_steps=8, adapter_modules=modules,
            gradient_maxima=seen, frozen_hashes=frozen_before, source_unchanged=True,
            peak_allocated_bytes=peak, training_state_sha256=sha256(a.out / 'training-state.pt'),
            adapter_sha256=sha256(a.out / 'adapters.safetensors'), expected_logits=fingerprints(),
            production_adopted=False, native_qualified=False))


if __name__ == '__main__':
    main()
