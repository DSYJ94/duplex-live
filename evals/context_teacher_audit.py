"""Separate teacher-history fit from failed free-running speech/control behavior."""


def audit(lm, spm, examples, checkpoints):
    import json
    from pathlib import Path
    import torch
    from safetensors.torch import load_file
    from contextspan.moshi.models.lm import _delay_sequence, _undelay_sequence
    from contextspan.moshi.utils.compile import no_compile, no_cuda_graph
    from evals.personaplex_adapters import install_adapters, adapter_parameters
    from evals.recovery_result_scope import private_result_frames
    from evals.moshi_model_download import sha256
    reports = [json.loads((p / 'report.json').read_text()) for p in checkpoints]
    for path, report in zip(checkpoints, reports):
        reload = json.loads((path / 'reload-report.json').read_text())
        if (not report['passed'] or not reload['passed'] or not reload['all_valid_logits_exact']
                or report['checkpoint_sha256'] != sha256(path / 'adapters.safetensors')
                or reload['checkpoint_sha256'] != report['checkpoint_sha256']):
            raise RuntimeError('Teacher diagnostic requires verified adapter artifacts')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        modules = install_adapters(lm, last_layers=4)
    learned = adapter_parameters(lm)
    if any(r['adapter_modules'] != modules for r in reports):
        raise RuntimeError('Teacher diagnostic adapter layouts differ')
    lm.requires_grad_(False).eval()
    prepared = []
    for e in examples:
        phrase = torch.zeros(e['codes'].shape[1], dtype=torch.bool)
        if e['record']['kind'] in ('private', 'clarify'):
            root = Path(e['record']['cache'])
            cache = json.loads((root / 'cache.json').read_text())
            side = json.loads((Path(cache['conversion']) / e['record']['id'] / 'conversation.json').read_text())
            codes = torch.load(root / (e['record']['id'] + '.pt'), map_location='cpu', weights_only=True)['codes']
            for f in private_result_frames(side, codes[0, 0].tolist(), spm.encode):
                phrase[e['record']['real_to_virtual'][f]] = True
        prepared.append((e, phrase))
    comparisons = []
    for path, report in [(None, None)] + list(zip(checkpoints, reports)):
        if path is not None:
            state = load_file(path / 'adapters.safetensors', device='cuda')
            if set(state) != set(learned): raise RuntimeError('Incomplete diagnostic adapter state')
            with torch.no_grad():
                for name, p in learned.items(): p.copy_(state[name])
        groups = {name: dict(count=0, nll_sum=0., top1_correct=0, pad_probability_sum=0., ret_probability_sum=0.)
            for name in ('ordinary_word', 'private_phrase', 'padding', 'eot', 'ret')}
        for e, phrase in prepared:
            codes = e['codes'].cuda()[None]
            initial = lm._get_initial_token()
            delayed = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], dim=2)
            with torch.no_grad(), no_compile(), no_cuda_graph(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, raw = lm.forward_codes(delayed[:, :, :-1])
                logits, valid = _undelay_sequence(lm.delays[:1], raw, float('nan'))
            targets = codes[:, 0]
            valid = valid[:, 0] & e['text_mask'].cuda()[None] & (targets >= 0)
            masks = {'ordinary_word': targets >= 5, 'private_phrase': phrase.cuda()[None],
                'padding': targets == 3, 'eot': targets == 0, 'ret': targets == 4}
            for name, mask in masks.items():
                keep = mask & valid
                if not keep.any(): continue
                selected = logits[:, 0][keep].float(); gold = targets[keep]
                if not torch.isfinite(selected).all(): raise RuntimeError('Nonfinite valid diagnostic logits')
                log_z = selected.logsumexp(-1)
                group = groups[name]
                group['count'] += len(gold)
                group['nll_sum'] += float((log_z-selected.gather(1, gold[:, None])[:, 0]).sum())
                group['top1_correct'] += int((selected.argmax(-1) == gold).sum())
                group['pad_probability_sum'] += float((selected[:, 3]-log_z).exp().sum())
                group['ret_probability_sum'] += float((selected[:, 4]-log_z).exp().sum())
        for group in groups.values():
            n = group['count']
            for dest, source in [('mean_nll','nll_sum'),('top1_accuracy','top1_correct'),
                    ('mean_pad_probability','pad_probability_sum'),('mean_ret_probability','ret_probability_sum')]:
                group[dest] = group[source] / n if n else None
        comparisons.append({'checkpoint': str(path) if path else 'zero wrapper',
            'optimizer_steps': report['optimizer_steps'] if report else 0, 'groups': groups})
    return {'schema': 'dl-context-teacher-calibration-v1', 'train_examples': len(examples),
        'heldout_labels_used': False, 'teacher_histories': True, 'free_running_qualified': False,
        'dtype': 'BF16 autocast with FP32 adapter storage; eager temporal forward, no CUDA graph',
        'comparisons': comparisons}
