import copy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from training.personaplex.distributed_native import NativeDispatch, NativeView, configure_lm
from training.personaplex.text_rehearsal import TextSpec, averaged_gradient_loss, backward_window, loss, pack, validate


def spec():
    return TextSpec(2, 32, 0, 3, 8, frozenset((0, 1, 2, 3, 4)))


class TextNative(nn.Module):
    """CPU test double, not a native checkpoint or admission proof."""
    n_q = 2
    text_initial_token_id = 0

    def __init__(self):
        super().__init__()
        self.text_emb = nn.Embedding(32, 8)
        self.emb = nn.ModuleList([nn.Embedding(16, 8) for _ in range(2)])
        self.transformer = nn.Linear(8, 8)
        self.depformer = nn.Linear(8, 8)
        self.text_linear = nn.Linear(8, 32)

    def forward_codes(self, codes):
        embedded = self.text_emb(codes[:, 0])
        for i, embedding in enumerate(self.emb):
            audio = embedding(codes[:, i + 1].clamp_min(0))
            embedded = embedded + torch.where((codes[:, i + 1] == -1).unsqueeze(-1), torch.zeros_like(audio), audio)
        hidden = self.transformer(embedded.cumsum(1))
        return hidden, self.text_linear(hidden).unsqueeze(1)

    def forward_train(self, codes):
        raise AssertionError('Audio/depth training must never run for text rehearsal')


def window_batches(rank):
    first = pack([5, 6], spec()) if rank == 0 else pack([7, 8, 9, 10], spec(), supervise=[False, True, True, True])
    second = pack([11], spec()) if rank == 0 else pack([12, 13], spec())
    if rank == 1:
        second['mask'].zero_()  # Valid rank padding; the global microbatch is nonempty.
    return [first, second]


def distributed_worker(rank, rendezvous, output, accumulated=False):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(101)
        model = TextNative()
        configure_lm(model, 'selective-trunk-text')
        wrapper = DistributedDataParallel(NativeDispatch(model), find_unused_parameters=True)
        view = NativeView(wrapper, model)
        if accumulated:
            metrics = backward_window(view, window_batches(rank), spec(), coefficient=.3)
        else:
            batch = window_batches(rank)[0]
            value, metrics = loss(view, batch, spec())
            value.backward()
        torch.save(dict(gradients={n: p.grad for n, p in model.named_parameters() if p.requires_grad},
                        metrics=metrics), Path(output) / ('rank-%d.pt' % rank))
    finally:
        dist.destroy_process_group()


class TextRehearsalTest(unittest.TestCase):
    def test_exact_shift_absent_audio_and_first_answer_boundary(self):
        batch = pack([5, 6, 7, 8], spec(), supervise=[False, False, True, True])
        self.assertEqual(batch['inputs'][0, 0].tolist(), [0, 5, 6, 7])
        self.assertTrue(bool((batch['inputs'][:, 1:] == -1).all()))
        self.assertEqual(batch['targets'][batch['mask']].tolist(), [7, 8])
        self.assertEqual(int(batch['inputs'][0, 0, 2]), 6)
        validate(batch, spec())
        damaged = copy.deepcopy(batch)
        damaged['inputs'][0, 0, 2] = 7
        with self.assertRaises(ValueError): validate(damaged, spec())

    def test_no_reserved_targets_truncation_or_empty_supervision(self):
        for ids in ([4, 5], [-1, 5], [32], [5] * 9, []):
            with self.assertRaises(ValueError): pack(ids, spec())
        with self.assertRaises(ValueError): pack([5, 6], spec(), supervise=[False, False])
        with self.assertRaises(ValueError): pack([5], spec(), supervise=[1])

    def test_public_dispatch_gradients_frozen_branches_and_future_causality(self):
        torch.manual_seed(10)
        model = TextNative()
        before = copy.deepcopy(model.state_dict())
        configure_lm(model, 'selective-trunk-text')
        dispatch = NativeDispatch(model)
        routes = []
        handle = dispatch.register_forward_pre_hook(lambda module, args: routes.append(args[0]))
        view = NativeView(dispatch, model)
        batch = pack([5, 6, 7, 8], spec())
        _, original = view.forward_text_rehearsal(batch['inputs'])
        changed = pack([5, 6, 11, 8], spec())
        _, future = view.forward_text_rehearsal(changed['inputs'])
        torch.testing.assert_close(original[:, :, :3], future[:, :, :3], rtol=0, atol=0)
        value, metrics = loss(view, batch, spec())
        value.backward()
        self.assertEqual(metrics['global_valid_text_tokens'], 4)
        self.assertFalse(metrics['acoustic_loss_computed'])
        self.assertGreater(model.transformer.weight.grad.abs().sum(), 0)
        self.assertGreater(model.text_linear.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.depformer.parameters()))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.step()
        for name, value in model.state_dict().items():
            if not name.startswith(('transformer.', 'text_linear.')):
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(route == 'text_rehearsal' for route in routes))
        handle.remove()

    def test_nonabsent_audio_bos_or_inherited_cache_is_rejected(self):
        model = TextNative()
        dispatch = NativeDispatch(model)
        batch = pack([5, 6], spec())
        batch['inputs'][0, 1, 0] = 0
        with self.assertRaises(ValueError): dispatch('text_rehearsal', batch['inputs'])
        model._streaming_state = object()
        with self.assertRaises(ValueError): dispatch('text_rehearsal', pack([5], spec())['inputs'])

    def test_global_weighting_is_pooled_token_mean_not_mean_of_rank_means(self):
        left = averaged_gradient_loss(torch.tensor(3.), 5, 2)
        right = averaged_gradient_loss(torch.tensor(12.), 5, 2)
        self.assertEqual(float((left + right) / 2), 3.)

    def test_accumulation_matches_pooled_tokens_and_preserves_existing_gradients(self):
        torch.manual_seed(203)
        model = TextNative()
        configure_lm(model, 'selective-trunk-text')
        reference = copy.deepcopy(model)
        batches = [pack([5], spec()), pack([6, 7, 8, 9], spec(), supervise=[False, True, True, True])]
        for current in (model, reference):
            for parameter in current.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, .125)
        view = NativeView(NativeDispatch(model), model)
        report = backward_window(view, batches, spec(), coefficient=.3)
        summed = 0.
        for batch in batches:
            _, logits = reference.forward_codes(batch['inputs'])
            summed = summed + F.cross_entropy(logits[:, 0][batch['mask']], batch['targets'][batch['mask']], reduction='sum')
        (.3 * summed / 4).backward()
        self.assertEqual(report['global_window_valid_text_tokens'], 4)
        self.assertFalse(report['optimizer_step_performed'])
        for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
            if actual.requires_grad:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
            else:
                self.assertIsNone(actual.grad)
        for invalid in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                backward_window(view, batches, spec(), coefficient=invalid)

    def test_real_two_rank_ddp_gradients_match_pooled_reference(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mp.spawn(distributed_worker, args=(str(root / 'rendezvous'), temp), nprocs=2, join=True)
            torch.manual_seed(101)
            model = TextNative()
            configure_lm(model, 'selective-trunk-text')
            total = 0.
            for batch in (pack([5, 6], spec()), pack([7, 8, 9, 10], spec(), supervise=[False, True, True, True])):
                _, logits = model.forward_codes(batch['inputs'])
                total = total + F.cross_entropy(logits[:, 0][batch['mask']], batch['targets'][batch['mask']], reduction='sum')
            (total / 5).backward()
            for rank in range(2):
                state = torch.load(root / ('rank-%d.pt' % rank), weights_only=True)
                self.assertEqual(state['metrics']['global_valid_text_tokens'], 5)
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(state['gradients'][name], parameter.grad, rtol=1e-5, atol=1e-7)

    def test_real_two_rank_accumulation_matches_pooled_reference_with_empty_local_mask(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mp.spawn(distributed_worker, args=(str(root / 'rendezvous'), temp, True), nprocs=2, join=True)
            torch.manual_seed(101)
            model = TextNative()
            configure_lm(model, 'selective-trunk-text')
            total = 0.
            for rank in range(2):
                for batch in window_batches(rank):
                    _, logits = model.forward_codes(batch['inputs'])
                    if batch['mask'].any():
                        total = total + F.cross_entropy(logits[:, 0][batch['mask']], batch['targets'][batch['mask']], reduction='sum')
            (.3 * total / 6).backward()
            for rank in range(2):
                state = torch.load(root / ('rank-%d.pt' % rank), weights_only=True)
                self.assertEqual(state['metrics']['global_window_valid_text_tokens'], 6)
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(state['gradients'][name], parameter.grad, rtol=1e-5, atol=1e-7)
