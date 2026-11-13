"""Two-process CPU fixture; never loads native weights or approves GPU training."""
import copy
import json
import os
import torch
import torch.distributed as dist
from evals.harness.test_distributed_native_build import ToyNative
from training.personaplex.distributed_native import DistributedNative


def main():
    dist.init_process_group('gloo')
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 2: raise ValueError('This fixture compares exactly two CPU ranks')
        torch.manual_seed(1717)
        runtime = DistributedNative(ToyNative(), 'broad-lora64', 'cpu')
        gold = copy.deepcopy(runtime.wrapped.module.lm)
        params = [p for p in gold.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(runtime.parameters(), lr=1e-3)
        reference = torch.optim.AdamW(params, lr=1e-3)
        for step in range(2):
            opt.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
            x = torch.full((1, 2, 4), float(rank + 1 + step))
            with runtime.accumulation(False):
                runtime.view.forward_codes(x)[1].square().mean().backward()
            runtime.view.forward_train(x)[0].square().mean().backward()
            for other in range(world):
                sample = torch.full((1, 2, 4), float(other + 1 + step))
                (gold.forward_codes(sample)[1].square().mean() / world).backward()
                (gold.forward_train(sample)[0].square().mean() / world).backward()
            runtime.clip_grad_norm(1.)
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step(); reference.step()
            for actual, expected in zip(runtime.wrapped.module.lm.parameters(), gold.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        state = runtime.full_state(opt)
        assert all('.lora_' in n for n in state['model'])
        vector = torch.cat([p.detach().flatten() for p in runtime.parameters()])
        all_ranks = [torch.empty_like(vector) for _ in range(world)]
        dist.all_gather(all_ranks, vector)
        torch.testing.assert_close(all_ranks[0], all_ranks[1], rtol=0, atol=0)
        if rank == 0:
            print(json.dumps(dict(passed=True, actual_world_size=world, backend='gloo', device='cpu',
                fixture_updates=2, native_optimizer_updates=0, native_weights_loaded=False,
                compared_to_single_process_global_mean=True, rank_parameters_bit_identical=True,
                full_CUDA_FSDP_admission=False)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
