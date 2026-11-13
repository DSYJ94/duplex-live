"""Single-owner sharded gradient clipping around the preserved OPAL runtime."""
from .distributed_native import cpu_state
from training.venus.native_optimizer import NativeOpal
from training.venus.opal_adapter import checkpoint_manifest


class DistributedOpal:
    def __init__(self, runtime, base, *, source, run_dir, recipe):
        clip = recipe['gradient_clip']
        if not isinstance(clip, (int, float)) or clip <= 0:
            raise ValueError('Positive declared distributed clipping norm required')
        self.runtime, self.recipe = runtime, dict(recipe)
        # OPAL owns updates/scheduling/program state; collective runtime alone
        # owns clipping. Never clip local FSDP shards a second time upstream.
        unclipped = dict(recipe, gradient_clip=None)
        self.active = NativeOpal(base, runtime.wrapped, source=source,
            run_dir=run_dir, recipe=unclipped, backend='keep', agent_budget=0.)

    def step(self, loss, step):
        import torch.distributed as dist
        local = checkpoint_manifest(self.active.optimizer)['update']
        ranks = [None] * dist.get_world_size()
        dist.all_gather_object(ranks, local)
        if any(value != step for value in ranks):
            raise ValueError('All OPAL ranks must agree before clipping or stepping')
        norm = self.runtime.clip_grad_norm(self.recipe['gradient_clip'])
        result = self.active.step(loss, step)
        return dict(program=result, global_preclip_norm=float(norm))

    def rank_state(self, rng):
        # Full upstream state is retained, including optimizer registry. Its
        # potentially large tensors must be saved PER RANK, not all-gathered.
        return dict(rng=rng, opal=cpu_state(self.active.optimizer.state_dict()),
                    diagnostics=self.active.diagnostics_state, recipe=self.recipe)

    def close(self):
        self.active.close()
