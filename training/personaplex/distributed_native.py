"""Distributed execution seam for unchanged native objectives.

No scheduler submission, new data approval, optimizer update or adoption occurs
here. CUDA full-shard/state restoration admission is still required on hardware.
"""
from contextlib import nullcontext
from functools import partial
import torch
from torch import nn


class NativeDispatch(nn.Module):
    """Both custom native forwards must enter DDP/FSDP's public forward hook."""
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, kind, codes):
        if kind == 'text':
            return self.lm.forward_codes(codes)
        if kind == 'audio':
            # Existing audio_forward owns native_depth_context outside this call.
            return self.lm.forward_train(codes)
        raise ValueError('Unknown native forward kind')


class NativeView:
    """Legacy Objective interface, routing computation through the wrapper.

    Streaming state and layer metadata remain with the original LM. Accessing
    those does not run an unwrapped parameter-dependent forward.
    """
    def __init__(self, wrapped, lm):
        self._wrapped, self._lm = wrapped, lm

    def forward_codes(self, codes):
        return self._wrapped('text', codes)

    def forward_train(self, codes):
        return self._wrapped('audio', codes)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._lm, name)


def configure_lm(lm, method):
    """Fresh native LM only; Mimi is never accepted as a child to be unfrozen."""
    from evals.personaplex_adapters import AdapterLinear
    if method not in ('broad-lora64', 'full-lm'):
        raise ValueError('Explicit broad adapter or full native LM method required')
    if not hasattr(lm, 'forward_codes') or not hasattr(lm, 'forward_train'):
        raise ValueError('Native text and acoustic LM required')
    if any(isinstance(m, AdapterLinear) for m in lm.modules()):
        raise ValueError('Fresh baseline required; never stack a saved adapter')
    if any(name.lower() in {'mimi', 'codec'} for name, _ in lm.named_children()):
        raise ValueError('Pass only the native LM; codec remains frozen and separate')
    if method == 'broad-lora64':
        from .capacity import configure
        details = configure(lm, method)
    else:
        # FP32 optimizer/master parameters; FSDP casts computation to BF16.
        lm.float().requires_grad_(True)
        details = dict(variant=method, new_model_weights=0)
    selected = {n: p for n, p in lm.named_parameters() if p.requires_grad}
    if not selected or any(p.dtype != torch.float32 for p in selected.values()):
        raise ValueError('Exact FP32 trainable master ownership required')
    return dict(method=method, trainable_parameters=sum(p.numel() for p in selected.values()),
                total_parameters=sum(p.numel() for p in lm.parameters()),
                trainable_names=sorted(selected), details=details, codec_included=False)


class DistributedNative:
    def __init__(self, lm, method, device, process_group=None):
        import torch.distributed as dist
        if not dist.is_initialized():
            raise ValueError('Initialize the declared distributed process group first')
        self.method, self.device = method, torch.device(device)
        self.ownership = configure_lm(lm, method)
        dispatch = NativeDispatch(lm)
        self.sharded = method == 'full-lm'
        if self.sharded:
            if self.device.type != 'cuda':
                raise ValueError('Full native sharding requires actual CUDA admission')
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            layer_types = {type(lm.transformer.layers[0]), type(lm.depformer.layers[0])}
            self.wrapped = FSDP(dispatch, process_group=process_group, device_id=self.device,
                sharding_strategy=ShardingStrategy.FULL_SHARD, use_orig_params=True,
                auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_types),
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                               buffer_dtype=torch.bfloat16),
                sync_module_states=True, limit_all_gathers=True)
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
            dispatch.to(self.device)
            ids = [self.device.index] if self.device.type == 'cuda' else None
            # Some losses supervise only temporal/text or assistant audio heads.
            self.wrapped = DDP(dispatch, device_ids=ids, process_group=process_group,
                               broadcast_buffers=False, find_unused_parameters=True)
        self.view = NativeView(self.wrapped, lm)

    def parameters(self):
        # Construct the optimizer AFTER wrapping, including FSDP's owned views.
        return [p for p in self.wrapped.parameters() if p.requires_grad]

    def accumulation(self, synchronize):
        # No CPU offload: accumulation retains the existing native loss weights.
        return nullcontext() if synchronize else self.wrapped.no_sync()

    def clip_grad_norm(self, maximum):
        if maximum <= 0:
            raise ValueError('Positive clipping norm required')
        norm = (self.wrapped.clip_grad_norm_(maximum) if self.sharded else
                torch.nn.utils.clip_grad_norm_(self.parameters(), maximum, error_if_nonfinite=True))
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite distributed gradient norm')
        return norm

    def full_state(self, optimizer):
        """Collective on ALL ranks. Only rank0 writes returned full CPU state."""
        if not self.sharded:
            learned = {n: p.detach().cpu().clone() for n, p in self.wrapped.module.named_parameters() if p.requires_grad}
            return dict(method=self.method, ownership=self.ownership, model=learned,
                        optimizer=cpu_state(optimizer.state_dict()))
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig, FullOptimStateDictConfig
        with FSDP.state_dict_type(self.wrapped, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                  FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            return dict(method=self.method, ownership=self.ownership, model=self.wrapped.state_dict(),
                        optimizer=FSDP.optim_state_dict(self.wrapped, optimizer))

    def load_full_state(self, state, optimizer):
        """ALL ranks receive the same checkpoint; never use adapter-only restore."""
        if state.get('method') != self.method or state.get('ownership') != self.ownership:
            raise ValueError('Exact distributed method and parameter ownership required')
        if not self.sharded:
            selected = {n: p for n, p in self.wrapped.module.named_parameters() if p.requires_grad}
            if selected.keys() != state['model'].keys():
                raise ValueError('Exact complete adapter state required')
            for name, p in selected.items():
                value = state['model'][name]
                if value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
                    raise ValueError('Invalid learned state: ' + name)
            with torch.no_grad():
                for name, p in selected.items(): p.copy_(state['model'][name].to(p.device))
            optimizer.load_state_dict(state['optimizer'])
            return
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(self.wrapped, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            self.wrapped.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(FSDP.optim_state_dict_to_load(self.wrapped, optimizer, state['optimizer']))


def cpu_state(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_state(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_state(v) for v in value)
    return value
