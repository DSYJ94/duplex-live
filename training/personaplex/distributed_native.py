"""Distributed execution seam for unchanged native objectives.

No scheduler submission, new data approval, optimizer update or adoption occurs
here. CUDA full-shard/state restoration admission is still required on hardware.
"""
from contextlib import nullcontext
from dataclasses import is_dataclass,replace
from functools import partial
import torch
from torch import nn

SHARDED_METHODS = ('selective-trunk-text', 'full-lm')


class NativeDispatch(nn.Module):
    """Both custom native forwards must enter DDP/FSDP's public forward hook."""
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, kind, codes):
        if kind == 'text_rehearsal':
            if (codes.ndim != 3 or codes.dtype != torch.long or codes.shape[1] != self.lm.n_q + 1
                    or codes.shape[-1] < 1 or bool((codes[:, 1:] != -1).any())
                    or bool((codes[:, 0] < 0).any())):
                raise ValueError('Text rehearsal requires absent audio at every position, including BOS')
            if any(getattr(module, '_streaming_state', None) is not None for module in self.lm.modules()):
                raise ValueError('Independent text documents cannot inherit a native streaming cache')
            return self.lm.forward_codes(codes)
        if kind == 'text':
            return self.lm.forward_codes(codes)
        if kind == 'audio':
            # Existing audio_forward owns native_depth_context outside this call.
            output=self.lm.forward_train(codes)
            # Native LMOutput is a dataclass. DDP discovers every tensor there,
            # but its output sink does not propagate unused dataclass branches.
            # This route supervises audio only: preserve text values as metadata,
            # and exclude its unsupervised head from reducer graph discovery.
            if is_dataclass(output) and hasattr(output,'text_logits'):
                output=replace(output,text_logits=output.text_logits.detach())
            return output
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

    def forward_text_rehearsal(self, codes):
        return self._wrapped('text_rehearsal', codes)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._lm, name)


def configure_lm(lm, method):
    """Fresh native LM only; Mimi is never accepted as a child to be unfrozen."""
    from evals.personaplex_adapters import AdapterLinear
    if method not in ('broad-lora64',) + SHARDED_METHODS:
        raise ValueError('Explicit broad adapter, selective trunk/text or full native LM method required')
    if not hasattr(lm, 'forward_codes') or not hasattr(lm, 'forward_train'):
        raise ValueError('Native text and acoustic LM required')
    if any(isinstance(m, AdapterLinear) for m in lm.modules()):
        raise ValueError('Fresh baseline required; never stack a saved adapter')
    if any(name.lower() in {'mimi', 'codec'} for name, _ in lm.named_children()):
        raise ValueError('Pass only the native LM; codec remains frozen and separate')
    if method == 'broad-lora64':
        from .capacity import configure
        devices=sorted({p.device.index for p in lm.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(17)
            for index in devices:
                with torch.cuda.device(index):torch.cuda.manual_seed(17)
            details = configure(lm, method)
            from .native_adapter_compile import compile_gates
            details["native_compiled_adapter_gates"] = compile_gates(lm)
    elif method == 'selective-trunk-text':
        if (not isinstance(getattr(lm, 'transformer', None), nn.Module)
                or not isinstance(getattr(lm, 'text_linear', None), nn.Linear)
                or not isinstance(getattr(lm, 'depformer', None), nn.Module)):
            raise ValueError('Unwrapped native temporal transformer/text head/acoustic depth required')
        # Uniform master dtype avoids mixed-dtype FSDP flattening. Frozen
        # acoustic branches still propagate gradients into the shared trunk.
        lm.float().requires_grad_(False)
        lm.transformer.requires_grad_(True)
        lm.text_linear.requires_grad_(True)
        for name, child in lm.named_children():
            if name not in ('transformer', 'text_linear') and any(p.requires_grad for p in child.parameters()):
                raise ValueError('Tied weights cannot silently unfreeze an acoustic/input branch')
        if any(p.requires_grad and not n.startswith(('transformer.', 'text_linear.')) for n, p in lm.named_parameters()):
            raise ValueError('Only declared shared temporal trunk and text head may train')
        details = dict(variant=method, new_model_weights=0,
                       selected_prefixes=['transformer.', 'text_linear.'],
                       acoustic_depth_and_embeddings_frozen=True,
                       shared_trunk_can_still_change_speech=True,
                       native_GPU_gradient_restore_admission_required=True)
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
    def __init__(self, lm, method, device, process_group=None, after_configure=None):
        import torch.distributed as dist
        if not dist.is_initialized():
            raise ValueError('Initialize the declared distributed process group first')
        self.method, self.device = method, torch.device(device)
        self.ownership = configure_lm(lm, method)
        if after_configure is not None:after_configure(lm,self.ownership)
        dispatch = NativeDispatch(lm)
        self.sharded = method in SHARDED_METHODS
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
        from .native_optimizer_state import load_exact
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
            self.last_restore_proof=load_exact(optimizer,state['optimizer'])
            return self.last_restore_proof
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        with FSDP.state_dict_type(self.wrapped, StateDictType.FULL_STATE_DICT,
                                  FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
            self.wrapped.load_state_dict(state['model'], strict=True)
        # Full named checkpoint -> exact rank-owned optimizer representation.
        # Comparing the full tree to this local shard would be invalid.
        local=FSDP.optim_state_dict_to_load(self.wrapped, optimizer, state['optimizer'])
        self.last_restore_proof=load_exact(optimizer,local)
        return self.last_restore_proof


def cpu_state(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_state(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_state(v) for v in value)
    return value
