"""Switch only checked post-branch weights at an aborted-request boundary.

The base and merged tensors share the original parameter objects. Eager native
execution reads the selected storage. No active KV may survive a switch.
"""
import time


class NativeSemanticSwitch:
    def __init__(self, reloads, record):
        self.reloads=reloads;self.record=record;self.initialized=False;self.active=True

    def initialize(self, model):
        import torch
        params=dict(model.named_parameters());targets={name:p for name,p,*_ in self.reloads}
        if not targets or any(params.get(name) is not p for name,p in targets.items()):
            raise RuntimeError('Native post-load processing replaced switchable parameters')
        cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
        self.targets=targets
        # Retain adapted storage, then replay original checkpoint shards into
        # new storage using the actual original native packed-weight loaders.
        self.adapted={name:p.detach() for name,p in targets.items()}
        with torch.no_grad():
            for p in targets.values():p.data=torch.empty_like(p)
            for name,p,loader,value,args,kwargs in self.reloads:loader(p,value,*args,**kwargs)
        self.base={name:p.detach() for name,p in targets.items()}
        torch.cuda.synchronize()
        if not torch.equal(cpu,torch.get_rng_state()) or not torch.equal(gpu,torch.cuda.get_rng_state()):
            raise RuntimeError('Semantic switch preparation consumed sampling RNG')
        self.reloads.clear();self.active=False;self.initialized=True
        self.record.update(switchable=True,initially_active=False,
            switchable_native_parameters=len(targets),
            switchable_scalar_weights=sum(p.numel() for p in targets.values()),
            switch_setup_rng_preserved=True,
            switch_method='same parameter objects; separate original and merged BF16 storage; eager only')

    def set(self, active):
        import torch
        if not self.initialized:raise RuntimeError('Native semantic switch is not initialized')
        before=time.monotonic();cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
        buffers=self.adapted if active else self.base
        with torch.no_grad():
            for name,p in self.targets.items():p.data=buffers[name]
        torch.cuda.synchronize()
        if (not torch.equal(cpu,torch.get_rng_state()) or not torch.equal(gpu,torch.cuda.get_rng_state())
                or any(p.data_ptr()!=buffers[name].data_ptr() for name,p in self.targets.items())):
            raise RuntimeError('Semantic switch changed RNG or selected incorrect storage')
        self.active=bool(active)
        return {'active':self.active,'switch_seconds':time.monotonic()-before,
            'sampling_rng_preserved':True,'native_parameters':len(self.targets)}


def initialize(framework):
    model=framework.vllm_engine.engine.model_executor.driver_worker.model_runner.model
    switch=model._dl_semantic_switch
    if framework.vllm_engine._active_request_id is not None:
        raise RuntimeError('Switch preparation must precede user input')
    switch.initialize(model)
    return switch


def set_active(framework, active):
    model=framework.vllm_engine.engine.model_executor.driver_worker.model_runner.model
    framework.model.close_active_request()
    if framework.vllm_engine._active_request_id is not None:
        raise RuntimeError('Cannot switch semantic weights with an active native request')
    result=model._dl_semantic_switch.set(active)
    result['active_request_aborted_before_switch']=True
    return result
