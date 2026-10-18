"""Bounded fullgraph decode diagnostic with unchanged existing dynamic cache.

Prefill stays eager. Compilation failures are surfaced, never silently replaced
with eager execution. This module is used only by an explicit offline probe.
"""
import functools
from pathlib import Path
from evals.moshi_model_download import sha256


def install(model):
    import torch
    from torch._dynamo.utils import counters
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Decode diagnostic requires frozen weights')
    original=model.forward
    compiled=torch.compile(original,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})
    report=dict(schema='dl-whole-decode-dynamic-cache-compile-v1',decode_calls=0,prefill_calls=0,
        whole_model_compiled=True,cache_changed=False,cuda_graphs=False,
        unique_graphs_before=counters['stats']['unique_graphs'],graph_breaks_before=sum(counters['graph_break'].values()),
        code_sha256=sha256(Path(__file__)))
    @functools.wraps(original)
    def forward(*args,**kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is not None and ids.ndim==2 and ids.shape==(1,1) and kwargs.get('past_key_values') is not None:
            report['decode_calls']+=1
            return compiled(*args,**kwargs)
        report['prefill_calls']+=1
        return original(*args,**kwargs)
    model.forward=forward
    return report
