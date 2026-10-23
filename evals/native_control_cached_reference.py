"""Bounded reuse of exact FP64 casts inside an independent numerical reference.

No production/training use. This changes allocation and casting frequency only;
each linear operation uses the same exact FP64 weights, input and output casts.
"""
from contextlib import contextmanager
import weakref


@contextmanager
def linear_reference(maximum_cached_bytes=20*1024**3):
    import torch
    from torch.nn import functional as F
    if torch.is_grad_enabled():raise ValueError('Independent cached reference requires no_grad')
    if type(maximum_cached_bytes) is not int or not 0<=maximum_cached_bytes<=24*1024**3:
        raise ValueError('Explicit bounded reference cache of at most24GiB required')
    original=F.linear
    if not original.__module__.startswith('torch'):raise ValueError('Do not nest numerical references')
    cache={}
    stats=dict(calls=0,cast_cache_hits=0,cast_cache_misses=0,cached_fp64_weight_bytes=0,
               maximum_cached_bytes=maximum_cached_bytes,peak_cached_bytes=0,
               arithmetic='fp32-weights-activations-fp64-linear-fp32-output',
               allocation_only=True,training_approved=False)
    def identity(t):return (t.data_ptr(),t._version,tuple(t.shape),tuple(t.stride()),t.dtype,t.device)
    def cast(t):
        if t is None:return None
        key=id(t);old=cache.get(key)
        if old is not None:
            ref,expected,value=old
            if ref() is t:
                if identity(t)!=expected:raise ValueError('Reference weight changed while cast was cached')
                stats['cast_cache_hits']+=1;return value
            stats['cached_fp64_weight_bytes']-=value.numel()*value.element_size();del cache[key]
        stats['cast_cache_misses']+=1
        value=t.double();size=value.numel()*value.element_size()
        if stats['cached_fp64_weight_bytes']+size<=maximum_cached_bytes:
            cache[key]=(weakref.ref(t),identity(t),value)
            stats['cached_fp64_weight_bytes']+=size
            stats['peak_cached_bytes']=max(stats['peak_cached_bytes'],stats['cached_fp64_weight_bytes'])
        return value
    def linear(x,weight,bias=None):
        if (torch.is_grad_enabled() or x.dtype!=torch.float32 or weight.dtype!=torch.float32
                or (bias is not None and bias.dtype!=torch.float32)):
            raise ValueError('FP32 no-gradient mathematical reference only')
        stats['calls']+=1
        return original(x.double(),cast(weight),cast(bias)).to(x.dtype)
    F.linear=linear
    try:yield stats
    finally:
        F.linear=original
        try:
            for ref,expected,_ in cache.values():
                if ref() is not None and identity(ref())!=expected:
                    raise ValueError('Cached reference weight changed before final verification')
        finally:cache.clear()
