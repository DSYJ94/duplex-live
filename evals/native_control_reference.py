"""The existing FP64-linear numerical reference without retaining wide weights.

Reference arithmetic only: original FP32 weights/activations, FP64 linear work,
FP32 outputs. No gradients, optimizer, compilation or production execution change.
"""
from contextlib import contextmanager


@contextmanager
def linear_reference():
    import torch
    from torch.nn import functional as F
    if torch.is_grad_enabled():
        raise ValueError('Independent numerical reference requires no_grad')
    original=F.linear
    if not original.__module__.startswith('torch'):
        raise ValueError('Do not nest numerical reference implementations')
    stats=dict(calls=0,cached_fp64_weight_bytes=0,maximum_single_fp64_weight_bytes=0,
               arithmetic='fp32-weights-activations-fp64-linear-fp32-output',training_approved=False)
    def linear(x,weight,bias=None):
        if torch.is_grad_enabled() or x.dtype!=torch.float32 or weight.dtype!=torch.float32:
            raise ValueError('FP32 no-gradient mathematical reference only')
        if bias is not None and bias.dtype!=torch.float32:
            raise ValueError('Reference bias must retain FP32 storage')
        stats['calls']+=1
        stats['maximum_single_fp64_weight_bytes']=max(stats['maximum_single_fp64_weight_bytes'],weight.numel()*8)
        # Same casts, same linear call and same output cast as the original
        # reference. The temporary wide matrix is released after this call.
        return original(x.double(),weight.double(),None if bias is None else bias.double()).to(x.dtype)
    F.linear=linear
    try:
        yield stats
    finally:
        F.linear=original
