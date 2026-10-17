"""Higher precision linear arithmetic for no-gradient mathematical checks only."""
from contextlib import contextmanager


@contextmanager
def linear_reference():
    import torch
    from torch.nn import functional as F
    if torch.is_grad_enabled():raise RuntimeError('Numerical reference cannot run with gradients')
    original=F.linear;wide={}
    def linear(x,weight,bias=None):
        key=id(weight)
        if key not in wide:wide[key]=(weight,weight.double())
        if wide[key][0] is not weight:raise RuntimeError('Numerical reference weight identity changed')
        return original(x.double(),wide[key][1],None if bias is None else bias.double()).to(x.dtype)
    F.linear=linear
    try:yield
    finally:
        F.linear=original;wide.clear()
