"""Diagnostic FP64 attention accumulation; never production or training policy."""
from contextlib import contextmanager


@contextmanager
def attention_reference():
    import torch
    from torch.nn import functional as F
    if torch.is_grad_enabled():raise ValueError('Attention reference requires no gradients')
    original=F.scaled_dot_product_attention
    if not original.__module__.startswith('torch'):raise ValueError('Do not nest attention references')
    stats=dict(calls=0,accumulation_dtype='float64',input_output_dtype='float32',
               dropout=0.,production_or_training_execution_changed=False)
    def attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
        if torch.is_grad_enabled() or any(x.dtype!=torch.float32 for x in (q,k,v)) or dropout_p!=0.:
            raise ValueError('Independent attention reference is FP32 inputs, zero dropout and no gradients only')
        if attn_mask is not None and attn_mask.dtype not in (torch.bool,torch.float32,torch.float64):
            raise ValueError('Unexpected reference attention mask')
        mask=attn_mask if attn_mask is None or attn_mask.dtype==torch.bool else attn_mask.double()
        stats['calls']+=1
        return original(q.double(),k.double(),v.double(),mask,dropout_p,is_causal,
                        scale=scale,enable_gqa=enable_gqa).float()
    F.scaled_dot_product_attention=attention
    try:yield stats
    finally:F.scaled_dot_product_attention=original
