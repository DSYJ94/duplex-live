"""Isolate cuDNN attention planning cost, with actual decode tensor checks.

The profiler found repeated single-query cuDNN attention setup and kernel launch
overhead. Disable only cuDNN SDPA for this experiment. Compare sampled actual
BF16 decode calls with cuDNN and float32 math, without changing weights/caches.
Backend numerical differences and generated behavior still need evaluation.
"""
from contextlib import contextmanager


@contextmanager
def without_cudnn_attention():
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    original = F.scaled_dot_product_attention
    enabled = torch.backends.cuda.cudnn_sdp_enabled()
    report = {"schema":"dl-lychee-attention-kernels-v1", "cudnn_disabled":True,
        "model_or_cache_modified":False, "decode_calls":0, "probes":[], "behavior_qualified":False}

    def attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False, *, scale=None, enable_gqa=False):
        options = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        output = original(q, k, v, **options)
        if q.ndim == 4 and q.shape[1:] == (28, 1, 128) and attn_mask is None:
            report["decode_calls"] += 1
            if len(report["probes"]) < 8 and report["decode_calls"] % 100 == 1:
                if dropout_p:
                    raise RuntimeError("Unexpected stochastic attention dropout during inference")
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    baseline = original(q, k, v, **options)
                with sdpa_kernel(SDPBackend.MATH):
                    precise = original(q.float(), k.float(), v.float(), **options)
                report["probes"].append({"query":list(q.shape), "key":list(k.shape), "dtype":str(q.dtype),
                    "candidate_vs_cudnn_max":float((output.float()-baseline.float()).abs().max()),
                    "candidate_vs_float32_max":float((output.float()-precise).abs().max()),
                    "cudnn_vs_float32_max":float((baseline.float()-precise).abs().max()),
                    "candidate_vs_float32_rms":float((output.float()-precise).square().mean().sqrt()),
                    "cudnn_vs_float32_rms":float((baseline.float()-precise).square().mean().sqrt())})
        return output

    torch.backends.cuda.enable_cudnn_sdp(False)
    F.scaled_dot_product_attention = attention
    try:
        yield report
    finally:
        F.scaled_dot_product_attention = original
        torch.backends.cuda.enable_cudnn_sdp(enabled)
