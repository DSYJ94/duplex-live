"""Explicit native execution provenance for actual-history restart diagnostics."""

def snapshot(lm):
    import os
    import torch
    from collections import Counter
    return {
        'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(),
        'capability': list(torch.cuda.get_device_capability()),
        'cudnn': torch.backends.cudnn.enabled,
        'cudnn_sdp': torch.backends.cuda.cudnn_sdp_enabled(),
        'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
        'mem_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math_sdp': torch.backends.cuda.math_sdp_enabled(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'bf16_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'float32_precision': torch.get_float32_matmul_precision(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'threads': torch.get_num_threads(),
        'parameter_dtypes': dict(Counter(str(p.dtype) for p in lm.parameters())),
        'environment': {k: os.environ.get(k) for k in (
            'NO_CUDA_GRAPH','NO_TORCH_COMPILE','TORCH_CUDNN_SDPA_ENABLED','CUBLAS_WORKSPACE_CONFIG')},
    }
