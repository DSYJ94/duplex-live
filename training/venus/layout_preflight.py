"""Tiny CUDA SDPA layout compatibility check before loading any model weights.

This proves only compatibility of saved-tensor storage with the actual backward
kernel. It does not replace native input, full-gradient, state or behavior gates.
"""
import torch
from .offload import ActivationOffload

POLICY='repeat4-prefix-pageable-v3-layout'


def check():
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA preflight required')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    rows=[]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(20261104)
        for layout in ('padded','broadcast','native-contiguous'):
            broadcast=layout=='broadcast'
            source=[torch.randn(1,32,n,128,device='cuda',dtype=torch.bfloat16) for n in (10,1535,1535)]
            mask=torch.zeros((1,1 if broadcast else 32,10,1536),device='cuda',dtype=torch.bfloat16)[...,:1535]
            if broadcast:mask=mask.expand(1,32,10,1535)
            if layout=='native-contiguous':mask=torch.zeros((1,1,10,1535),device='cuda',dtype=torch.bfloat16)
            def run(hook=None):
                from contextlib import nullcontext
                values=[x.detach().clone().requires_grad_(True) for x in source]
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    with hook.context() if hook else nullcontext():
                        output=torch.nn.functional.scaled_dot_product_attention(*values,attn_mask=mask)
                        loss=output.float().square().mean();loss.backward()
                return output.detach(),[x.grad.detach().clone() for x in values]
            baseline,grads=run()
            hook=ActivationOffload(torch.nn.Module(),minimum_bytes=1,policy=POLICY)
            actual,restored=run(hook)
            if not torch.equal(baseline,actual) or not all(torch.equal(a,b) and torch.isfinite(b).all() for a,b in zip(grads,restored)):
                raise ValueError('Actual CUDA attention output/gradients differ after layout restoration')
            rows.append(dict(layout=layout,broadcast=broadcast,mask_shape=list(mask.shape),mask_stride=list(mask.stride()),
                             exact_output=True,exact_all_QKV_gradients=True,saved_tensors=hook.tensors,
                             physical_saved_bytes=hook.saved_bytes))
    return dict(passed=True,actual_CUDA=True,torch_version=torch.__version__,policy=POLICY,
                backend='EFFICIENT_ATTENTION',cases=rows,new_optimizer_updates=0,loaded_model_weights=False,
                replaces_full_native_admission=False)
