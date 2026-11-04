"""Tiny CUDA kernel/storage proof before loading model weights.

Exact gradient comparison uses deterministic backward only inside this test;
PyTorch2.4 efficient backward otherwise defaults to split-key nondeterminism.
Native training settings, numerical gates and optimizer state are unchanged.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import torch
from .offload import ActivationOffload

POLICY='repeat4-prefix-pageable-v3-layout'


@contextmanager
def deterministic_comparison():
    enabled=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        yield
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)


def difference(a,b):
    delta=a.detach().float()-b.detach().float()
    maximum=float(delta.abs().max())
    relative=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.float()).clamp_min(1e-30))
    return dict(exact=torch.equal(a,b),all_finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                differing_elements=int(torch.count_nonzero(a!=b)),
                maximum_absolute=maximum if math.isfinite(maximum) else None,
                relative_l2=relative if math.isfinite(relative) else None)


class RoundtripAudit(ActivationOffload):
    """Tiny-test only: retain originals to verify EVERY unpack on actual CUDA."""
    def __init__(self):
        super().__init__(torch.nn.Module(),minimum_bytes=1,policy=POLICY)
        self.checks=0;self.failures=[]

    def pack(self,tensor):
        return super().pack(tensor),tensor.detach().clone(),tensor.stride()

    def unpack(self,packed):
        result=super().unpack(packed[0]);original=packed[1];self.checks+=1
        same=(result.shape==original.shape and result.dtype==original.dtype and result.device==original.device
              and result.stride()==packed[2]
              and torch.equal(result.contiguous().reshape(-1).view(torch.uint8),original.contiguous().reshape(-1).view(torch.uint8)))
        if not same:self.failures.append(dict(shape=list(original.shape),expected_stride=list(packed[2]),actual_stride=list(result.stride())))
        return result


def check(report_path=None):
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA preflight required')
    from torch.nn.attention import sdpa_kernel, SDPBackend
    report=dict(passed=False,actual_CUDA=True,torch_version=torch.__version__,policy=POLICY,
                backend='EFFICIENT_ATTENTION',cases=[],new_optimizer_updates=0,loaded_model_weights=False,
                replaces_full_native_admission=False,strict_equality_requirement_unchanged=True,
                training_determinism_flag_changed=False)
    path=Path(report_path) if report_path is not None else None
    if path is not None:
        with path.open('x') as stream:json.dump(report,stream,indent=2)
    def persist():
        if path is not None:
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(path)
    try:
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
                    return [output.detach()]+[x.grad.detach().clone() for x in values]
                reference=run();repeat=run()
                row=dict(layout=layout,mask_shape=list(mask.shape),mask_stride=list(mask.stride()),
                         default_reference_repeat={name:difference(a,b) for name,a,b in zip(('output','dQ','dK','dV'),reference,repeat)})
                report['cases'].append(row);persist()
                with deterministic_comparison():
                    baseline=run();hook=RoundtripAudit();actual=run(hook)
                row.update(deterministic_reference_offload={name:difference(a,b) for name,a,b in zip(('output','dQ','dK','dV'),baseline,actual)},
                           roundtrip_checks=hook.checks,roundtrip_failures=hook.failures,
                           saved_tensors=hook.tensors,physical_saved_bytes=hook.saved_bytes)
                row['passed']=(hook.checks>0 and not hook.failures and all(v['exact'] and v['all_finite'] for v in row['deterministic_reference_offload'].values()))
                persist()
                if not row['passed']:raise ValueError('Strict deterministic CUDA kernel/storage equality failed; detailed report preserved')
        report['passed']=True;persist();return report
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error));persist();raise
