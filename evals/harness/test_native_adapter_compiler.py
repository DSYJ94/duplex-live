"""Actual small native gating: adapters must preserve published compiler semantics."""
import sys,unittest
from pathlib import Path
import torch
from torch import nn
from training.personaplex.distributed_native import configure_lm
VENDOR=Path('/projects/u6ada/duplex-live/vendor/context-spanning-79df-v1')

@unittest.skipUnless(VENDOR.exists(),'Actual pinned native vendor required')
class NativeAdapterCompiler(unittest.TestCase):
    def test_zero_adapter_keeps_real_bfloat16_native_gating_exact(self):
        sys.path.insert(0,str(VENDOR))
        from contextspan.moshi.modules.gating import ActivationGating
        class Native(nn.Module):
            def __init__(self):
                super().__init__();self.transformer=nn.Module();self.transformer.layers=nn.ModuleList([ActivationGating(32,128,torch.nn.functional.silu,dtype=torch.bfloat16)])
                self.text_linear=nn.Linear(32,32,bias=False,dtype=torch.bfloat16)
            def forward_codes(self,x):return self.transformer.layers[0](x)
            def forward_train(self,x):return self.forward_codes(x)
        torch.manual_seed(17);lm=Native().eval();x=torch.randn(2,33,32,dtype=torch.bfloat16)
        with torch.no_grad():before=lm.forward_codes(x).clone()
        configure_lm(lm,'broad-lora64')
        with torch.no_grad():after=lm.forward_codes(x)
        self.assertTrue(torch.equal(before,after),'Published compiled gating differs after zero adapter; max='+str(float((before-after).abs().max())))
        lm.forward_codes(x).float().square().mean().backward()
        self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad)for n,p in lm.named_parameters()if 'lora_B' in n))

if __name__=='__main__':unittest.main()
