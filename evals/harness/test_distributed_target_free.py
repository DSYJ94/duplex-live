"""Real Gloo reducer over target-free prefixes and audio-only objectives."""
import copy,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import torch
from torch import nn
import torch.distributed as dist
from training.personaplex.distributed_native import DistributedNative
from training.personaplex.matched_losses import recorded_text

class Native(nn.Module):
    def __init__(self):
        super().__init__();self.transformer=nn.Module();self.transformer.layers=nn.ModuleList([nn.Sequential(nn.Linear(3,4))])
        self.text_linear=nn.Linear(4,8);self.depformer=nn.Module();self.depformer.layers=nn.ModuleList([nn.Linear(4,4)])
        self.cache=torch.zeros(1,2,1,1,1);self.flags=[]
    def forward_codes(self,x):
        self.flags.append(torch.is_grad_enabled());h=self.transformer.layers[0](x.transpose(1,2));return h,self.text_linear(h)[:,None]
    def forward_train(self,x):
        h,z=self.forward_codes(x);return self.depformer.layers[0](h),z
    def streaming(self,batch):return nullcontext()
    def get_streaming_state(self):return {'layer':SimpleNamespace(kv_cache=SimpleNamespace(cache=self.cache))}

class TargetFree(unittest.TestCase):
    def run_native(self,callback):
        with tempfile.TemporaryDirectory()as folder:
            dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=0,world_size=1)
            try:
                torch.manual_seed(17);lm=Native().eval();runtime=DistributedNative(lm,'broad-lora64','cpu');callback(lm,runtime)
            finally:dist.destroy_process_group()
    def test_prefix_no_targets_has_no_reduction_and_preserves_supervised_gradient(self):
        def exercise(lm,runtime):
            reference=copy.deepcopy(lm);x=torch.randn(1,3,3);mask=torch.tensor([False,True,True]);targets=torch.tensor([0,6,3]);calls=[dict(start=i,frames=1,phase='prefix'if i==0 else'live')for i in range(3)]
            expected=recorded_text(reference,x,mask,targets,calls,1.)
            actual=recorded_text(runtime.view,x,mask,targets,calls,1.)
            self.assertEqual(expected,actual);self.assertEqual(lm.flags,[False,True,True]);self.assertEqual(actual['gradient_calls'],2)
            self.assertTrue(torch.equal(lm.cache,reference.cache))
            for (n,p),(m,q)in zip(lm.named_parameters(),reference.named_parameters()):
                self.assertEqual(n,m)
                if p.grad is None:self.assertIsNone(q.grad)
                else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
        self.run_native(exercise)
    def test_audio_unused_text_output_does_not_poison_next_reduction(self):
        def exercise(lm,runtime):
            x=torch.randn(1,3,3)
            for _ in range(2):
                runtime.wrapped.zero_grad(set_to_none=True);out=runtime.view.forward_train(x);out[0].square().mean().backward()
                self.assertIsNone(lm.text_linear.lora_B.weight.grad)
                self.assertGreater(lm.transformer.layers[0][0].lora_B.weight.grad.abs().sum(),0)
        self.run_native(exercise)

if __name__=='__main__':unittest.main()
