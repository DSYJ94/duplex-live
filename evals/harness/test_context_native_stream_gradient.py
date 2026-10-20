import types
import unittest
from evals.context_native_stream_gradient import coefficients,detach_caches,validate


class NativeStreamGradient(unittest.TestCase):
    def test_global_balance_matches_original_loss_and_gradients_across_calls(self):
        import torch
        from torch.nn import functional as F
        from evals.context_teacher_text_balance import loss
        torch.manual_seed(17)
        z=torch.randn(9,12,requires_grad=True);y=torch.tensor([-100,3,5,3,6,0,7,3,-100]);m=y!=-100
        expected,_=loss(z,y,m);g=torch.autograd.grad(expected,z)[0]
        zz=z.detach().clone().requires_grad_();coef=coefficients(y,m);value=0.
        for start,stop in [(0,2),(2,5),(5,6),(6,9)]:
            active=m[start:stop]
            v=(F.cross_entropy(zz[start:stop][active],y[start:stop][active],reduction='none')*coef[start:stop][active]).sum()
            value+=float(v.detach());v.backward()
        self.assertAlmostEqual(value,float(expected),places=6)
        torch.testing.assert_close(zz.grad,g,rtol=1e-6,atol=1e-7)
        self.assertEqual(float(coef[~m].sum()),0.)
        self.assertAlmostEqual(float(coef[m & (y>=5)].sum()),.5,places=6)
        self.assertAlmostEqual(float(coef[m & (y<5)].sum()),.5,places=6)

    def test_real_ring_cache_detach_preserves_values_and_offsets_across_wraparound(self):
        import torch
        from contextspan.moshi.modules.transformer import RingKVCache
        torch.manual_seed(7)
        cache=RingKVCache(1,1,2,3,device=torch.device('cpu'),dtype=torch.float32)
        state=types.SimpleNamespace(kv_cache=cache)
        lm=types.SimpleNamespace(get_streaming_state=lambda:{'attention':state})
        w=torch.nn.Parameter(torch.randn(2,2));reference=[];observed=[]
        for step in range(7):
            x=torch.tensor([float(step+1),1.]);kv=(w@x).view(1,1,1,2)
            out=cache.complete(kv,kv*.5)
            # Consume only initialized slots, as native attention's position mask does.
            score=out.values[:,:,out.positions>=0].square().sum();score.backward()
            self.assertIsNotNone(w.grad);self.assertTrue(torch.isfinite(w.grad).all())
            before=cache.cache.detach().clone();offset=cache.end_offset.clone();ptr=cache.cache.data_ptr()
            self.assertEqual(detach_caches(lm),1)
            self.assertEqual(cache.cache.data_ptr(),ptr)
            self.assertFalse(cache.cache.requires_grad);self.assertIsNone(cache.cache.grad_fn)
            torch.testing.assert_close(cache.cache,before,rtol=0,atol=0)
            torch.testing.assert_close(cache.end_offset,offset,rtol=0,atol=0)
            observed.append(float(score.detach()));reference.append(before)
            w.grad=None
        self.assertEqual(int(cache.end_offset),7)
        self.assertEqual(len(observed),7)

    def test_untruncated_mutable_cache_reproduces_backward_graph_reuse_error(self):
        import torch
        from contextspan.moshi.modules.transformer import RingKVCache
        cache=RingKVCache(1,1,2,3,device=torch.device('cpu'),dtype=torch.float32)
        w=torch.nn.Parameter(torch.ones(2))
        first=cache.complete(w.view(1,1,1,2),w.view(1,1,1,2));first.values.square().sum().backward()
        with self.assertRaises(RuntimeError):
            second=cache.complete((w*2).view(1,1,1,2),(w*2).view(1,1,1,2));second.values.square().sum().backward()

    def test_empty_and_one_group_masks_cannot_weaken_balance(self):
        import torch
        for y,m in [(torch.tensor([3,3]),torch.tensor([True,True])),(torch.tensor([5,6]),torch.tensor([True,True])),(torch.tensor([3,5]),torch.tensor([False,False]))]:
            with self.assertRaises(ValueError):coefficients(y,m)
        with self.assertRaises(ValueError):detach_caches(types.SimpleNamespace(get_streaming_state=lambda:{}))

    def test_readiness_is_not_training_or_another_diagnostic(self):
        from evals.harness.test_context_native_bf16_stream_diagnostic import BF16StreamDiagnostic
        args=BF16StreamDiagnostic().args();args.native_audio49_bf16_stream_diagnostic=False
        args.native_audio49_stream_auxiliary=True;args.native_audio49_parent_adam=True;args.native_audio49_complete_batch=True
        validate(args)
        for name,value in [('native_audio49_bf16_stream_diagnostic',True),('reload',True),('resume_training','state.pt'),('native_audio49_preflight_only',False),('native_audio49_complete_batch',False)]:
            changed=types.SimpleNamespace(**vars(args));setattr(changed,name,value)
            with self.subTest(name=name),self.assertRaises(ValueError):validate(changed)
