import unittest
from evals.context_native_bf16_stream_diagnostic import call_partition,compare
from evals.context_native_audio49_training import validate_configuration

class BF16StreamDiagnostic(unittest.TestCase):
 def args(self):
  from evals.harness.test_context_native_audio49_training import Audio49PilotPolicy
  a=Audio49PilotPolicy().args();a.native_audio49_bf16_stream_diagnostic=True;a.native_audio49_preflight_only=True;a.native_audio49_preflight=None;a.native_audio49_gradient_probe=False;a.native_audio49_parent_adam=False;a.native_audio49_complete_batch=False;return a
 def test_complete_recorded_call_partition(self):
  call_partition([dict(start=0,frames=7),dict(start=7,frames=1),dict(start=8,frames=3)],11)
  for calls in [[dict(start=1,frames=11)],[dict(start=0,frames=10)],[dict(start=0,frames=7),dict(start=6,frames=5)],[dict(start=0,frames=True)]]:
   with self.assertRaises(ValueError):call_partition(calls,11)
 def test_mode_cannot_optimize_resume_or_mix_other_diagnostics(self):
  self.assertTrue(validate_configuration(self.args()))
  for name,value in [('native_audio49_preflight_only',False),('native_audio49_parent_adam',True),('native_audio49_complete_batch',True),('native_audio49_gradient_probe',True),('reload',True),('resume_training','state.pt'),('actual_fit_updates',32)]:
   a=self.args();setattr(a,name,value)
   with self.subTest(name=name),self.assertRaises(ValueError):validate_configuration(a)
 def test_raw_comparison_keeps_changed_predictions_and_excludes_masked_targets(self):
  import torch
  h=torch.zeros(1,3,2);z=torch.tensor([[[[3.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,3.],[0.,0.,0.,0.,3.,1.]]]])
  sz=z.clone();sz[0,0,1,0]=5.;mask=torch.tensor([False,True,True]);targets=torch.tensor([-100,5,4]);r=compare(h,z,h,sz,mask,targets)
  self.assertEqual(r['valid_targets'],2);self.assertEqual(r['spoken_targets'],1);self.assertEqual(r['valid_argmax_disagreements'],1);self.assertEqual(r['spoken_argmax_disagreements'],1);self.assertEqual(r['valid_logits_maximum_absolute_error'],5.);self.assertGreater(r['stream_target_ce'],r['batch_target_ce']);self.assertNotIn('passed',r)
