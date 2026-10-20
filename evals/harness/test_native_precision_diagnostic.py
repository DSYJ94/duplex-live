import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from evals.native_precision_diagnostic import case,policy,convert
from evals.native_text_sampling_diagnostic import policy as dispatch

class PrecisionPolicy(unittest.TestCase):
 def args(self,i):
  r=case(i)
  return SimpleNamespace(native_precision_diagnostic=True,native_precision_diagnostic_case=i,scenario=r['scenario'],expected_kind='simple',backend_mode='observe-only',train_history=True,adapter_proof=Path('/runs/context-training-proof-'+r['checkpoint_job']),adapter_disabled=False,adapter_zero=False,native_text_temperature=.7,voice='accepted-synthetic',warm_audio=False,ret_logit_bias=0.,pending_status=False,persona_profile='recorded-records',lookup_delivery_delay_s=0.)
 def test_complete_checkpoint_precision_factorial(self):
  rows=[case(i) for i in range(16)];self.assertEqual(len({(r['scenario'],r['role'],r['precision']) for r in rows}),16)
  for i in range(16):self.assertEqual(dispatch(self.args(i))['precision'],rows[i]['precision'])
 def test_no_outside_checkpoint_sampler_or_context_change(self):
  for key,value in [('adapter_proof',Path('/runs/context-training-proof-7558824')),('native_text_temperature',.3),('persona_profile','released-default'),('adapter_disabled',True),('voice','released'),('native_ancestor_diagnostic',True),('native_precision_diagnostic',False)]:
   a=self.args(3);setattr(a,key,value)
   with self.subTest(key=key),self.assertRaises(ValueError):policy(a)
 def test_index_requires_optin_and_exact_complete_range(self):
  for i in [-1,16,True,None]:
   with self.assertRaises(ValueError):case(i)
  a=self.args(0);a.native_precision_diagnostic=False
  with self.assertRaises(ValueError):dispatch(a)
 def test_fp32_conversion_retains_every_frozen_numeric_value_and_integer_buffer(self):
  import torch
  from evals.moshi_training_smoke import tensor_hashes
  lm=torch.nn.Linear(3,2).bfloat16().requires_grad_(False);lm.register_buffer('integer',torch.tensor([7]));lm.register_buffer('high_precision',torch.tensor([1/3],dtype=torch.float64));before=tensor_hashes(lm.named_parameters());flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
  try:
   with patch('evals.context_execution.snapshot',side_effect=lambda m:dict(parameter_dtypes={str(p.dtype):1 for p in m.parameters()})):
    result=convert(lm,'fp32-lm')
   self.assertTrue(result['numeric_values_roundtrip_exact']);self.assertTrue(all(p.dtype==torch.float32 and not p.requires_grad for p in lm.parameters()));self.assertEqual(lm.integer.dtype,torch.int64);self.assertEqual(lm.high_precision.dtype,torch.float64);self.assertEqual(float(lm.high_precision[0]),1/3);self.assertEqual((torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32),flags)
   self.assertEqual(before,tensor_hashes((n,p.bfloat16()) for n,p in lm.named_parameters()))
  finally:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags

 def test_scoped_tf32_policy_is_explicit_and_cannot_orphan(self):
  a=self.args(3);a.native_precision_no_tf32=True;r=dispatch(a);self.assertIs(r['lm_matmul_tf32'],False);self.assertEqual(r['mode'],'native-checkpoint-precision16-no-tf32-diagnostic-v1')
  self.assertNotIn('lm_matmul_tf32',case(3));self.assertIs(case(3,no_tf32=True)['lm_matmul_tf32'],False)
  a.native_precision_diagnostic=False
  with self.assertRaises(ValueError):dispatch(a)
 def test_lm_tf32_scope_restores_original_policy_on_success_and_error(self):
  import torch
  from evals.native_precision_diagnostic import scoped_lm_no_tf32
  flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
  class LM:
   def forward_codes(self,value):
    if torch.backends.cuda.matmul.allow_tf32:raise AssertionError('LM TF32 must be disabled')
    return value+1
   def forward_depformer(self):
    if torch.backends.cuda.matmul.allow_tf32:raise AssertionError('LM TF32 must be disabled')
    raise ValueError('test-error')
  try:
   torch.backends.cuda.matmul.allow_tf32=True;lm=LM();e=scoped_lm_no_tf32(lm)
   self.assertEqual(lm.forward_codes(3),4);self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
   with self.assertRaises(ValueError):lm.forward_depformer()
   self.assertTrue(torch.backends.cuda.matmul.allow_tf32);self.assertEqual(torch.backends.cudnn.allow_tf32,flags[1]);self.assertEqual(e['wrapper_calls'],dict(forward_codes=1,forward_depformer=1));self.assertTrue(all(not x['matmul_tf32'] and x['float32_precision']=='highest' for x in e['first_call_execution'].values()))
  finally:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
