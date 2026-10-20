import unittest
from pathlib import Path
from types import SimpleNamespace
from evals.native_ancestor_diagnostic import case,policy
from evals.native_text_sampling_diagnostic import policy as dispatch
class AncestorPolicy(unittest.TestCase):
 def args(self,index):
  r=case(index)
  return SimpleNamespace(native_ancestor_diagnostic=True,native_ancestor_diagnostic_case=index,scenario=r['scenario'],expected_kind='simple',backend_mode='observe-only',train_history=True,adapter_proof=Path('/runs/context-training-proof-'+r['checkpoint_job']),adapter_disabled=r['adapter_disabled'],adapter_zero=False,native_text_temperature=.7,warm_audio=False,ret_logit_bias=0.,pending_status=False,persona_profile='recorded-records',lookup_delivery_delay_s=0.,native_sampling_diagnostic=False,native_seen_retention_diagnostic=False,native_persona_diagnostic=False)
 def test_complete_three_way_four_questions(self):
  rows=[case(i) for i in range(12)]
  self.assertEqual(len({(r['scenario'],r['role']) for r in rows}),12)
  self.assertEqual(sum(r['adapter_disabled'] for r in rows),4)
  for i in range(12):self.assertEqual(dispatch(self.args(i))['role'],rows[i]['role'])
 def test_invalid_indices_fail(self):
  for i in [-1,12,True,None]:
   with self.assertRaises(ValueError):case(i)
 def test_only_declared_adapter_presence_may_change(self):
  for name,value in [('adapter_disabled',False),('adapter_zero',True),('scenario','other'),('adapter_proof',Path('/runs/context-training-proof-7558824'))]:
   a=self.args(0);setattr(a,name,value)
   with self.subTest(name=name),self.assertRaises(ValueError):policy(a)
 def test_sampler_persona_and_audio_forcing_cannot_change(self):
  for name,value in [('native_text_temperature',.3),('persona_profile','released-default'),('warm_audio',True),('ret_logit_bias',1.),('lookup_delivery_delay_s',2.),('native_sampling_diagnostic',True)]:
   a=self.args(1);setattr(a,name,value)
   with self.subTest(name=name),self.assertRaises(ValueError):policy(a)
 def test_index_without_diagnostic_is_rejected(self):
  a=self.args(1);a.native_ancestor_diagnostic=False
  with self.assertRaises(ValueError):dispatch(a)
