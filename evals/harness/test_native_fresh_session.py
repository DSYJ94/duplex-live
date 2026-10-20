import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evals import native_fresh_session as session


class FreshSession(unittest.TestCase):
 def test_complete40_pairs_fit_two_gpus_and_keep_two_for_live_asr(self):
  rows=session.cases('100');batches=session.batches(rows)
  self.assertEqual(len(rows),40);self.assertEqual(len(batches),28)
  self.assertEqual(sorted(i for batch in batches for i in batch),list(range(40)))
  for batch in batches:
   self.assertLessEqual(sum(rows[i]['required_gpus'] for i in batch),2)
   if rows[batch[0]]['backend']=='native-asr':self.assertEqual(len(batch),1);self.assertEqual(rows[batch[0]]['required_gpus'],2)
   else:self.assertEqual(len(batch),2);self.assertEqual(rows[batch[0]]['scenario'],rows[batch[1]]['scenario'])
  self.assertTrue(all(rows[i]['checkpoint_job']=='7510331' and rows[i+20]['checkpoint_job']=='100' for i in range(20)))
 def test_missing_cases_wrong_candidate_or_mixed_pair_cannot_schedule(self):
  for candidate in ['7510331','x','',None]:
   with self.assertRaises(ValueError):session.cases(candidate)
  with self.assertRaises(ValueError):session.batches(session.cases('100')[:-1])
  rows=session.cases('100');rows[20]['scenario']='different'
  with self.assertRaises(ValueError):session.batches(rows)
 def paths(self,root):
  review=root/'docs/review.json';review.parent.mkdir();review.write_text('fixture-only-review')
  quality=root/'quality.json';quality.write_text('fixture-only-quality')
  return review,root/'runs/native-fresh2-inputs-1',quality,root/'preflight.json'
 def test_rejected_development_review_blocks_source_reads_and_outputs(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);review,data,quality,out=self.paths(root)
   with patch.object(session,'ROOT',root),patch('evals.native_fresh_review.validate',side_effect=ValueError('rejected')),patch('evals.native_fresh_source_reapproval.approved_inputs') as inputs:
    with self.assertRaises(ValueError):session.preflight('100',review,data,quality,out)
    inputs.assert_not_called();self.assertFalse(out.exists())
 def test_preflight_rechecks_both_models_and_complete_source_without_adoption(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);review,data,quality,out=self.paths(root)
   identity=dict(native_protocol=dict(name='native'),checkpoint_sha256='fixture')
   parent=dict(voice_sha256='voice',data_sha256='data')
   rows=[dict(scenario_id='native-fresh2-%03d'%i) for i in range(20)]
   with patch.object(session,'ROOT',root),patch('evals.native_fresh_review.validate') as reviewed,patch('evals.context_rollout_conditioning.verify_checkpoint',return_value=(identity,parent)) as checkpoint,patch('evals.native_fresh_source_reapproval.approved_inputs',return_value=dict(rows=rows)) as inputs:
    r=session.preflight('100',review,data,quality,out)
    self.assertEqual(reviewed.call_count,2);self.assertEqual(checkpoint.call_count,2);self.assertEqual(inputs.call_count,1)
    self.assertTrue(r['complete40']);self.assertFalse(r['checkpoint_adopted']);self.assertFalse(r['training_approved']);self.assertFalse(r['scheduler_array'])
    out.unlink();inputs.return_value=dict(rows=rows[:-1])
    with self.assertRaises(ValueError):session.preflight('100',review,data,quality,out)
    self.assertFalse(out.exists())
 def test_actual_launch_assignment_has_no_fake_array_or_gpu_fallback(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'launch.json'
   for index,step,devices in [(True,'0','0'),(40,'0','0'),(8,'0','0'),(8,'0','0,0'),(0,'batch','0')]:
    with self.subTest(index=index,step=step,devices=devices),self.assertRaises(ValueError):session.launch('100',index,'200',step,devices,out)
   r=session.launch('100',28,'200','3','0,1',out)
   self.assertEqual(r['checkpoint_job'],'100');self.assertEqual(r['required_gpus'],2);self.assertFalse(r['scheduler_array']);self.assertFalse(r['training_approved']);self.assertEqual(json.loads(out.read_text())['allocation_job'],'200')
