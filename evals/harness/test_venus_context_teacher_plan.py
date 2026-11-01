import json,unittest
from pathlib import Path
from data.venus.context_teacher_plan import build

class TeacherDraftTests(unittest.TestCase):
 def setUp(self):self.plan=json.loads(Path('evals/fixtures/venus-context-repair-authoring-v2-20261101.json').read_text());self.draft=build(self.plan)
 def test_complete_draft_never_claims_native_or_training_approval(self):
  self.assertEqual(len(self.draft['scenes']),36);self.assertFalse(self.draft['training_approved']);self.assertFalse(self.draft['actual_worker_receipts'])
  self.assertTrue(all(not s['training_approved'] for s in self.draft['scenes']))
 def test_selective_cancel_and_corrected_revision_preserve_scope(self):
  for scene in self.draft['scenes']:
   if scene['family']=='selective_cancel':
    self.assertEqual([a['task_label'] for a in scene['actions'] if a['kind']=='cancel'],['travel'])
    self.assertEqual([a['requires_result_for'] for a in scene['actions'] if 'requires_result_for' in a],['meal'])
   if scene['family']=='correction':
    replacements=[a for a in scene['actions'] if a['kind']=='replace'];self.assertEqual(len(replacements),1);self.assertEqual(replacements[0]['retire_revision'],1)
    result=scene['actions'][-1];self.assertEqual(result['revision'],2)
 def test_quiet_and_pause_have_no_acknowledgement_target(self):
  for scene in self.draft['scenes']:
   if scene['family']=='quiet_scope':self.assertEqual([a['after_user_turn'] for a in scene['actions']],[2])
   if scene['family']=='pause_resume':
    self.assertEqual(scene['actions'][1]['text'],'');self.assertTrue(scene['actions'][1]['speech_forbidden']);self.assertTrue(scene['actions'][2]['resume_without_restart'])
