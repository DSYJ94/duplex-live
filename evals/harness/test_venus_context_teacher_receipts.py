import json,unittest
from pathlib import Path
from data.venus.context_teacher_plan import build
from data.venus.context_teacher_receipts import collect

class TeacherReceiptTests(unittest.IsolatedAsyncioTestCase):
 async def test_full36_real_local_work_preserves_retired_results_without_delivering_them(self):
  plan=json.loads(Path('evals/fixtures/venus-context-repair-authoring-v2-20261101.json').read_text());by_id={s['id']:s for s in plan['scenes']};draft=build(plan)
  rows=[await collect(by_id[d['id']],d) for d in draft['scenes']]
  self.assertEqual(sum(len(r['all_worker_receipts']) for r in rows),36)
  self.assertEqual(sum(len(r['delivered_receipts']) for r in rows),24)
  self.assertEqual(sum(e['kind']=='stale_result_delivery_suppressed' for r in rows for e in r['events']),12)
  for r in rows:
   retired={e['work_id'] for e in r['events'] if e['kind']=='delivery_retired'}
   self.assertFalse(retired & {e['work_id'] for e in r['delivered_receipts']})
   self.assertFalse(r['native_student_dispatch_claimed']);self.assertFalse(r['training_approved'])
   if by_id[r['scene_id']]['family']=='correction':
    self.assertEqual([x['revision'] for x in r['delivered_receipts']],[2])
   if by_id[r['scene_id']]['family']=='selective_cancel':
    result=r['delivered_receipts'][0]['result'];self.assertEqual(result['total_GBP'],{'pasta':18,'soup':12});self.assertNotIn('travel',result)
