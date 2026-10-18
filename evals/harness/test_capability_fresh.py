import json
from pathlib import Path
import unittest
from evals.capability_fresh import payload


class FreshCapabilityFixtures(unittest.TestCase):
    def test_closed_tasks_never_expose_cancellation_but_resume_remains_available(self):
        scenes=json.loads(Path('evals/fixtures/fresh-capability-controls-20261018.json').read_text())['cases']
        self.assertEqual(len(scenes),15)
        for scene in scenes:
            request,world,_,options=payload(scene,'fixture-model')
            names={x['name'] for x in request['tools']}
            self.assertNotIn('cancel_reasoning',names);self.assertNotIn('cancel_reasoning_and_resume_speech',names)
            self.assertFalse(request['local_worker_state']['available']['tasks'])
            self.assertEqual('resume_speech' in names,scene['kind']=='closed-resume')
            for identity in scene.get('completed_calculations',[]):
                self.assertTrue(world.tasks[identity]['closed']);self.assertFalse(world.tasks[identity]['cancelled'])
            self.assertEqual(bool(options),scene['kind']=='closed-resume')


if __name__=='__main__':unittest.main()
