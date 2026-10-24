import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from portable.context import native_reasoning_pilot as pilot


class ReasoningPilot(unittest.TestCase):
    def test_sequence_preserves_old_approvals_and_new_complete_stages(self):
        names = (*pilot.INPUTS, 'python', 'root', 'out', 'source', 'control_cache',
                 'control_approval', 'development_evidence', 'prepared49',
                 'original_inventory', 'workset_inventory')
        args = SimpleNamespace(**{n: Path('/fixture') / n for n in names})
        with patch.object(pilot, 'arguments', return_value=['--actual-fit-updates', '8']) as old:
            stages = dict(pilot.commands(args))
        self.assertEqual(list(stages), ['whole_workset', 'numerical32', 'gradient32',
            'combined_readiness', 'bounded8_training', 'independent_restore', 'strict_proof'])
        self.assertEqual(old.call_args.args[3:5], (args.control_approval, args.control_approval))
        for name in ('combined_readiness', 'bounded8_training', 'independent_restore'):
            command = stages[name]
            self.assertEqual(command[command.index('--actual-fit-updates')+1], '8')
            for field in ('inputs', 'version', 'semantic', 'data', 'model', 'codec-source',
                          'interleaver-source', 'cache', 'engine', 'numerical', 'gradient'):
                self.assertIn('--reasoning32-' + field, command)
        self.assertNotIn('--reload', stages['bounded8_training'])
        self.assertIn('--reload', stages['independent_restore'])
        self.assertIn('evals.native_reasoning_stage', stages['strict_proof'][-1])

    def test_fit_requires_restore_headroom_and_numeric_is_bounded(self):
        self.assertEqual(pilot.stage_seconds('numerical32', 100, 20000, 4000), 3900)
        with self.assertRaises(TimeoutError):
            pilot.stage_seconds('bounded8_training', 100, 10899, 4000)
        self.assertEqual(pilot.stage_seconds('bounded8_training', 100, 10900, 4000), 7200)
        with self.assertRaises(TimeoutError):
            pilot.stage_seconds('gradient32', 20000, 20000, 4000)

    def test_guard_rejects_stale_wrong_pod_and_short_deadline(self):
        guard = dict(phase='armed', pod_id='new', at_utc='1970-01-01T00:01:40+00:00',
                     deadline_utc='1970-01-01T01:00:00+00:00')
        pilot.guard_check(guard, 'new', 3000, 100)
        for changed, pod, now in ((guard, 'old', 100), (guard, 'new', 191),
                                  ({**guard, 'phase': 'pod_terminated'}, 'new', 100)):
            with self.assertRaises(ValueError):
                pilot.guard_check(changed, pod, 3000, now)
        with self.assertRaises(ValueError):
            pilot.guard_check(guard, 'new', 3500, 100)


if __name__ == '__main__':
    unittest.main()
