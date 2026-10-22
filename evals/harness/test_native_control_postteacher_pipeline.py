"""Supervisor ordering fixtures; no API, speech or training approval."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import native_control_postteacher_pipeline as pipeline


class WholeApprovalChain(unittest.TestCase):
    def test_independent_verifiers_precede_paid_review_and_conversion(self):
        rows = list(pipeline.stages(*(Path(x) for x in ('raw', 'source', 'batch', 'parent', 'quality', 'out', 'weights', 'env'))))
        self.assertEqual([r[0] for r in rows], ['quality40', 'verify_quality40', 'semantic40',
                                              'verify_semantic40', 'convert40', 'verify_conversion40'])
        self.assertIn('--verify', rows[1][2])
        self.assertIn('--verify', rows[3][2])
        self.assertEqual(rows[-1][2], ['--verify', 'out/converted'])
        self.assertTrue(all(bound > 0 for _, _, _, bound in rows))
        self.assertFalse(any('train' in module or 'runpod' in module for _, module, _, _ in rows))

    def test_exact_protocol_still_requires_whole_quality_before_semantic(self):
        rows = list(pipeline.stages(*(Path(x) for x in ('raw', 'source', 'batch', 'parent', 'quality', 'out', 'weights', 'env')),
                                    exact_teacher=True))
        self.assertEqual(rows[0][1], 'evals.native_control_teacher_exact_quality')
        self.assertEqual(rows[1][1], 'evals.native_control_teacher_exact_quality')
        self.assertIn('--verify', rows[1][2])
        self.assertEqual(rows[2][1], 'evals.native_control_semantic')


if __name__ == '__main__':
    unittest.main()
