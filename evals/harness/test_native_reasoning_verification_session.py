"""Safety boundaries for the nested source-readback performance repair."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.native_reasoning_verification_session import Memo, inventory, source_readbacks


class VerificationSessionTests(unittest.TestCase):
    def test_real_v4_source_readback_reused_through_32_teacher_alias_calls(self):
        from evals.harness.test_native_reasoning_source_pacing import PacingTests
        from evals import native_reasoning_source_pacing as pacing
        from evals import native_reasoning_teacher_v6 as teacher
        fixture = PacingTests()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source, quality, failed, binding = fixture.fixture(root)
            with patch.object(pacing, 'preflight', return_value=binding):
                pacing.collect(source, quality, failed, root / 'take', None,
                               post=lambda *a, **k: fixture.response())
                pacing.prepare(source, quality, root / 'take', root / 'v4', require_actual=False)
                expected = pacing.verify(root / 'v4', require_actual=False)
                with source_readbacks([root]) as memo:
                    for _ in range(32):
                        self.assertEqual(teacher.source_manifest(root / 'v4', require_actual=False), expected)
                key = 'evals.native_reasoning_source_pacing.verify'
                self.assertEqual(memo.calls[key], 1)
                self.assertEqual(memo.hits[key], 31)

    def test_distinct_strictness_and_path_are_never_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); calls = []
            def verify(path, *, require_actual=True):
                calls.append((path, require_actual)); return {'rows': [str(path)]}
            memo = Memo([root]); checked = memo.wrap(verify)
            result = checked(root); result['rows'].append('caller mutation')
            self.assertEqual(checked(root, require_actual=True), {'rows': [str(root)]})
            checked(root, require_actual=False); checked(root / 'other')
            self.assertEqual(len(calls), 3)

    def test_failures_never_populate_cache_and_outside_paths_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); calls = []
            def verify(path): calls.append(path); raise ValueError('original rejection')
            memo = Memo([root]); checked = memo.wrap(verify)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'original rejection'): checked(root)
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(ValueError, 'escaped'): checked(root.parent)

    def test_added_changed_removed_files_and_symlinks_change_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'wave'; path.write_bytes(b'original')
            before = inventory([root]); path.write_bytes(b'changed!')
            self.assertNotEqual(inventory([root]), before)
            path.write_bytes(b'original'); extra = root / 'extra'; extra.write_bytes(b'x')
            self.assertNotEqual(inventory([root]), before)
            extra.unlink(); path.unlink(); self.assertNotEqual(inventory([root]), before)
            path.symlink_to('/tmp')
            with self.assertRaises(ValueError): inventory([root])

    def test_actual_imported_aliases_restore_and_changed_inputs_abort(self):
        from evals import native_reasoning_teacher_v6 as teacher
        from evals import native_reasoning_source_pacing as pacing
        original = pacing.verify
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); path = root / 'input'; path.write_text('before')
            with self.assertRaisesRegex(ValueError, 'changed during'):
                with source_readbacks([root]) as memo:
                    self.assertIs(teacher.source_manifest, pacing.verify)
                    self.assertIsNot(pacing.verify, original)
                    path.write_text('after')
            self.assertIs(pacing.verify, original)
            self.assertIs(teacher.source_manifest, original)
            self.assertEqual(memo.cache, {})


if __name__ == '__main__': unittest.main()
