import tempfile
import unittest
from pathlib import Path
from evals.context_component_kind import manifest_name


class MixedComponentScope(unittest.TestCase):
    def test_mixed_components_reject_missing_or_ambiguous_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(RuntimeError):manifest_name(root,mixed=True)
            (root/'early-future.json').touch()
            self.assertEqual(manifest_name(root,mixed=True),'early-future.json')
            (root/'actual-future.json').touch()
            with self.assertRaises(RuntimeError):manifest_name(root,mixed=True)
            (root/'early-future.json').unlink()
            self.assertEqual(manifest_name(root,mixed=True),'actual-future.json')

    def test_existing_explicit_modes_do_not_change(self):
        self.assertEqual(manifest_name(Path('/not-read'),early=True),'early-future.json')
        self.assertEqual(manifest_name(Path('/not-read')),'actual-future.json')


if __name__=='__main__':unittest.main()
