import tempfile
import unittest
from pathlib import Path
from evals.context_component_kind import manifest_name,balanced_index


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
            (root/'action-future.json').touch()
            with self.assertRaises(RuntimeError):manifest_name(root,mixed=True)
            (root/'actual-future.json').unlink()
            self.assertEqual(manifest_name(root,mixed=True),'action-future.json')

    def test_existing_explicit_modes_do_not_change(self):
        self.assertEqual(manifest_name(Path('/not-read'),early=True),'early-future.json')
        self.assertEqual(manifest_name(Path('/not-read')),'actual-future.json')

    def test_request_prefix_is_explicitly_mixed_and_cannot_hide_another_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'request-prefix.json').touch()
            self.assertEqual(manifest_name(root,mixed=True),'request-prefix.json')
            (root/'action-future.json').touch()
            with self.assertRaises(RuntimeError):manifest_name(root,mixed=True)

    def test_balancing_equalises_kinds_without_losing_examples_or_resume_position(self):
        from collections import Counter
        kinds=['ordinary_answer']*3+['private_result']*11
        order=[balanced_index(kinds,s) for s in range(264)]
        self.assertEqual(Counter(kinds[i] for i in order),{'ordinary_answer':132,'private_result':132})
        counts=Counter(order)
        self.assertEqual([counts[i] for i in range(3)],[44]*3)
        self.assertEqual([counts[i] for i in range(3,14)],[12]*11)
        self.assertEqual(order[8:],[balanced_index(kinds,s) for s in range(8,264)])
        with self.assertRaises(ValueError):balanced_index([],0)


if __name__=='__main__':unittest.main()
