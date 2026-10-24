import subprocess
import unittest
from pathlib import Path

from evals.native_control_gradient_binding import loader_extension_only


class GradientBinding(unittest.TestCase):
    def test_only_exact_new_routing_can_be_removed(self):
        root = Path(__file__).resolve().parents[2]
        old = subprocess.check_output(['git', 'show', 'c9efaaa:evals/context_adapter_load.py'], cwd=root, text=True)
        current = (root / 'evals/context_adapter_load.py').read_text()
        self.assertTrue(loader_extension_only(old, current))
        mutations = (
            current.replace('lm.requires_grad_(False)', 'lm.requires_grad_(True)'),
            current.replace('verify_reasoning32(proof)', 'pass'),
            current.replace("protocol['name']!='bootstrap-preserve-provided-v1'", 'False'),
            current.replace('specific=True', 'specific=False', 1),
        )
        for changed in mutations:
            with self.subTest(change=changed != current), self.assertRaises(ValueError):
                loader_extension_only(old, changed)
        with self.assertRaises(ValueError):
            loader_extension_only(old, old)


if __name__ == '__main__':
    unittest.main()
