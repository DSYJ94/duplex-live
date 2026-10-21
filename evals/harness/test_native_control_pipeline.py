"""No generation failure or passing subset may trigger downstream API calls."""
import json
from pathlib import Path
import tempfile
import unittest

from evals.native_control_pipeline import generation_complete


class PipelineSourceGate(unittest.TestCase):
    def test_live_partial_and_failed_source_cannot_be_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / 'generation.json'
            ledger.write_text('{')
            self.assertFalse(generation_complete(root, root / 'unused-plan'))
            ledger.write_text(json.dumps(dict(status='collecting_unapproved')))
            self.assertFalse(generation_complete(root, root / 'unused-plan'))
            for status in ('failed_preserved_unapproved', 'interrupted_preserved_unapproved'):
                ledger.write_text(json.dumps(dict(status=status)))
                with self.assertRaisesRegex(ValueError, 'no downstream submissions'):
                    generation_complete(root, root / 'unused-plan')
            ledger.write_text(json.dumps(dict(status='complete_raw_unapproved', clips=[{}] * 167)))
            with self.assertRaisesRegex(ValueError, 'no downstream submissions'):
                generation_complete(root, root / 'unused-plan')


if __name__ == '__main__':
    unittest.main()
