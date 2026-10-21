"""Prevent a rejected credential from fanning out paid source requests."""
import json
from pathlib import Path
import tempfile
import unittest

from data.synthetic.native_control_generate import collect
from data.synthetic.native_control_repair import build


class SourceCollectionFailure(unittest.TestCase):
    def test_rejected_probe_stops_collection_and_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.json'
            plan.write_text(json.dumps(build()))
            calls = []

            def rejected(text, voice, path):
                calls.append((text, voice))
                raise RuntimeError('Speech API rejected request: HTTP 401')

            with self.assertRaisesRegex(RuntimeError, 'Source collection failed'):
                collect(plan, root / 'source', 'fixture-not-a-key', root / 'lease', generate=rejected)
            self.assertEqual(len(calls), 1)
            ledger = json.loads((root / 'source/generation.json').read_text())
            self.assertEqual(ledger['failures'][0]['http_status'], 401)
            self.assertEqual(ledger['status'], 'failed_preserved_unapproved')
            self.assertFalse(ledger['training_approved'])
            self.assertNotIn('fixture-not-a-key', (root / 'source/generation.json').read_text())
            with self.assertRaisesRegex(ValueError, 'NEW version'):
                collect(plan, root / 'source', 'fixture-not-a-key', root / 'lease', generate=rejected)
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
