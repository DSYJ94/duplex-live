import json
from pathlib import Path
import unittest

from evals.native_gcp_transfer_inventory import rows
from portable.context.native_zoneb_preservation import semantic_entries


class ClosureOrderingTests(unittest.TestCase):
    def test_actual_complete_closure_satisfies_reader_without_dropping_or_changing_entries(self):
        root = Path(__file__).resolve().parents[2]
        closure = json.loads((root/'docs/GCP-NATIVE-SEMANTIC-RECEIPT-CLOSURE-2026-10-25.json').read_text())
        entries = semantic_entries(closure)
        self.assertEqual(len(entries), 33)
        self.assertEqual(rows(entries), sum(row['bytes'] for row in closure['files']))
        self.assertEqual({row['path']: (row['bytes'], row['sha256']) for row in entries},
                         {row['name']: (row['bytes'], row['sha256']) for row in closure['files']})


if __name__ == '__main__':
    unittest.main()
