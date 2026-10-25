import json
from pathlib import Path
import unittest

from evals.native_gcp_transfer_inventory import rows
from portable.context.native_zoneb_preservation import semantic_entries, ordered_files


class ClosureOrderingTests(unittest.TestCase):
    def test_saved_inventory_can_be_unsorted_without_changing_any_rows(self):
        original = [dict(path=name, bytes=1, sha256='a'*64) for name in ('z.json', 'a.json')]
        ordered = ordered_files(original)
        self.assertEqual(ordered, list(reversed(original)))
        self.assertEqual([r['path'] for r in original], ['z.json', 'a.json'])

    def test_duplicates_are_rejected_rather_than_deduplicated(self):
        row = dict(path='same.json', bytes=1, sha256='a'*64)
        with self.assertRaises(ValueError):
            ordered_files([row, dict(row)])

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
