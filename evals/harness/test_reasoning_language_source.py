import hashlib
import json
import unittest
from training.personaplex.reasoning_language_source import build


class ReasoningLanguageSourceTest(unittest.TestCase):
    def source(self):
        return dict(exact_semantic_records=[dict(id='source-'+str(i),
            fold='TRAIN' if i < 1008 else 'DEV', question='Question '+str(i),
            response='Exact original answer '+str(i)) for i in range(1072)])

    def encode(self, value):
        raw = json.dumps(value).encode()
        return raw, hashlib.sha256(raw).hexdigest()

    def test_exact_train_words_ids_and_no_implied_qualification(self):
        source = self.source()
        rows = build(*self.encode(source))
        self.assertEqual(len(rows), 1008)
        originals = {r['id']: r for r in source['exact_semantic_records']}
        for row in rows:
            old = originals[row['id']]
            self.assertEqual(old['fold'], 'TRAIN')
            self.assertEqual(row['turns'], [dict(role='user', text=old['question']),
                dict(role='assistant', text=old['response'])])
            self.assertEqual(row['original_semantic_record'], old)
            self.assertFalse(row['admitted_for_training'])
            self.assertFalse(row['family_grouping_complete'])
            self.assertFalse(row['licence_qualified'])

    def test_changed_bytes_partial_membership_and_missing_answer_rejected(self):
        raw, digest = self.encode(self.source())
        with self.assertRaises(ValueError): build(raw+b' ', digest)
        source = self.source(); source['exact_semantic_records'].pop()
        with self.assertRaises(ValueError): build(*self.encode(source))
        source = self.source(); source['exact_semantic_records'][0]['response'] = ''
        with self.assertRaises(ValueError): build(*self.encode(source))


if __name__ == '__main__': unittest.main()
