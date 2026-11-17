import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.personaplex.broad_corpus_overlap import Screen, patterns, registry, run

QUESTION = 'A workshop has 123 washers and uses 5 before receiving 2 more. How many washers remain?'


class BroadCorpusOverlapTest(unittest.TestCase):
    def value(self):
        return dict(schema='dl-known-evaluation-lexical-registry-v1', patterns=patterns(QUESTION, 'protected-input'))

    def test_exact_number_swaps_and_embedded_prompt_are_withheld(self):
        screen = Screen(self.value())
        for text in (QUESTION, QUESTION.replace('123', '456').replace('5', '7'), 'Use the supplied example carefully. ' + QUESTION + ' Explain your answer briefly.'):
            with self.subTest(text=text):
                self.assertTrue(screen.matches([dict(role='user', text=text)]))
        self.assertFalse(screen.matches([dict(role='user', text='Explain why a database uses an index to speed up lookups.')]))
        self.assertFalse(screen.matches([dict(role='assistant', text=QUESTION)]))

    def test_registry_exports_input_hashes_not_questions_answers_or_train_items(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'evals/fixtures').mkdir(parents=True)
            (root / 'docs').mkdir()
            source = dict(scenes=[dict(id='protected', fold='DEV', question=QUESTION, expected_answer='SECRET-GOLD'),
                                  dict(id='training-item', fold='TRAIN', question='PRIVATE-TRAIN-TEXT'),
                                  dict(id='unknown-input', fold='TEST', unsupported_input='UNHANDLED')])
            (root / 'evals/fixtures/input.json').write_text(json.dumps(source))
            value = registry(root)
            encoded = json.dumps(value)
            self.assertNotIn(QUESTION, encoded)
            self.assertNotIn('SECRET-GOLD', encoded)
            self.assertNotIn('PRIVATE-TRAIN-TEXT', encoded)
            self.assertEqual(value['counts']['unhandled_reviewed_or_evaluation_rows'], 1)
            self.assertFalse(value['semantic_decontamination_complete'])

    def test_full_candidate_screen_preserves_kept_bytes_and_originals(self):
        with tempfile.TemporaryDirectory() as temp, patch('builtins.print'):
            root = Path(temp)
            incoming = root / 'incoming'
            incoming.mkdir()
            sources = {}
            for dataset, folder in [('oasst', 'OpenAssistant--oasst1'), ('dolly', 'databricks--databricks-dolly-15k')]:
                directory = incoming / folder
                directory.mkdir()
                rows = [dict(id=dataset + str(i), source_group_id=dataset, content_sha256=str(i),
                             admitted_for_training=False, semantic_qualified=False,
                             turns=[dict(role='user', text=text)]) for i, text in enumerate([QUESTION, 'Explain why database indexes help queries.'])]
                data = ''.join(json.dumps(row) + '\n' for row in rows)
                (directory / 'candidates.jsonl').write_text(data)
                sources[dataset] = dict(eligible_text_candidates=2)
            (incoming / 'eligibility.json').write_text(json.dumps(dict(sources=sources)))
            fingerprints = root / 'registry.json'
            fingerprints.write_text(json.dumps(self.value()))
            result = run(incoming, fingerprints, root / 'out')
            self.assertEqual(sum(v['withheld_known_input_overlap'] for v in result['sources'].values()), 2)
            self.assertEqual(sum(v['retained_lexical_candidates'] for v in result['sources'].values()), 2)
            for folder in ('OpenAssistant--oasst1', 'databricks--databricks-dolly-15k'):
                before = (incoming / folder / 'candidates.jsonl').read_text().splitlines(keepends=True)
                self.assertEqual((root / 'out' / folder / 'candidates.jsonl').read_text(), before[1])
                self.assertEqual(len(before), 2)
            self.assertTrue(result['semantic_near_duplicate_checks_pending'])
            self.assertFalse(result['admitted_for_training'])

    def test_explicit_protected_file_and_lowercase_splits_remain_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'evals/fixtures').mkdir(parents=True)
            (root / 'docs').mkdir()
            value = dict(do_not_train=True, cases=[dict(id='protected', context=[
                dict(role='user', text=QUESTION), dict(role='assistant', text='SECRET-GOLD')]),
                dict(id='source-train', split='train', question='EXCLUDED-TRAIN-INPUT')])
            (root / 'evals/fixtures/source.json').write_text(json.dumps(value))
            result = registry(root)
            self.assertTrue(Screen(result).matches([dict(role='user', text=QUESTION)]))
            self.assertNotIn('SECRET-GOLD', json.dumps(result))
            self.assertNotIn('EXCLUDED-TRAIN-INPUT', json.dumps(result))
