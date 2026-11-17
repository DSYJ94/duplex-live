import copy
import unittest

from training.personaplex.broad_corpus_filter import candidate, dolly, oasst


def message(identity, role, text, rank=0, replies=()):
    labels = {name: dict(value=0., count=3) for name in ('pii', 'spam', 'lang_mismatch')}
    labels['quality'] = dict(value=.8, count=3)
    return dict(message_id=identity, role=role, text=text, rank=rank, replies=list(replies),
                lang='en', review_result=True, deleted=False, synthetic=False,
                review_count=3, labels=labels)


class BroadCorpusFilterTest(unittest.TestCase):
    def fixture(self):
        answer = message('a2', 'assistant', 'An answer using the supplied follow-up.')
        followup = message('u2', 'prompter', 'Here is the missing context.', replies=[answer])
        preferred = message('a1', 'assistant', 'Please clarify.', replies=[followup])
        other = message('other', 'assistant', 'Alternative', rank=1)
        root = message('root', 'prompter', 'Explain this concept.', replies=[other, preferred])
        return dict(message_tree_id='tree', prompt=root, tree_state='ready_for_export'), {'root', 'other', 'a1', 'u2', 'a2'}

    def test_selected_path_preserves_context_and_prefers_reviewed_source_rank(self):
        tree, train = self.fixture()
        value, reason = oasst(tree, train, set(), 'pinned-revision')
        self.assertIsNone(reason)
        self.assertEqual(value['source_record_ids'], ['root', 'a1', 'u2', 'a2'])
        self.assertEqual([t['role'] for t in value['turns']], ['user', 'assistant', 'user', 'assistant'])
        self.assertEqual(value['turns'][2]['text'], 'Here is the missing context.')
        self.assertFalse(value['semantic_qualified'])
        self.assertFalse(value['admitted_for_training'])
        self.assertIsNone(value['split'])

    def test_validation_sibling_excludes_entire_tree_even_if_not_selected(self):
        tree, train = self.fixture()
        value, reason = oasst(tree, train - {'other'}, {'other'}, 'pinned')
        self.assertIsNone(value)
        self.assertEqual(reason, 'official_validation_tree')

    def test_unknown_or_failed_reviews_are_not_accepted_as_success(self):
        for change in ('pii_unknown', 'pii_positive', 'low_quality', 'synthetic', 'not_train'):
            tree, train = self.fixture()
            with self.subTest(change=change):
                root = tree['prompt']
                if change == 'pii_unknown': del root['labels']['pii']
                if change == 'pii_positive': root['labels']['pii']['value'] = .1
                if change == 'low_quality': root['labels']['quality']['value'] = .25
                if change == 'synthetic': root['synthetic'] = True
                if change == 'not_train': train.remove('root')
                value, _ = oasst(tree, train, set(), 'pinned')
                self.assertIsNone(value)

    def test_dolly_original_context_is_never_truncated_and_grouping_is_honest(self):
        context = ' '.join('source-%d' % i for i in range(400))
        row = dict(instruction='Use this source.', context=context, response='Answer with caveats.', category='closed_qa')
        value, reason = dolly(row, 4, 'pinned')
        self.assertIsNone(reason)
        self.assertIn(context, value['turns'][0]['text'])
        self.assertEqual(value['original_context'], context)
        self.assertTrue(value['requires_length_qualified_rewrite'])
        self.assertEqual(value['source_licence'], 'CC-BY-SA-3.0')
        changed = copy.deepcopy(row)
        changed['instruction'] = 'Another question on this source.'
        second, _ = dolly(changed, 5, 'pinned')
        self.assertEqual(value['source_group_id'], second['source_group_id'])
        self.assertNotEqual(value['content_sha256'], second['content_sha256'])

    def test_incomplete_turn_cannot_be_published_as_answered_episode(self):
        with self.assertRaises(ValueError):
            candidate('source', 'revision', 'licence', 'group', ['id'], [dict(role='user', text='Question')], 'family')
