import unittest
from evals.context_backend_reference import reference
from evals.recovery_worker import RecordWorld


class BackendReferenceTest(unittest.TestCase):
    def setUp(self):
        self.world = RecordWorld({'Alder North': {'room marker': 'teal star'}})
        self.receipt = self.world.read('ret-0', 'Alder North', 'room marker')
        # Captured failure: the lookup succeeds, but prose answers the later
        # physics question and omits the requested private record entirely.
        self.result = {'route': 'delegate', 'receipts': [self.receipt],
            'content': 'Light travels much faster than sound.'}

    def test_successful_receipt_survives_unrelated_backend_prose(self):
        text, count = reference(self.result, self.world)
        self.assertIn('The room marker for Project Alder North is teal star.', text)
        self.assertIn(self.result['content'], text)
        self.assertEqual(count, 1)

    def test_cancelled_or_replaced_result_cannot_be_injected(self):
        for action in ('cancel', 'replace'):
            with self.subTest(action=action):
                self.setUp()
                if action == 'cancel': self.world.cancel('ret-0')
                else: self.world.reserve('ret-0')
                with self.assertRaisesRegex(RuntimeError, 'stale'):
                    reference(self.result, self.world)

    def test_fabricated_value_with_valid_task_revision_is_rejected(self):
        self.result['receipts'] = [{**self.receipt, 'value': 'invented value'}]
        with self.assertRaisesRegex(RuntimeError, 'Unverified'):
            reference(self.result, self.world)

    def test_failed_request_does_not_promote_executed_read_to_success(self):
        self.result.update(route='error', content='The backend failed to provide a usable reply.')
        text, count = reference(self.result, self.world)
        self.assertEqual(text, self.result['content'])
        self.assertNotIn('teal star', text)
        self.assertEqual(count, 0)

    def test_answer_without_lookup_is_unchanged(self):
        self.result.update(route='answer', receipts=[])
        self.assertEqual(reference(self.result, self.world), (self.result['content'], 0))

    def test_plain_context_preserves_both_actual_fact_and_independent_answer(self):
        text, count = reference(self.result, self.world, plain_text=True)
        self.assertEqual(text, 'The room marker for Project Alder North is teal star.\n'
            'Light travels much faster than sound.')
        self.assertEqual(count, 1)

    def test_plain_context_cannot_reintroduce_a_cancelled_value(self):
        self.world.cancel('ret-0')
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            reference(self.result, self.world, plain_text=True)

    def test_plain_context_does_not_duplicate_identical_backend_sentence(self):
        self.result['content'] = 'The room marker for Project Alder North is teal star.'
        text, _ = reference(self.result, self.world, plain_text=True)
        self.assertEqual(text, self.result['content'])


if __name__ == '__main__': unittest.main()
