import unittest
from evals.recovery_worker import WorkerContractError
from evals.context_backend_failure import notice


class BackendFailureTest(unittest.TestCase):
    def test_blocked_identification_stays_a_failure_and_requests_clarification(self):
        result = notice(WorkerContractError('Private project was not identified by the user',
            {'arguments': {'project': 'Cedar'}, 'events': []}), 'ret-0')
        self.assertEqual(result['route'], 'error')
        self.assertIn('which project', result['content'])
        self.assertEqual(result['receipts'], [])

    def test_executed_receipt_is_kept_as_evidence_without_inventing_a_reply(self):
        receipt = {'tool': 'lookup_record', 'value': 'violet triangle', 'executed': True}
        result = notice(WorkerContractError('Backend returned empty result text',
            {'receipts': [receipt]}), 'ret-0')
        self.assertEqual(result['receipts'], [receipt])
        self.assertEqual(result['route'], 'error')
        self.assertNotIn('violet triangle', result['content'])

    def test_unknown_runtime_failure_is_not_silently_converted(self):
        with self.assertRaises(TypeError): notice(RuntimeError('ASR stopped'), 'ret-0')


if __name__ == '__main__': unittest.main()
