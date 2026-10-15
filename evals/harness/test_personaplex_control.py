import unittest
from evals.personaplex_control import feedback_schedule, live_feedback_state


class ControlFeedbackTests(unittest.TestCase):
    def test_action_does_not_leak_into_its_own_neural_input(self):
        actions = [0]*20
        actions[3] = 1
        states = feedback_schedule(actions, [{"ready_at_s":.8}], "private")
        self.assertEqual(states[:4], [1]*4)
        self.assertEqual(states[4:10], [2]*6)
        self.assertEqual(states[10:], [3]*10)

    def test_fast_cancellation_receipt_is_never_a_future_action_hint(self):
        actions = [0]*20
        actions[3], actions[10] = 1, 2
        states = feedback_schedule(actions, [{"ready_at_s":.8}], "cancel")
        self.assertEqual(states[10], 2)
        self.assertEqual(states[11], 5)

    def test_pending_cancel_is_distinct_from_successful_cancel(self):
        jobs = {"read":{"observed":False,"action":1}, "cancel":{"observed":False,"action":2}}
        self.assertEqual(live_feedback_state(jobs, {"read"}, []), 4)
        jobs["cancel"]["observed"] = True
        events = [{"event":"worker_completed", "result":{"receipts":[{"tool":"cancel_lookup","cancelled":True}]}}]
        self.assertEqual(live_feedback_state(jobs, {"read"}, events), 5)

    def test_unobserved_future_is_still_pending(self):
        self.assertEqual(live_feedback_state({"read":{"observed":False,"action":1}}, set(), []), 2)
        self.assertEqual(live_feedback_state({}, set(), []), 1)


if __name__ == "__main__":
    unittest.main()
