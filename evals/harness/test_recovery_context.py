import unittest
from evals.recovery_context import causal_dialogue


class CausalContext(unittest.TestCase):
    def test_questions_stay_separate_without_future_or_script_answers(self):
        sidecar = {"timeline":{"events":[
            {"ch":"user","kind":"utterance","actual_start_ms":0,"actual_end_ms":2000,"text":"untrusted script"},
            {"ch":"assistant","kind":"utterance","actual_start_ms":2100,"actual_end_ms":2800,"text":"PRIVATE SECRET"},
            {"ch":"user","kind":"utterance","actual_start_ms":3000,"actual_end_ms":5000,"text":"future script"}]},
            "user_words":[{"text":"Check Birch","start_s":.5,"end_s":1.5},
                {"text":"Why","start_s":3.1,"end_s":3.3},{"text":"lightning?","start_s":4.5,"end_s":4.8}]}
        packets = [{"text":" Checking.","packet_available_s":2.5},
            {"text":"future answer","packet_available_s":4.}]
        result = causal_dialogue(sidecar,3.4,packets,3.4)
        users = [entry for entry in result if entry["role"] == "user"]
        self.assertEqual([entry["text"] for entry in users],["Check Birch","Why"])
        self.assertTrue(users[0]["utterance_complete"])
        self.assertFalse(users[1]["utterance_complete"])
        self.assertEqual(len({entry["utterance_id"] for entry in users}),2)
        self.assertNotIn("SECRET",str(result))
        self.assertNotIn("future",str(result))
        self.assertIn("Checking",str(result))


if __name__ == "__main__": unittest.main()
