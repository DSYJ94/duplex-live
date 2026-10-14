"""Teacher data must reject missing words, early facts and unnecessary workers."""
import copy
import unittest
from evals.recovery_teacher_quality import assess


class TeacherQualityTests(unittest.TestCase):
    def setUp(self):
        text = "Light travels faster than sound. Blue circle."
        self.speech = {"model_text":text,"wer_against_model_text":0.,"audio_duration_s":20.,
            "asr_segments":[{"words":[{"text":w,"start":12+i*.3,"end":12.2+i*.3} for i,w in enumerate(text.split())]}]}
        self.meta = {"project":"Birch North","field":"badge phrase","phrase":"blue circle"}
        self.events = [{"type":"session.delegation.created"},{"type":"local.worker.tool_result",
            "received_at_s":10.,"task_id":"work","result":{"content":"Blue circle.","receipts":[{
                "executed":True,"read_only":True,"tool":"lookup_record",**self.meta,"value":"blue circle"}]}}]
        self.session = {"session_finalized":True,"errors":[]}

    def test_receipt_and_measured_words_required(self):
        result = assess(self.speech,self.events,self.meta,"private",self.session)
        self.assertTrue(result["speech_and_fact_checks_pass"])
        self.assertFalse(result["training_approved"])
        bad = copy.deepcopy(self.speech)
        bad["asr_segments"][0]["words"].pop()
        self.assertFalse(assess(bad,self.events,self.meta,"private",self.session)["speech_and_fact_checks_pass"])

    def test_early_fact_rejected(self):
        self.events[1]["received_at_s"] = 19.
        self.assertIn("Checked private phrase missing or spoken before the actual tool result",
            assess(self.speech,self.events,self.meta,"private",self.session)["reasons"])

    def test_wrong_tool_value_rejected(self):
        self.events[1]["result"]["receipts"][0]["value"] = "red star"
        self.assertFalse(assess(self.speech,self.events,self.meta,"private",self.session)["speech_and_fact_checks_pass"])

    def test_simple_worker_rejected(self):
        self.meta["source_events"] = [{"ch":"user","text":"Why do I see lightning before hearing thunder?"}]
        self.assertFalse(assess(self.speech,self.events,self.meta,"simple",self.session)["speech_and_fact_checks_pass"])

    def test_invalid_measured_interval_rejected(self):
        self.speech["asr_segments"][0]["words"][0]["end"] = float("nan")
        self.assertFalse(assess(self.speech,self.events,self.meta,"private",self.session)["speech_and_fact_checks_pass"])


if __name__ == "__main__":
    unittest.main()
