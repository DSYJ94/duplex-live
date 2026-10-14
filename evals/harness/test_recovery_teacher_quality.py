"""Teacher data must reject missing words, early facts and unnecessary workers."""
import copy
import unittest
from evals.recovery_teacher_quality import assess
from evals.recovery_teacher_convert import measured_timeline


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


class TeacherGridTests(unittest.TestCase):
    def setUp(self):
        self.source = {"events":[{"ch":"user","start_ms":0,"actual_start_ms":0,
            "actual_end_ms":1000,"text":"Check project Birch."}],"meta":{"kind":"private"}}
        self.words = [{"text":"Checked","start":3.,"end":3.3},{"text":"blue","start":3.4,"end":3.7}]

    def test_actual_word_clock_retained(self):
        timeline,offset = measured_timeline("teacher",self.words,self.source,[{"received_at_s":2.13}],[])
        event = next(e for e in timeline.events if e.ch == "assistant" and e.kind == "utterance")
        self.assertEqual(event.word_onsets_ms,[0,400])
        self.assertLessEqual(offset,40)
        marker = next(e for e in timeline.events if e.kind == "marker")
        self.assertGreaterEqual(marker.start_ms,2130)

    def test_causal_input_guard(self):
        with self.assertRaisesRegex(ValueError,"predates"):
            measured_timeline("teacher",self.words,self.source,[{"received_at_s":.8}],[])

    def test_no_invented_subword_intervals(self):
        self.words[0]["text"] = "checked blue"
        with self.assertRaisesRegex(ValueError,"own measured interval"):
            measured_timeline("teacher",self.words,self.source,[],[])

    def test_actual_collision_bar_retained(self):
        words = [{"text":str(i),"start":3.,"end":3.2} for i in range(5)]
        with self.assertRaisesRegex(ValueError,"lost|200ms"):
            measured_timeline("teacher",words,self.source,[],[])


if __name__ == "__main__":
    unittest.main()
