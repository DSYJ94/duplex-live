"""Teacher data must reject missing words, early facts and unnecessary workers."""
import copy
import unittest
import tempfile
from pathlib import Path
from evals.recovery_teacher_quality import assess,validate_input_audio
from evals.recovery_teacher_convert import measured_timeline


class TeacherQualityTests(unittest.TestCase):
    def setUp(self):
        text = "Light travels faster than sound. Blue circle."
        self.speech = {"model_text":text,"wer_against_model_text":0.,"audio_duration_s":20.,
            "asr_segments":[{"words":[{"text":w,"start":12+i*.3,"end":12.2+i*.3} for i,w in enumerate(text.split())]}]}
        self.meta = {"project":"Birch North","field":"badge phrase","phrase":"blue circle"}
        self.events = [{"type":"session.delegation.created"},{"type":"local.worker.tool_result",
            "received_at_s":10.,"task_id":"work","result":{"content":"Blue circle.","receipts":[{
                "executed":True,"read_only":True,"tool":"lookup_record",**self.meta,"value":"blue circle"}]}} ,{"type":"session.closed"}]
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

    def test_actual_event_finalization_required(self):
        self.events.pop()
        self.assertFalse(assess(self.speech,self.events,self.meta,"private",self.session)["speech_and_fact_checks_pass"])

    def test_clarification_before_identification_and_lookup_after_it(self):
        self.meta["source_events"] = [
            {"ch":"user","kind":"utterance","actual_start_ms":0,"actual_end_ms":1000},
            {"ch":"user","kind":"utterance","actual_start_ms":5000,"actual_end_ms":6000}]
        text = "Which project? Liquid gas. Blue circle."
        times = [2., 2.3, 7., 7.3, 12., 12.3]
        self.speech["model_text"] = text
        self.speech["asr_segments"][0]["words"] = [
            {"text":word,"start":start,"end":start+.2} for word,start in zip(text.split(),times)]
        self.events[0]["received_at_s"] = 6.5
        self.assertTrue(assess(self.speech,self.events,self.meta,"clarify",self.session)["speech_and_fact_checks_pass"])
        self.events[0]["received_at_s"] = 4.
        self.assertIn("Private lookup dispatched before project identification completed",
            assess(self.speech,self.events,self.meta,"clarify",self.session)["reasons"])

    def test_cancel_requires_actual_receipt_before_acknowledgement(self):
        text = "Light travels faster than sound. Cancelled."
        self.speech["model_text"] = text
        self.speech["asr_segments"][0]["words"] = [
            {"text":w,"start":12+i*.3,"end":12.2+i*.3} for i,w in enumerate(text.split())]
        self.events = [
            {"type":"session.delegation.created","received_at_s":2.,"delegation":{"id":"lookup"}},
            {"type":"session.delegation.created","received_at_s":8.,"delegation":{"id":"cancel"}},
            {"type":"local.worker.tool_result","received_at_s":10.,"task_id":"cancel",
             "result":{"content":"Cancelled.","receipts":[{"tool":"cancel_lookup","cancelled":True,"task_ids":["lookup"]}]}},
            {"type":"session.closed"}]
        self.assertTrue(assess(self.speech,self.events,self.meta,"cancel",self.session)["speech_and_fact_checks_pass"])
        self.events[2]["received_at_s"] = 19.
        self.assertIn("Spoken cancellation acknowledgement missing or predates actual receipt",
            assess(self.speech,self.events,self.meta,"cancel",self.session)["reasons"])

    def test_cancel_rejects_incomplete_target_set(self):
        self.test_cancel_requires_actual_receipt_before_acknowledgement()
        self.events[2]["received_at_s"] = 10.
        self.events.insert(1,{"type":"session.delegation.created","received_at_s":3.,"delegation":{"id":"duplicate"}})
        self.assertIn("Cancellation receipt does not cancel all prior related lookups",
            assess(self.speech,self.events,self.meta,"cancel",self.session)["reasons"])

    def test_teacher_input_must_match_approved_audio(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/"source.wav"
            recorded = Path(temp)/"recorded.wav"
            expected = np.full(2400,.2,dtype=np.float32)
            sf.write(source,expected,24000,subtype="FLOAT")
            actual = np.concatenate([expected,np.zeros(15*24000,dtype=np.float32)])
            sf.write(recorded,actual,24000,subtype="FLOAT")
            validate_input_audio(recorded,source)
            actual[0] = .8
            sf.write(recorded,actual,24000,subtype="FLOAT")
            with self.assertRaisesRegex(RuntimeError,"differs"):
                validate_input_audio(recorded,source)


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
