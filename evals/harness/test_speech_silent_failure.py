"""A silent teacher is rejected without preventing later clips being checked."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class SilentSpeechTests(unittest.TestCase):
    def test_silent_failure_retained_and_next_clip_transcribed(self):
        import numpy as np
        import soundfile as sf
        from evals.moshi_speech_audit import main
        from evals.recovery_teacher_quality import assess
        transcribed = []

        def transcribe(model, audio, **kwargs):
            transcribed.append(len(audio))
            return {"text": "Hello.", "segments": [{"start": 0., "end": .5,
                "words": [{"text": "Hello.", "start": 0., "end": .5}]}]}

        whisper = types.SimpleNamespace(load_model=lambda *a, **k: object(), transcribe=transcribe)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = [root / "silent", root / "spoken"]
            for i, run in enumerate(runs):
                run.mkdir()
                audio = np.zeros(24000, dtype=np.float32) if i == 0 else np.full(24000, .1, dtype=np.float32)
                sf.write(run / "model_out.wav", audio, 24000, subtype="FLOAT")
                (run / "transcripts.json").write_text(json.dumps([{"speaker": "assistant", "text": "Hello."}]))
            argv = ["audit", "--out", str(root / "audit"), "--device", "cpu",
                "--trim-trailing-silence", "--keep-silent-failures", "--live-runs", *map(str, runs)]
            with patch.dict(sys.modules, {"whisper_timestamped": whisper}), patch.object(sys, "argv", argv), patch(
                    "importlib.metadata.version", return_value="test"):
                main()
            results = json.loads((root / "audit" / "report.json").read_text())["results"]
            self.assertEqual(len(results), 2)
            self.assertEqual(len(transcribed), 1)
            self.assertEqual(results[0]["asr_segments"], [])
            self.assertIn("audit_failure", results[0])
            self.assertEqual(results[1]["asr_text"], "Hello.")
            decision = assess(results[0], [{"type": "session.closed"}],
                {"source_events": [{"ch": "user", "text": "What is a triangle?"}]},
                "simple", {"session_finalized": True, "errors": []})
            self.assertFalse(decision["speech_and_fact_checks_pass"])
            self.assertIn(results[0]["audit_failure"], decision["reasons"])


if __name__ == "__main__":
    unittest.main()
