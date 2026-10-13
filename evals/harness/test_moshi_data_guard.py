"""Plain words: altered or unapproved native artifacts must stop before training."""
import json
import tempfile
import unittest
from pathlib import Path

from evals.moshi_training_smoke import file_hash, validate_data


class NativeDataGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = self.root / "approved-source"
        self.data = self.root / "converted"
        self.batch.mkdir()
        self.data.mkdir()
        quality = [{"schema": "dl-quality-v1", "scenario_id": f"dialog-{i}",
                    "verified": True, "timing": "ok"} for i in range(20)]
        (self.batch / "quality.jsonl").write_text("".join(json.dumps(x)+"\n" for x in quality))
        rows = []
        for i in range(20):
            sid = f"dialog-{i}"
            source = self.batch / sid
            target = self.data / sid
            source.mkdir()
            target.mkdir()
            for name in ("timeline.actual.json", "stem_assistant.wav", "stem_user.wav"):
                (source / name).write_bytes(f"source-fixture-{i}-{name}".encode())
            wav = target / "conversation.wav"
            wav.write_bytes(f"converted-fixture-{i}".encode())
            sidecar = wav.with_suffix(".json")
            sidecar.write_text('{"alignments": [["hello", [0.1, 0.5], "SPEAKER_MAIN"]]}')
            rows.append({"source_batch": str(self.batch), "scenario_id": sid,
                         "source_row": {"scenario_id": sid, "dir": sid}, "verified": True,
                         "path": str(wav), "wav_sha256": file_hash(wav),
                         "sidecar_sha256": file_hash(sidecar),
                         "source_timeline_sha256": file_hash(source / "timeline.actual.json"),
                         "source_assistant_sha256": file_hash(source / "stem_assistant.wav"),
                         "source_user_sha256": file_hash(source / "stem_user.wav"),
                         "source_quality_sha256": file_hash(self.batch / "quality.jsonl")})
        (self.data / "ingestion.json").write_text(json.dumps({"accepted": rows}))
        for label, selected in (("train16", rows[:16]), ("eval4", rows[16:])):
            (self.data / f"{label}.jsonl").write_text("".join(
                json.dumps({"path": r["path"], "duration": 1.})+"\n" for r in selected))
        (self.data / "summary.json").write_text(json.dumps({"verified": True, "manifest_sha256": {
            name: file_hash(self.data / name) for name in ("train16.jsonl", "eval4.jsonl")}}))

    def test_approved_unchanged_artifacts_pass(self):
        validate_data(self.data)

    def test_changed_alignment_aborts(self):
        (self.data / "dialog-0/conversation.json").write_text('{"alignments": []}')
        with self.assertRaisesRegex(RuntimeError, "artifacts changed"):
            validate_data(self.data)

    def test_duplicate_source_identity_aborts(self):
        path = self.data / "ingestion.json"
        value = json.loads(path.read_text())
        value["accepted"][1] = value["accepted"][0]
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            validate_data(self.data)

    def test_revoked_source_approval_aborts(self):
        path = self.batch / "quality.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["verified"] = False
        path.write_text("".join(json.dumps(r)+"\n" for r in records))
        # Even updating the evidence hash must not bypass the live quality check.
        ing = self.data / "ingestion.json"
        value = json.loads(ing.read_text())
        for row in value["accepted"]:
            row["source_quality_sha256"] = file_hash(path)
        ing.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "no passing quality record"):
            validate_data(self.data)


if __name__ == "__main__":
    unittest.main()
