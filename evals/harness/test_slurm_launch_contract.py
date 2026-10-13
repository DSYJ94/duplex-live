"""Plain words: execute the real launcher with a fake container, without GPUs.

Shell syntax checks miss a blank line terminating a continued command. This
test checks what the trainer would actually receive, including its data split.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LaunchContractTest(unittest.TestCase):
    def test_delegation_passes_selection_initialization_and_destination(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "repo").mkdir()
            shards = root / "runs" / "shards-merged"
            shards.mkdir(parents=True)
            for sid in ("q2_dense_dialog-001", "q2_dense_barge-001", "d_delegation-001",
                        "q2_dense_dialog-113", "q2_dense_barge-113"):
                (shards / (sid + ".pt")).touch()
            bindir = root / "bin"
            bindir.mkdir()
            fake = bindir / "apptainer"
            fake.write_text("#!/usr/bin/env python3\nimport json, os, sys\n"
                            "with open(os.environ['CAPTURE'], 'a') as f:\n"
                            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n")
            fake.chmod(0o755)
            source = (repo / "evals" / "deleg-round.slurm").read_text()
            source = source.replace("DL=/projects/u6si/duplex-live", f'DL="{root}"')
            script = root / "run.sh"
            script.write_text(source)
            capture = root / "calls.jsonl"
            env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                       CAPTURE=str(capture))
            proc = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            calls = [json.loads(row) for row in capture.read_text().splitlines()]
            train = next(row for row in calls if "layera.train_p1" in row)
            for flag in ("--lessons", "--init-ckpt", "--out"):
                self.assertIn(flag, train, f"Trainer lost {flag}; stderr: {proc.stderr}")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(train[train.index("--out") + 1], str(root / "runs" / "deleg-round"))
            lessons = train[train.index("--lessons") + 1].split(",")
            self.assertIn("d_delegation-001", lessons)
            self.assertNotIn("q2_dense_dialog-113", lessons)
            diagnose = next(row for row in calls if "layera.freerun_diag" in row)
            held = diagnose[diagnose.index("--scenarios") + 1].split(",")
            self.assertEqual(set(held), {"q2_dense_dialog-113", "q2_dense_barge-113"})
            self.assertFalse(set(held) & set(lessons))


if __name__ == "__main__":
    unittest.main()
