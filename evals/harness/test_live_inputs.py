"""Plain words: reject unverified live-demo fixtures before loading a model,
and preserve actual clip endings instead of padding each answer to 17 hours.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from data.synthetic.quality import DataNotVerified
from m2 import live


class LiveInputTest(unittest.TestCase):
    def test_unverified_fixture_stops_before_model_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["live", "--ckpt", "unused", "--batch", tmp,
                    "--scenarios", "unverified", "--out", tmp]
            with patch("sys.argv", argv), self.assertRaises(DataNotVerified):
                live.main()

    def test_playback_clip_ends_at_its_real_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answer.wav"
            sf.write(path, np.ones(320, dtype=np.float32) * 0.1, 16000)
            clip = live.load_stem_mono(path)
            self.assertEqual(len(clip), 320)
            # Timeline stems can still explicitly request zero padding.
            stem = live.load_stem_mono(path, 640)
            self.assertEqual(len(stem), 640)
            self.assertTrue(np.all(stem[320:] == 0))


if __name__ == "__main__":
    unittest.main()
