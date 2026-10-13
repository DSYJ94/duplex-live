"""Check normalized sample values, channel order and slices for voice loading."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from evals.minicpm_recovery_probe import read_reference_wav


class ReferenceWavTest(unittest.TestCase):
    def test_preserves_channels_rate_normalization_and_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.wav"
            pcm = np.array([[0.0, 0.25], [0.5, -0.5], [-0.25, 0.0]], dtype=np.float32)
            sf.write(path, pcm, 24000, subtype="PCM_16")
            actual, rate = read_reference_wav(path, backend="soundfile")
            self.assertEqual(rate, 24000)
            np.testing.assert_array_equal(actual, pcm.T)
            sliced, _ = read_reference_wav(path, frame_offset=1, num_frames=1, channels_first=False)
            np.testing.assert_array_equal(sliced, pcm[1:2])
            with self.assertRaises(ValueError):
                read_reference_wav(path, normalize=False)


if __name__ == "__main__":
    unittest.main()
