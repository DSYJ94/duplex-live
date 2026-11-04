"""Run directly on deployment Python, without importing Torch or model weights."""
import hashlib
import time
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from portable.venus import speech_completion as completion


class CompletionRuntimeTest(unittest.TestCase):
    def test_pinned_method_transforms_and_executes_on_this_python(self):
        fixture = Path('evals/fixtures/venus_duplex_generate_4681004.py')
        namespace = {'time': time}
        exec(compile(fixture.read_text(), str(fixture), 'exec'), namespace)
        duplex = SimpleNamespace(is_session_stop_set=lambda: True,
                                 _generate_silence_waveform=lambda: [0],
                                 audio_chunk_idx=0)
        method = MethodType(namespace['streaming_generate'], duplex)
        with patch.object(completion, 'SOURCE_HASHES',
                          {hashlib.sha256(fixture.read_bytes()).hexdigest()}):
            transformed = completion._duplex_generate(method, SimpleNamespace())
        self.assertTrue(transformed()['is_listen'])
        self.assertIn('_completion', transformed.__func__.__code__.co_names)


if __name__ == '__main__':
    unittest.main()
