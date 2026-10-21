"""Synthetic transport fixture checks explicit source model and no fake instructions."""
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from evals.native_control_tts1hd import synthesize
from data.synthetic.native_control_render import load_clip


class ExplicitSynthesizer(unittest.TestCase):
    def test_exact_words_and_bytes_without_unsupported_instruction_payload(self):
        import numpy as np
        import soundfile as sf
        data = io.BytesIO()
        sf.write(data, np.full(4800, .1), 24000, format='WAV', subtype='PCM_16')
        body = data.getvalue()
        sent = []

        def post(url, **kwargs):
            sent.append(kwargs['json'])
            return types.SimpleNamespace(status_code=200, content=body, headers={'x-request-id': 'fixture-only'})

        requests = types.SimpleNamespace(post=post, exceptions=types.SimpleNamespace(
            Timeout=TimeoutError, ConnectionError=ConnectionError))
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', {'requests': requests}), patch.dict(
                os.environ, {'OPENAI_API_KEY': 'fixture-not-a-key'}):
            wav = Path(directory) / 'harvest.wav'
            synthesize('Harvest.', 'alloy', wav, instructions='must not be sent or claimed')
            self.assertEqual(sent, [dict(model='tts-1-hd', input='Harvest.', voice='alloy', response_format='wav')])
            self.assertEqual(wav.read_bytes(), body)
            receipt = json.loads(wav.with_suffix('.receipt.json').read_text())
            self.assertIsNone(receipt['instructions'])
            self.assertNotIn('fixture-not-a-key', wav.with_suffix('.receipt.json').read_text())
            _, result = load_clip(wav, dict(text='Harvest.', voice='alloy'))
            self.assertEqual(result['source_model'], 'tts-1-hd')
            with self.assertRaisesRegex(ValueError, 'scoped'):
                synthesize('Harvest.', 'marin', Path(directory) / 'forbidden.wav')
            self.assertEqual(len(sent), 1)


if __name__ == '__main__':
    unittest.main()
