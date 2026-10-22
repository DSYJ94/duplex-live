"""Input reflection and echoed configuration are required, not transcript guesses."""
import base64
import hashlib
import io
import unittest
from evals import native_control_webrtc_capture as capture


class WebRTCCapture(unittest.TestCase):
    def test_reflection_preserves_all_pcm_bytes_and_offsets(self):
        target = io.BytesIO()
        for pcm in (b'\x01\x00\xff\xff', b'\x00\x00'):
            before = target.tell()
            result = capture.reflected_input(dict(audio=base64.b64encode(pcm).decode()), target)
            self.assertEqual(result['byte_offset'], before)
            self.assertEqual(result['bytes'], len(pcm))
            self.assertEqual(result['pcm_sha256'], hashlib.sha256(pcm).hexdigest())
        self.assertEqual(target.getvalue(), b'\x01\x00\xff\xff\x00\x00')
        for bad in ('invalid!', base64.b64encode(b'x').decode(), ''):
            with self.assertRaises(ValueError): capture.reflected_input(dict(audio=bad), target)

    def test_echo_preserves_every_requested_setting(self):
        expected = capture.configuration()
        self.assertTrue(capture.configuration_matches(dict(expected, status='running'), expected))
        for field, value in [('model','other'), ('instructions','different'), ('delegation',{})]:
            self.assertFalse(capture.configuration_matches(dict(expected, **{field:value}), expected))


if __name__ == '__main__': unittest.main()
