import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from data.venus import render_context_source as renderer


class Response:
    headers={'request-id':'fixture-only'}
    def __init__(self,data):self.data=data
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self):return self.data


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.payload=renderer.PAYLOAD.resolve()
        self.posts=0

    def open(self,request,**kwargs):
        url=request.full_url
        if url.endswith('/user/subscription'):return Response(b'{"character_limit":100000,"character_count":0}')
        if url.endswith('/models'):return Response(b'[{"model_id":"eleven_v3"}]')
        if '/voices/' in url:return Response(b'{}')
        self.posts+=1
        if self.posts==2:raise TimeoutError('Fixture ambiguous provider failure')
        return Response(bytes(48000))

    def test_unknown_attempt_is_counted_and_claim_blocks_second_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);auth=root/'fixture-auth.json'
            auth.write_text(json.dumps(dict(status='APPROVED',payload_sha256=hashlib.sha256(self.payload.read_bytes()).hexdigest(),maximum_paid_calls=72,maximum_additional_api_usd=5)))
            old=Path.cwd()
            try:
                os.chdir(root)
                with patch.object(renderer,'PAYLOAD',self.payload),patch.object(renderer,'AUTH',auth),patch.dict(os.environ,{'ELEVENLABS_API_KEY':'fixture-not-a-secret'}),patch.object(renderer.urllib.request,'urlopen',self.open):
                    with self.assertRaises(RuntimeError):renderer.execute(root/'first')
                    ledger=json.loads((root/'first/ledger.json').read_text())
                    self.assertEqual([x['status'] for x in ledger['attempts']],['complete','failed_or_unknown'])
                    self.assertGreater(ledger['conservative_attempted_cost_upper_usd'],ledger['attempts'][0]['conservative_attempt_cost_upper_usd'])
                    with self.assertRaises(FileExistsError):renderer.execute(root/'different')
                    self.assertEqual(self.posts,2)
                    self.assertFalse((root/'different').exists())
            finally:os.chdir(old)

    def test_missing_authorization_blocks_network_and_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(renderer,'AUTH',root/'absent'),patch.object(renderer.urllib.request,'urlopen') as network:
                with self.assertRaises(ValueError):renderer.execute(root/'output')
                network.assert_not_called()
                self.assertFalse((root/'output').exists())
