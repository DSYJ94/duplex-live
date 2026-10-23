import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evals.recovery_transport import ENDPOINT, PROXY_ENV, RUN_ENV, endpoint
from portable.context.local_responses_proxy import ProxyServer


class RecoveryTransport(unittest.TestCase):
    def test_provider_key_is_never_sent_to_the_loopback_client(self):
        url='http://127.0.0.1:18788/v1/responses'
        actual,headers,kind=endpoint({PROXY_ENV:url,RUN_ENV:'synthetic-native10-v1', 'OPENAI_API_KEY':'must-stay-local'})
        self.assertEqual(actual,url);self.assertEqual(headers,{'X-Duplex-Run':'synthetic-native10-v1'})
        self.assertNotIn('must-stay-local',json.dumps((actual,headers,kind)))
        self.assertEqual(endpoint({'OPENAI_API_KEY':'test-key'})[:2],(ENDPOINT,{'Authorization':'Bearer test-key'}))
        with self.assertRaises(RuntimeError):endpoint({})

    def test_proxy_cannot_be_an_arbitrary_host_or_unscoped_endpoint(self):
        for url in ('https://127.0.0.1:18788/v1/responses','http://example.com:18788/v1/responses',
                    'http://127.0.0.1:18788/other','http://user@127.0.0.1:18788/v1/responses',
                    'http://127.0.0.1:18788/v1/responses?target=example.com','http://127.0.0.1:bad/v1/responses'):
            with self.subTest(url=url),self.assertRaises(ValueError):endpoint({PROXY_ENV:url,RUN_ENV:'test'})
        with self.assertRaises(ValueError):endpoint({PROXY_ENV:'http://127.0.0.1:18788/v1/responses'})

    def test_actual_http_boundary_forwards_unchanged_data_but_keeps_auth_only_upstream(self):
        calls=[];payload=dict(model='gpt-6-luna',max_output_tokens=2048,input='authored synthetic question')
        reply=dict(id='unit-response',status='completed',output=[])
        def post(url,headers,json,timeout,allow_redirects):
            calls.append(dict(url=url,headers=headers,payload=json,timeout=timeout,redirects=allow_redirects))
            class Response:
                status_code=200
                content=__import__('json').dumps(reply).encode()
                headers={'x-request-id':'unit-provider-request'}
                def json(self):return reply
            return Response()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);server=ProxyServer(0,'server-only-unit-credential','synthetic-test',out,1,time.time()+60,post=post)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url='http://127.0.0.1:'+str(server.server_address[1])+'/v1/responses'
            def send(value,run='synthetic-test'):
                request=Request(url,json.dumps(value).encode(),headers={'X-Duplex-Run':run,'Content-Type':'application/json'})
                return urlopen(request,timeout=5)
            try:
                with self.assertRaises(HTTPError) as rejected:send(payload,run='wrong-execution')
                self.assertEqual(rejected.exception.code,403)
                with self.assertRaises(HTTPError) as rejected:send({**payload,'model':'different-model'})
                self.assertEqual(rejected.exception.code,400)
                with send(payload) as response:self.assertEqual(json.loads(response.read()),reply)
                with self.assertRaises(HTTPError) as rejected:send(payload)
                self.assertEqual(rejected.exception.code,429)
                self.assertEqual(len(calls),1);self.assertEqual(calls[0]['payload'],payload)
                self.assertEqual(calls[0]['url'],ENDPOINT)
                self.assertEqual(calls[0]['headers'],{'Authorization':'Bearer server-only-unit-credential'})
                self.assertFalse(calls[0]['redirects'])
                receipt=(out/'proxy.json').read_text()
                self.assertNotIn('server-only-unit-credential',receipt)
                self.assertNotIn('authored synthetic question',receipt)
                self.assertFalse(json.loads(receipt)['real_requests_transport'])
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)


if __name__=='__main__':unittest.main()
