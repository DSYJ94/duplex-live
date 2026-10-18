from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from serving.hybrid.local_access import read_token,handler


class LocalSessionAccess(unittest.TestCase):
    def test_private_file_required_and_cross_origin_cannot_read_session_credential(self):
        with tempfile.TemporaryDirectory() as folder:
            token=Path(folder)/'token';token.write_text('fixture-only-not-a-real-secret-123456789');token.chmod(0o644)
            with self.assertRaises(RuntimeError):read_token(token)
            token.chmod(0o600)
            with ThreadingHTTPServer(('127.0.0.1',0),handler(token,'http://127.0.0.1:8768',0)) as server:
                port=server.server_port;server.RequestHandlerClass=handler(token,'http://127.0.0.1:8768',port)
                thread=threading.Thread(target=server.serve_forever);thread.start()
                try:
                    url='http://127.0.0.1:'+str(port)+'/session-access'
                    with urllib.request.urlopen(url) as response:
                        data=json.load(response);self.assertEqual(data['token'],read_token(token));self.assertEqual(data['websocket_url'],'ws://127.0.0.1:8768/ws')
                        self.assertEqual(response.headers['Cache-Control'],'no-store')
                    request=urllib.request.Request(url,headers={'Origin':'https://foreign.invalid'})
                    with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(request)
                    self.assertEqual(error.exception.code,403)
                finally:server.shutdown();thread.join(timeout=2)
