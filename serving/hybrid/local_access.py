"""Loopback-only access helper for an authenticated SSH-forwarded voice server.

Keeps the ephemeral socket credential out of URLs, command output and repository
artifacts. It never reads an API key. Static files come through the SSH forward;
a same-origin browser request retrieves only this temporary session credential.
"""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import stat
import threading
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def read_token(path):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_mode&0o077:raise RuntimeError('Local session credential must be a private owned regular file')
    token=path.read_text().strip()
    if not 32<=len(token)<=128:raise RuntimeError('Invalid bounded temporary session credential')
    return token


def handler(token_file,upstream,port):
    allowed={'http://localhost:'+str(port),'http://127.0.0.1:'+str(port)}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,code,body,kind):
            self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('Referrer-Policy','no-referrer');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlsplit(self.path).path
            if self.headers.get('Origin') not in allowed|{None} or self.headers.get('Sec-Fetch-Site')=='cross-site':
                self.reply(403,b'Forbidden','text/plain');return
            if path=='/session-access':
                try:
                    token=read_token(token_file)
                    body=json.dumps(dict(token=token,websocket_url=upstream.replace('http://','ws://',1)+'/ws')).encode()
                    self.reply(200,body,'application/json')
                except Exception:self.reply(503,b'Session access is unavailable.','text/plain')
                return
            if path not in ('/','/client.js','/worklet.js'):self.reply(404,b'Not found','text/plain');return
            try:
                with urllib.request.urlopen(upstream+path,timeout=5) as response:self.reply(200,response.read(),response.headers['Content-Type'])
            except (OSError,urllib.error.URLError):self.reply(503,b'The voice service is not connected.','text/plain')
    return Handler


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--token-file',type=Path,required=True)
    ap.add_argument('--upstream',default='http://127.0.0.1:8768');ap.add_argument('--port',type=int,default=8767);ap.add_argument('--lifetime',type=int,default=14400);args=ap.parse_args()
    address=urlsplit(args.upstream)
    if address.scheme!='http' or address.hostname not in ('127.0.0.1','localhost') or address.path or address.query or address.fragment:
        raise ValueError('Access helper only proxies an explicit loopback SSH forward')
    if not 1<=args.lifetime<=14400:raise ValueError('Invalid temporary helper lifetime')
    read_token(args.token_file)
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler(args.token_file,args.upstream,args.port)) as server:
        timer=threading.Timer(args.lifetime,server.shutdown);timer.daemon=True;timer.start()
        print(json.dumps(dict(local_url='http://localhost:'+str(args.port),temporary=True,credential_printed=False)),flush=True)
        try:server.serve_forever()
        finally:timer.cancel()


if __name__=='__main__':main()
