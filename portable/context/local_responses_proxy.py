"""Bounded Mac-only Responses proxy for synthetic native evaluation over SSH.

Listen only on127.0.0.1; forward only to the fixed official HTTPS endpoint. The
provider key is loaded from the existing local0600 file and never leaves this
process except in the official provider's Authorization header. No payload text,
auth headers or provider key is stored in receipts or HTTP access logs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time

from evals.native_control_teacher import atomic_json, collection_lock, local_key
from evals.recovery_transport import ENDPOINT

MODEL = 'gpt-6-luna'
TOOLS = {'lookup_record', 'cancel_lookup', 'replace_lookup', 'respond_to_user'}
MAX_BODY = 128 * 1024


def validate(payload, profile='context-lookup'):
    if profile == 'venus-reasoning':
        from portable.venus.reasoning import validate_payload
        return validate_payload(payload)
    if profile != 'context-lookup':
        raise ValueError('Unknown proxy request profile')
    if (not isinstance(payload, dict) or payload.get('model') != MODEL
            or type(payload.get('max_output_tokens')) is not int
            or not 0 < payload['max_output_tokens'] <= 2048
            or payload.get('stream', False) is not False
            or not isinstance(payload.get('input'), (str, list))):
        raise ValueError('Only the bounded existing native worker request is allowed')
    tools = payload.get('tools', [])
    if (not isinstance(tools, list) or len(tools) > 4
            or any(not isinstance(t, dict) or t.get('type') != 'function' or t.get('name') not in TOOLS for t in tools)):
        raise ValueError('Only the existing synthetic worker function definitions are allowed')


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, key, run, out, maximum_requests, deadline, post=None, profile='context-lookup'):
        import requests
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', run):
            raise ValueError('Explicit bounded synthetic execution identifier required')
        if type(maximum_requests) is not int or not 1 <= maximum_requests <= 128:
            raise ValueError('At most128 actual provider calls per proxy execution')
        if profile not in ('context-lookup', 'venus-reasoning'):
            raise ValueError('Explicit supported proxy profile required')
        self.profile = profile
        self.key, self.run, self.out = key, run, out
        self.post = requests.post if post is None else post
        self.lock = threading.RLock()
        self.deadline, self.maximum = deadline, maximum_requests
        self.count = 0
        self.receipts = []
        self.real_transport = post is None
        super().__init__(('127.0.0.1', port), Handler)
        self.timeout = 1
        self.record()

    def record(self):
        with self.lock:
            atomic_json(self.out / 'proxy.json', dict(schema='dl-local-responses-proxy-v1', pid=os.getpid(),
                run_id=self.run, listen_host='127.0.0.1', port=self.server_address[1],
                endpoint='http://127.0.0.1:' + str(self.server_address[1]) + '/v1/responses',
                upstream_https_endpoint=ENDPOINT, model=MODEL if self.profile == 'context-lookup' else 'gpt-6-astra',
                profile=self.profile, provider_credential_loaded_locally=True,
                provider_credential_transferred=False, actual_requests=self.count, maximum_requests=self.maximum,
                real_requests_transport=self.real_transport, deadline_utc=datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
                receipts=self.receipts, at_utc=datetime.now(timezone.utc).isoformat(),
                native_voice_or_routing_qualified=False, training_approved=False))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, body):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Duplex-Transport', 'provider' if self.server.real_transport else 'fixture')
        self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError): pass

    def do_GET(self):
        if self.path != '/health':
            self.reply(404, b'{"error":"not_found"}'); return
        s = self.server
        self.reply(200, json.dumps(dict(run_id=s.run, actual_requests=s.count,
                                      real_requests_transport=s.real_transport)).encode())

    def do_POST(self):
        s = self.server
        if self.path != '/v1/responses' or self.headers.get('X-Duplex-Run') != s.run:
            self.reply(403, b'{"error":"scope_rejected"}'); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY: raise ValueError('bounded request body required')
            body = self.rfile.read(length)
            if s.key.encode() in body: raise ValueError('credential must never be payload content')
            payload = json.loads(body)
            validate(payload, s.profile)
        except (ValueError, TypeError, json.JSONDecodeError):
            self.reply(400, b'{"error":"invalid_native_worker_payload"}'); return
        with s.lock:
            if s.count >= s.maximum or time.time() >= s.deadline:
                self.reply(429, b'{"error":"bounded_execution_finished"}'); return
            s.count += 1
            row = dict(index=s.count, request_sha256=hashlib.sha256(body).hexdigest(), status='inflight',
                       started_utc=datetime.now(timezone.utc).isoformat())
            s.receipts.append(row); s.record()
        began = time.monotonic()
        try:
            response = s.post(ENDPOINT, headers={'Authorization': 'Bearer ' + s.key},
                              json=payload, timeout=90, allow_redirects=False)
            data = response.content
            with s.lock:
                row.update(status='provider_returned', http_status=response.status_code,
                           response_sha256=hashlib.sha256(data).hexdigest(), seconds=time.monotonic()-began,
                           request_id=response.headers.get('x-request-id'))
                try:
                    value = response.json()
                    row['response_id'] = value.get('id')
                    if isinstance(value.get('usage'), dict):
                        row['usage'] = {k:v for k,v in value['usage'].items()
                                        if k in ('input_tokens', 'output_tokens', 'total_tokens') and type(v) is int and v >= 0}
                except (ValueError, AttributeError): pass
                s.record()
            self.reply(response.status_code, data)
        except Exception as exc:
            with s.lock:
                row.update(status='transport_error', error_type=type(exc).__name__, seconds=time.monotonic()-began)
                s.record()
            self.reply(502, b'{"error":"actual_provider_transport_failed"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--maximum-requests', type=int, default=128)
    parser.add_argument('--profile', choices=['context-lookup', 'venus-reasoning'], default='context-lookup')
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--lock', type=Path, default=Path('runs/local-migration-20261023/.private/native-responses-proxy.lock'))
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc.replace('Z', '+00:00')).timestamp()
    if deadline <= time.time(): raise ValueError('Future bounded proxy deadline required')
    key = local_key(args.env_file)
    with collection_lock(args.lock):
        args.out.mkdir(parents=True, exist_ok=False)
        server = ProxyServer(args.port, key, args.run_id, args.out, args.maximum_requests, deadline, profile=args.profile)
        print(json.dumps(dict(pid=os.getpid(), run_id=args.run_id, port=server.server_address[1], credential_transferred=False)), flush=True)
        try:
            while time.time() < deadline: server.handle_request()
        finally:
            server.server_close()
            server.record()


if __name__ == '__main__': main()
