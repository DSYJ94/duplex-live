"""Experimental native voice audition through localhost SSH forwarding.

Uses the actual pinned engine/adapter/layout/voice, with unforced words/audio.
This audition has no external tool backend and is not an acceptance evaluation.
No microphone audio or conversation transcripts are persisted by this launcher.
"""
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


class NoExternalASR:
    def transcribe(self, clip, sample_rate):
        return ''


class ObserveOnlyBackend:
    last_source = 'observe-only'
    last_args = None

    def retrieve(self, *args, **kwargs):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--proof', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--ui-origin', default='http://127.0.0.1:8765')
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source))
    assets = json.loads((args.root/'runs/context-spanning-assets-7454205/manifest.json').read_text())
    vendor = Path(assets['source'])
    sys.path.insert(0, str(vendor))
    base = args.root/'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'
    os.environ.update(CS_BASE_DIR=str(base), CS_WEIGHTS_DIR=assets['path'],
                      CS_VOICES_DIR=str(Path(assets['path'])/'voices'), HF_HUB_OFFLINE='1')
    import torch
    from aiohttp import web
    from evals.context_spanning_probe import audited_loads
    from evals.context_adapter_load import before_engine
    from evals.lychee_context import BASE_INSTRUCTIONS
    from evals.context_execution import snapshot
    from contextspan.model.engine import Engine, seed_all
    import contextspan.model.engine as engine_module
    from contextspan.runtime import websocket_server as upstream
    records = []
    seed_all(42424242)
    with before_engine(engine_module, args.proof) as adapter:
        with audited_loads(records):
            engine = Engine(checkpoint=str(Path(assets['path'])/'context_spanning_7b.pt'),
                            device='cuda', temp=.8, temp_text=.7)
    actual = [r for r in records if r['module_identity'] == id(engine.lm)]
    assert actual and actual[-1]['strict'] and not actual[-1]['missing'] and not actual[-1]['unexpected']
    assert engine.frame_size == 1920 and engine.frame_rate == 12.5
    voice = torch.load(args.proof/'voice.pt', map_location='cpu', weights_only=True)['codes']
    receipt = {'schema': 'dl-native-voice-audition-v1', 'model_job': args.proof.name,
               'launcher_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
               'adapter': adapter, 'runtime': snapshot(engine.lm),
               'strict_published_lm_load': True, 'word_or_audio_forcing': False,
               'voice_codes_from_verified_proof': True, 'external_backend_connected': False,
               'conversation_audio_or_transcripts_saved': False,
               'native_full_numerical_or_behavior_approved': False}
    with args.receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')

    @web.middleware
    async def same_origin(request, handler):
        if request.path == '/ws' and request.headers.get('Origin') != args.ui_origin:
            raise web.HTTPForbidden(text='Use the local audition page')
        response = await handler(request)
        if request.path != '/ws':
            response.headers['Cache-Control'] = 'no-store'
        return response

    async def index(request):
        return web.FileResponse(Path(__file__).with_name('native_audition.html'))

    app = web.Application(middlewares=[same_origin])
    app.update(engine=engine, backend=ObserveOnlyBackend(), asr=NoExternalASR(),
               persona=BASE_INSTRUCTIONS, voice=voice, token='', listen_s=12.,
               raw_user_audio=True, enhancer=None, lock=asyncio.Lock())
    app.add_routes([web.get('/', index), web.get('/health', upstream.health),
                    web.get('/ws', upstream.ws_handler), web.static('/static', upstream.WEB)])
    logging.getLogger('contextspan.runtime.websocket_server').setLevel(logging.WARNING)
    logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    print('NATIVE_AUDITION_READY model7590052 actual sampled voice; external tools disconnected', flush=True)
    web.run_app(app, host='127.0.0.1', port=args.port, print=None, access_log=None)


if __name__ == '__main__':
    main()
