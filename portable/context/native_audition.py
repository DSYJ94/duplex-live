"""Experimental native voice audition through localhost SSH forwarding.

Uses the actual pinned engine/adapter/layout/voice, with unforced words/audio.
Default mode has no external backend. Opt-in live delegation uses actual arrived
audio recognition and native RET requests through the Mac-only Responses proxy.
This is an experiment, not an acceptance evaluation. Conversation history is not
saved; opt-in recognition uses transient audio files removed after each request.
"""
import argparse
import asyncio
import hashlib
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


def configure_cuda_toolkit():
    """Expose installed toolkit headers/assembler without changing packages."""
    include_dirs = [Path(p) for p in os.environ.get('CPATH', '').split(os.pathsep) if p]
    include_dirs += [Path(os.environ[p])/'include' for p in ('CUDA_HOME', 'CUDA_PATH')
                     if os.environ.get(p)]
    include_dirs.append(Path('/usr/local/cuda/include'))
    header = next((p/'cuda.h' for p in include_dirs if (p/'cuda.h').is_file()), None)
    if header is None:
        raise RuntimeError('Installed CUDA toolkit header cuda.h was not found')
    prior = os.environ.get('CPATH', '')
    if str(header.parent) not in prior.split(os.pathsep):
        os.environ['CPATH'] = str(header.parent)+(os.pathsep+prior if prior else '')
    assembler = Path(os.environ.get('TRITON_PTXAS_PATH',
                                   str(header.parent.parent/'bin/ptxas')))
    if not assembler.is_file() or not os.access(assembler, os.X_OK):
        raise RuntimeError('Installed CUDA assembler ptxas was not found')
    os.environ['TRITON_PTXAS_PATH'] = str(assembler)
    return {'header': str(header), 'header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
            'assembler': str(assembler),
            'assembler_sha256': hashlib.sha256(assembler.read_bytes()).hexdigest(),
            'packages_changed': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--proof', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--ui-origin', default='http://127.0.0.1:8765')
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--compiler-check-only', action='store_true')
    ap.add_argument('--live-delegation', action='store_true')
    ap.add_argument('--asr-python', type=Path)
    args = ap.parse_args()
    if args.receipt.exists() or (args.receipt.parent/'audition-driver').exists():
        ap.error('Use a new receipt directory; preserve prior driver aliases and receipts')
    if args.live_delegation:
        from evals.recovery_transport import endpoint, PROXY_ENV, RUN_ENV
        if (args.asr_python is None or not args.asr_python.is_file()
                or not os.environ.get(PROXY_ENV) or not os.environ.get(RUN_ENV)
                or os.environ.get('OPENAI_API_KEY')):
            ap.error('Live delegation requires isolated ASR and scoped Mac loopback transport; no provider key on this host')
        endpoint(os.environ)
    sys.path.insert(0, str(args.source))
    # Use the same driver-library discovery as the pinned native probe. Some
    # provider images expose libcuda.so.1 without the linker-name symlink.
    os.environ['TRITON_LIBCUDA_PATH'] = subprocess.check_output(
        [sys.executable, '-m', 'evals.cuda_linker_path', '--out',
         str(args.receipt.parent/'audition-driver')], text=True).strip()
    compiler = configure_cuda_toolkit()
    # Exercise the exact driver compiler before loading 16GB of model weights.
    from triton.backends.nvidia.driver import CudaUtils
    CudaUtils()
    compiler['actual_triton_driver_compile_passed'] = True
    import triton
    compiler['assembler_cuda_version'] = triton.knobs.nvidia.ptxas.version
    if args.compiler_check_only:
        import torch
        value = torch.ones(8, device='cuda', dtype=torch.bfloat16)
        result = torch.compile(lambda x: x + 1, fullgraph=True)(value)
        torch.cuda.synchronize()
        assert torch.equal(result, torch.full_like(value, 2))
        compiler['actual_compiled_bf16_gpu_kernel_passed'] = True
        with args.receipt.open('x') as stream:
            json.dump({'schema': 'dl-native-cuda-compiler-check-v1', **compiler}, stream, indent=2)
        print(json.dumps(compiler))
        return
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
    asr = NoExternalASR()
    handler = upstream.ws_handler
    if args.live_delegation:
        from portable.context.native_live_asr import IsolatedLiveASR
        from portable.context.native_live import ws_handler
        asr = IsolatedLiveASR(args.asr_python, args.source)
        handler = ws_handler
    receipt = {'schema': 'dl-native-voice-audition-v1', 'model_job': args.proof.name,
               'launcher_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
               'adapter': adapter, 'runtime': snapshot(engine.lm),
               'cuda_compiler': compiler,
               'strict_published_lm_load': True, 'word_or_audio_forcing': False,
               'voice_codes_from_verified_proof': True, 'external_backend_connected': False,
               'external_backend_configured': args.live_delegation,
               'actual_live_delegation_verified': False,
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

    async def configuration(request):
        return web.json_response(dict(external_backend_enabled=args.live_delegation,
            session_maximum_seconds=60 if args.live_delegation else None,
            native_behavior_qualified=False))

    app = web.Application(middlewares=[same_origin])
    app.update(engine=engine, backend=ObserveOnlyBackend(), asr=asr,
               persona=BASE_INSTRUCTIONS, voice=voice, token='', listen_s=12.,
               raw_user_audio=True, enhancer=None, lock=asyncio.Lock())
    app.add_routes([web.get('/', index), web.get('/health', upstream.health), web.get('/session', configuration),
                    web.get('/ws', handler), web.static('/static', upstream.WEB)])
    if args.live_delegation:
        async def cleanup(application):
            await asyncio.to_thread(asr.close)
        app.on_cleanup.append(cleanup)
    logging.getLogger('contextspan.runtime.websocket_server').setLevel(logging.WARNING)
    logging.getLogger('aiohttp.access').setLevel(logging.WARNING)
    print('NATIVE_AUDITION_READY '+args.proof.name+' actual sampled voice; external backend '+
          ('configured; live qualification pending' if args.live_delegation else 'disconnected'), flush=True)
    web.run_app(app, host='127.0.0.1', port=args.port, print=None, access_log=None)


if __name__ == '__main__':
    main()
