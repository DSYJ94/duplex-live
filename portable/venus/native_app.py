"""Native Venus microphone prototype and synthetic-only diagnostic endpoint.

The released streaming model owns speech/listen decisions. No keyword mute, ASR
controller, canned ordinary answers or forced speech. A bounded arithmetic worker
handles ONLY model-emitted delegate requests; it is not a reasoning-model backend.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


def ensure_guard():
    g=json.loads(Path(os.environ['DUPLEX_LIVE_GUARD_FILE']).read_text())
    from .guard import resource
    lease=json.loads(Path(os.environ['DUPLEX_LIVE_LEASE_FILE']).read_text()) if os.environ.get('DUPLEX_LIVE_LEASE_FILE') else {}
    b=resource(lease)
    if (g['vm_id']!=b['ident'] or g['disk_id']!=b['disk']
            or (lease and g['authorization_sha256']!=lease['authorization_sha256'])
            or not time.time()<min(g['expires_unix'],g['stop_unix'])):
        raise RuntimeError('Independent provider/cost guard is absent or expired')


class ArithmeticWorker:
    def __init__(self, delay=8):
        self.delay = delay
        self.tasks = {}
        self.events = []
        self.completed = []

    def event(self, kind, **details):
        self.events.append(dict(type=kind, monotonic=time.monotonic(), **details))

    def submit(self, query):
        ident = str(uuid.uuid4())
        self.event('native_delegate', work_id=ident, query=query)
        if re.search(r'\b(cancel|stop|abort)\b|never mind', query, re.I):
            pending = [k for k, t in self.tasks.items() if not t.done()]
            for k in pending:
                self.tasks[k].cancel()
            self.event('cancel_requested', work_id=ident, targets=pending)
            self.tasks[ident] = asyncio.create_task(self.confirm_cancel(ident, pending))
            return ident
        self.tasks[ident] = asyncio.create_task(self.execute(ident, query))
        return ident

    async def confirm_cancel(self, ident, targets):
        await asyncio.gather(*(self.tasks[k] for k in targets), return_exceptions=True)
        self.event('cancellation_confirmed', work_id=ident, targets=targets)
        self.completed.append(dict(work_id=ident, cancelled_work_ids=targets,
            result='The pending work has been cancelled.' if targets else 'No work was pending.'))

    async def execute(self, ident, query):
        self.event('worker_started', work_id=ident)
        try:
            await asyncio.sleep(self.delay)
            # Read-only, bounded deterministic capability, never eval()/shell.
            # In this small capability commas separate input numbers. Thousands
            # grouping/word-number interpretation needs the real reasoning backend.
            numbers = [int(x) for x in re.findall(r'[+-]?\d+', query)]
            if len(numbers) > 20 or any(abs(x) > 10**9 for x in numbers):
                result = 'Arithmetic request exceeds this diagnostic worker limit.'
            elif numbers and re.search(r'sum|total|add|plus', query, re.I):
                result = f'Computed sum: {sum(numbers)}. Input numbers: {numbers}.'
            else:
                result = 'This diagnostic worker only computes sums of explicitly supplied numbers.'
            receipt = dict(work_id=ident, query=query, result=result)
            receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
            self.completed.append(receipt)
            self.event('worker_completed', **receipt)
        except asyncio.CancelledError:
            self.event('worker_cancelled', work_id=ident)
            raise

    def take_results(self):
        results, self.completed = self.completed, []
        if not results:
            return None
        return '<backend>' + json.dumps(results, ensure_ascii=False) + '</backend>'

    async def close(self):
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)


class DelegateParser:
    """Consume raw generated token text incrementally, without editing speech."""
    def __init__(self):
        self.buffer = ''

    def feed(self, delta):
        self.buffer += delta
        if len(self.buffer) > 16384:
            raise ValueError('Native delegate stream exceeds bounded parser buffer')
        queries = []
        while '</delegate>' in self.buffer:
            end = self.buffer.index('</delegate>')
            start = self.buffer.rfind('<delegate>', 0, end)
            if start >= 0:
                queries.append(self.buffer[start + len('<delegate>'):end])
            self.buffer = self.buffer[end + len('</delegate>'):]
        if '<delegate>' not in self.buffer:
            self.buffer = self.buffer[-len('<delegate>'):]
        return queries


class NativeEngine:
    def __init__(self, model_dir):
        ensure_guard()
        import numpy as np
        import torch
        from transformers import AutoModel, set_seed
        self.np, self.torch = np, torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one CUDA GPU required; never run this on the Mac CPU')
        set_seed(42)
        self.model_dir = Path(model_dir).resolve()
        self.model = AutoModel.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16,
            init_vision=False, init_audio=True, init_tts=True,
        ).eval().cuda()
        self.model.requires_grad_(False)
        self.duplex = self.model.as_duplex(generate_audio=True)
        self.profile = dict(model='Realtime-Venus-Audio', seed=42,
            torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), precision='bfloat16', attention='sdpa',
            input_sample_rate=16000, output_sample_rate=24000,
            system_prompt='Streaming Omni Conversation.', decode_mode='sampling',
            max_new_speak_tokens_per_chunk=20, temperature=0.7, top_k=20,
            top_p=0.8, listen_prob_scale=1.0, optimizer_updates=0)
        self.profile.update(force_listen_count=int(self.duplex.force_listen_count),
                            listen_speak_mode=str(self.duplex.ls_mode))

    def prepare(self):
        ensure_guard()
        with self.torch.inference_mode():
            self.duplex.prepare(prefix_system_prompt=self.profile['system_prompt'],
                prompt_wav_path=str(self.model_dir / 'assets/HT_ref_audio.wav'))
        self.cursor = 0

    def step(self, pcm, feedback):
        ensure_guard()
        np, torch = self.np, self.torch
        wave = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
        if len(wave) != 16000:
            raise ValueError('Exactly one second of16kHz mono PCM16 required')
        started = time.monotonic()
        with torch.inference_mode():
            prefill = self.duplex.streaming_prefill(audio_waveform=wave,
                text_list=[feedback] if feedback else None)
            if not prefill.get('success'):
                raise RuntimeError('Native prefill failed: ' + str(prefill.get('reason')))
            result = self.duplex.streaming_generate(**{k:self.profile[k] for k in
                ('decode_mode','max_new_speak_tokens_per_chunk','temperature','top_k','top_p','listen_prob_scale')})
        ids = self.duplex.total_ids[self.cursor:]
        self.cursor = len(self.duplex.total_ids)
        raw_text = self.duplex.tokenizer.decode(ids, skip_special_tokens=False)
        audio = result.get('audio_waveform')
        if audio is None:
            audio = np.zeros(0, dtype=np.float32)
        elif hasattr(audio, 'detach'):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not np.isfinite(audio).all():
            raise RuntimeError('Nonfinite native waveform; no clipping/normalization repair')
        return dict(text=result.get('text',''), raw_text=raw_text,
            token_ids=[int(x) for x in ids], is_listen=bool(result.get('is_listen')),
            end_of_turn=bool(result.get('end_of_turn')),
            elapsed=time.monotonic()-started, audio_samples=len(audio),
            peak=float(np.max(np.abs(audio))) if len(audio) else 0,
            audio=base64.b64encode(audio.astype('<f4').tobytes()).decode(),
            audio_sample_rate=24000)


def create_app(model_dir):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse
    from starlette.websockets import WebSocketDisconnect

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await asyncio.to_thread(NativeEngine, model_dir)
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    slot = asyncio.Lock()

    @app.get('/')
    async def index():
        return FileResponse(Path(__file__).with_name('app.html'))

    @app.get('/health')
    async def health():
        return dict(ready=True, busy=slot.locked(), **app.state.engine.profile)

    @app.websocket('/ws')
    async def conversation(ws: WebSocket):
        origin = ws.headers.get('origin')
        from urllib.parse import urlsplit
        if origin and urlsplit(origin).netloc != ws.headers.get('host'):
            await ws.close(code=1008)
            return
        await ws.accept()
        if slot.locked():
            await ws.send_json({'error':'Another conversation is active.'})
            await ws.close()
            return
        async with slot:
            worker, parser = ArithmeticWorker(), DelegateParser()
            try:
                engine = app.state.engine
                await asyncio.to_thread(engine.prepare)
                await ws.send_json({'ready':True, 'profile':engine.profile})
                sequence = 0
                while True:
                    pcm = await asyncio.wait_for(ws.receive_bytes(), timeout=45)
                    feedback = worker.take_results()
                    if feedback:
                        worker.event('backend_injected', payload=feedback, sequence=sequence)
                    result = await asyncio.to_thread(engine.step, pcm, feedback)
                    for query in parser.feed(result['raw_text']):
                        worker.submit(query)
                    result.update(sequence=sequence, events=worker.events)
                    worker.events = []
                    sequence += 1
                    await ws.send_json(result)
            except WebSocketDisconnect:
                pass
            except Exception as error:
                try:
                    await ws.send_json({'error':str(error),'type':type(error).__name__})
                    await ws.close()
                except RuntimeError:
                    pass
            finally:
                await worker.close()
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--port', type=int, default=8032)
    args = parser.parse_args()
    import uvicorn
    # Loopback and SSH tunnel only. No microphone or conversation recording.
    uvicorn.run(create_app(args.model_dir), host='127.0.0.1', port=args.port)
