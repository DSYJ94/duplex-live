"""Native Venus microphone prototype and synthetic-only diagnostic endpoint.

The released streaming model owns speech/listen decisions. No keyword mute, ASR
controller, canned ordinary answers or forced speech. The default arithmetic
diagnostic or opt-in Mac reasoning proxy handles ONLY model-emitted requests.
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
from contextlib import ExitStack


def ensure_guard():
    g=json.loads(Path(os.environ['DUPLEX_LIVE_GUARD_FILE']).read_text())
    lease=json.loads(Path(os.environ['DUPLEX_LIVE_LEASE_FILE']).read_text()) if os.environ.get('DUPLEX_LIVE_LEASE_FILE') else {}
    if lease.get('provider')=='Slurm':
        from training.venus.isambard_guard import ensure
        ensure(lease,g)
        return
    from .guard import resource
    b=resource(lease)
    if (g['vm_id']!=b['ident'] or g['disk_id']!=b['disk']
            or (lease and g['authorization_sha256']!=lease['authorization_sha256'])
            or not time.time()<min(g['expires_unix'],g['stop_unix'])):
        raise RuntimeError('Independent provider/cost guard is absent or expired')


def explicit_sum_numbers(query):
    """Bounded integers, including comma/plus-separated English forms below1000.

    Ambiguous word sequences fail closed; no model, eval, or shell interpretation.
    """
    if '{' in query or '}' in query:
        raise ValueError('Structured requests need an explicit argument schema')
    small=dict(zip(('zero one two three four five six seven eight nine ten eleven twelve '
        'thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split(),range(20)))
    tens=dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10)))
    words=set(small)|set(tens)|{'hundred','and'}
    if set(re.findall(r'[a-z]+',query.lower()))&{'thousand','million','billion','minus','negative'}:
        raise ValueError('Unsupported scale or signed word form needs clarification')
    if re.search(r'\d',query):
        if re.search(r'\d\.\d',query) or any(w in small or w in tens or w=='hundred' for w in re.findall(r'[a-z]+',query.lower())):
            raise ValueError('Mixed or fractional numeric form needs clarification')
        return [int(x) for x in re.findall(r'[+-]?\d+',query)]
    numbers=[]
    for part in re.split(r'[,;]|\bplus\b',query.lower().replace('-',' ')):
        tokens=re.findall(r'[a-z]+',part)
        indices=[i for i,w in enumerate(tokens) if w in small or w in tens]
        if not indices:continue
        first=indices[0];end=first
        while end<len(tokens) and tokens[end] in words:end+=1
        if indices[-1]>=end:raise ValueError('Separate integer groups require explicit delimiters')
        seq=tokens[first:end]
        if any(w not in words for w in seq):raise ValueError('Non-numeric words inside an integer')
        seq=[w for w in seq if w!='and'];value=0
        if len(seq)>=2 and seq[0] in small and 1<=small[seq[0]]<=9 and seq[1]=='hundred':
            value=small[seq[0]]*100;seq=seq[2:]
        if seq:
            if seq[0] in tens:
                value+=tens[seq[0]];seq=seq[1:]
                if seq and seq[0] in small and 0<small[seq[0]]<10:value+=small[seq.pop(0)]
            elif seq[0] in small:value+=small[seq.pop(0)]
            else:raise ValueError('Unsupported English integer')
        if seq:raise ValueError('Ambiguous English integer sequence')
        numbers.append(value)
    return numbers


class ArithmeticWorker:
    def __init__(self, delay=8):
        self.delay = delay
        self.tasks = {}
        self.events = []
        self.completed = []

    def event(self, kind, **details):
        self.events.append(dict(type=kind, monotonic=time.monotonic(), **details))

    def submit(self, query, raw_query=None):
        ident = str(uuid.uuid4())
        self.event('native_delegate', work_id=ident, query=query,
            raw_query=query if raw_query is None else raw_query)
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
            try:numbers=explicit_sum_numbers(query)
            except ValueError:numbers=None
            if numbers is None:
                result = 'Arithmetic request is ambiguous; provide comma-separated integers.'
            elif len(numbers) > 20 or any(abs(x) > 10**9 for x in numbers):
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
    def __init__(self, capture_opening=None):
        self.buffer = ''
        self.last_raw_queries = []
        self.capture_opening = capture_opening
        self.last_contexts = []
        self._pending_start = None
        self._pending_context = None

    def feed(self, delta):
        self.last_raw_queries = []
        self.last_contexts = []
        self.buffer += delta
        if len(self.buffer) > 16384:
            raise ValueError('Native delegate stream exceeds bounded parser buffer')
        queries = []
        while '</delegate>' in self.buffer:
            end = self.buffer.index('</delegate>')
            start = self.buffer.rfind('<delegate>', 0, end)
            if start >= 0:
                raw=self.buffer[start + len('<delegate>'):end]
                # Native unit framing is transport, not worker-request text.
                # Remove only known frame tokens; keep original bytes in trace/events.
                query=re.sub(r'</?unit>|<\|(?:speak|listen|turn_eos|chunk_eos|chunk_tts_eos|tts_bos|tts_eos)\|>','',raw)
                self.last_raw_queries.append(raw)
                queries.append(query)
                context = (self._pending_context if start == self._pending_start
                           else self.capture_opening() if self.capture_opening else None)
                self.last_contexts.append(context)
            self.buffer = self.buffer[end + len('</delegate>'):]
            self._pending_start = self._pending_context = None
        if '<delegate>' not in self.buffer:
            self.buffer = self.buffer[-len('<delegate>'):]
            self._pending_start = self._pending_context = None
        elif self.capture_opening:
            start = self.buffer.rfind('<delegate>')
            if self._pending_start != start:
                self._pending_start, self._pending_context = start, self.capture_opening()
        return queries


class NativeEngine:
    def __init__(self, model_dir, system_prompt='Streaming Omni Conversation.', adapter_checkpoint=None,
                 voice_reference=None,voice_reference_sha256=None,nonspoken_actions=False,tts_attention='released',speech_completion=False):
        ensure_guard()
        if speech_completion and nonspoken_actions:
            from .speech_composition import require_authorization
            require_authorization(voice_reference,voice_reference_sha256)
        elif speech_completion:
            from .speech_completion import require_authorization
            require_authorization()
        import numpy as np
        import torch
        from .wav_io import install as install_wav_io
        install_wav_io()
        from transformers import AutoModel, set_seed
        self.np, self.torch = np, torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one CUDA GPU required; never run this on the Mac CPU')
        set_seed(42)
        self.model_dir = Path(model_dir).resolve()
        from .voice import reference
        self.voice_reference=reference(self.model_dir,voice_reference,voice_reference_sha256)
        self.model = AutoModel.from_pretrained(
            str(self.model_dir), trust_remote_code=True, local_files_only=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16,
            init_vision=False, init_audio=True, init_tts=True,
        ).eval().cuda()
        self.model.requires_grad_(False)
        from .tts_attention import configure_tts_attention
        tts_backend=configure_tts_attention(self.model,tts_attention)
        candidate = None
        if adapter_checkpoint:
            from training.venus.inference import load_candidate
            candidate = load_candidate(self.model.llm, adapter_checkpoint)
        self.duplex = self.model.as_duplex(generate_audio=True)
        self.speech_completion=None
        if speech_completion:
            from .speech_completion import install as install_completion
            self.speech_completion=install_completion(self.duplex)
        self.action_channel=None
        if nonspoken_actions:
            from .speech_channel import install
            self.action_channel=install(self.duplex)
        self.profile = dict(model='Realtime-Venus-Audio', seed=42,
            torch=torch.__version__, cuda=torch.version.cuda,tts_attention=tts_backend,
            gpu=torch.cuda.get_device_name(), precision='bfloat16', attention='sdpa',
            input_sample_rate=16000, output_sample_rate=24000,
            system_prompt=system_prompt, decode_mode='sampling',
            max_new_speak_tokens_per_chunk=20, temperature=0.7, top_k=20,
            top_p=0.8, listen_prob_scale=1.0, optimizer_updates=0,
            voice_reference=self.voice_reference,nonspoken_action_channel=bool(self.action_channel),
            speech_completion=dict(enabled=bool(self.speech_completion),version='native-terminal-resume-v1' if self.speech_completion else None,maximum_terminal_tokens=200 if self.speech_completion else None,qualified=False))
        self.profile.update(force_listen_count=int(self.duplex.force_listen_count),
                            listen_speak_mode=str(self.duplex.ls_mode))
        if candidate:
            self.profile.update(optimizer_updates=candidate['optimizer_updates'], candidate=candidate)

    def prepare(self):
        ensure_guard()
        if self.action_channel:self.action_channel.reset()
        with self.torch.inference_mode():
            # duplex.prepare resets text/processor state, but the released
            # model's audio KV cache otherwise survives across conversations.
            self.model.reset_session()
            self.duplex.prepare(prefix_system_prompt=self.profile['system_prompt'],
                prompt_wav_path=self.voice_reference['path'])
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
            prefilled_at=time.monotonic()
            if not prefill.get('success'):
                raise RuntimeError('Native prefill failed: ' + str(prefill.get('reason')))
            result = self.duplex.streaming_generate(**{k:self.profile[k] for k in
                ('decode_mode','max_new_speak_tokens_per_chunk','temperature','top_k','top_p','listen_prob_scale')})
        generated_at=time.monotonic()
        from .timing import stage_timings
        timings=stage_timings(result,prefill_seconds=prefilled_at-started,generation_seconds=generated_at-prefilled_at)
        # Preserve released-model counts as observations. Unknown counts stay
        # unknown; token counts alone do not prove spoken completeness or EOS.
        native_counts={name:(result.get(key) if type(result.get(key)) is int and result[key]>=0 else None)
                       for name,key in (('text_tokens','n_tokens'),('audio_tokens','n_tts_tokens'))}
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
        omit_silence=bool(result.get('is_listen')) and not np.any(audio)
        wire_audio=b'' if omit_silence else audio.astype('<f4').tobytes()
        exported=dict(text=result.get('text',''), raw_text=raw_text,
            token_ids=[int(x) for x in ids], is_listen=bool(result.get('is_listen')),
            end_of_turn=bool(result.get('end_of_turn')),
            elapsed=time.monotonic()-started, audio_samples=len(audio),
            native_generation_counts=native_counts,
            peak=float(np.max(np.abs(audio))) if len(audio) else 0,
            audio=base64.b64encode(wire_audio).decode(),wire_silence_omitted=omit_silence,wire_audio_bytes=len(wire_audio),timings=timings,
            audio_sample_rate=24000)
        if 'speech_completion' in result:exported['speech_completion']=result['speech_completion']
        return exported


def create_app(model_dir, system_prompt='Streaming Omni Conversation.', adapter_checkpoint=None,
               voice_reference=None,voice_reference_sha256=None,nonspoken_actions=False,compare_foundation=False,tts_attention='released',worker_factory=None,
               reasoning_context=None,speech_completion=False):
    if compare_foundation and not adapter_checkpoint:
        raise ValueError('A comparison requires the explicit trained checkpoint')
    if reasoning_context is not None:
        if worker_factory is not None or not isinstance(reasoning_context,tuple) or len(reasoning_context)!=2:
            raise ValueError('One explicit contextual reasoning endpoint/identity required')
        from .context_reasoning import ContextProxy
        context_proxy=ContextProxy(*reasoning_context)
    # A contextual ASR pass and native streaming share one GPU. Serialize their
    # execution, keeping provider waits off this lock and off the streaming loop.
    import threading
    gpu_lock=threading.Lock()
    def locked_call(fn,*args):
        with gpu_lock:return fn(*args)
    async def engine_call(fn,*args):
        if reasoning_context is not None:return await asyncio.to_thread(locked_call,fn,*args)
        return await asyncio.to_thread(fn,*args)
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse
    from starlette.websockets import WebSocketDisconnect

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await asyncio.to_thread(NativeEngine, model_dir, system_prompt, adapter_checkpoint,
                                                  voice_reference,voice_reference_sha256,nonspoken_actions,tts_attention,speech_completion)
        if reasoning_context is not None:
            from .context_asr import GPUContextASR
            from .context_reasoning import EvidenceTransport
            from .reasoning import ReasoningWorker
            recognizer=await asyncio.to_thread(GPUContextASR)
            async def transcribe(evidence):return await engine_call(recognizer,evidence)
            app.state.context_worker_factory=lambda:ReasoningWorker(None,
                evidence_request=EvidenceTransport(transcribe,context_proxy))
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    slot = asyncio.Lock()
    worker_backend = ('mac-context-reasoning-proxy' if reasoning_context is not None else
                      'mac-reasoning-proxy' if worker_factory else 'arithmetic-diagnostic')

    @app.get('/')
    async def index():
        return FileResponse(Path(__file__).with_name('app.html'))

    @app.get('/health')
    async def health():
        return dict(ready=True, busy=slot.locked(), comparison_available=compare_foundation,
                    worker_backend=worker_backend, **app.state.engine.profile)

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
            factory=getattr(app.state,'context_worker_factory',worker_factory)
            worker = factory() if factory else ArithmeticWorker()
            from .task_context import CausalAudioBuffer
            evidence = CausalAudioBuffer() if getattr(worker,'accepts_audio_evidence',False) else None
            parser = DelegateParser(evidence.freeze_at_request_opening if evidence else None)
            variants = ExitStack()
            try:
                engine = app.state.engine
                variant=ws.query_params.get('variant','candidate')
                if variant!='candidate' and not compare_foundation:
                    raise ValueError('Model comparison is not enabled on this server')
                from .live_variant import variant_context,variant_profile
                variants.enter_context(variant_context(engine.model.llm,variant))
                await engine_call(engine.prepare)
                await ws.send_json({'ready':True, 'profile':dict(variant_profile(engine.profile,variant),worker_backend=worker_backend,causal_task_audio_context=bool(evidence))})
                sequence = 0
                while True:
                    pcm = await asyncio.wait_for(ws.receive_bytes(), timeout=45)
                    if evidence: evidence.append(sequence,pcm)
                    feedback = worker.take_results()
                    if feedback:
                        worker.event('backend_injected', payload=feedback, sequence=sequence)
                    result = await engine_call(engine.step, pcm, feedback)
                    queries=parser.feed(result['raw_text'])
                    for query,raw_query,context in zip(queries,parser.last_raw_queries,parser.last_contexts):
                        if evidence: worker.submit(query,raw_query=raw_query,audio_evidence=context)
                        else: worker.submit(query,raw_query=raw_query)
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
                try:await worker.close()
                finally:variants.close()
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--port', type=int, default=8032)
    parser.add_argument('--system-prompt-file')
    parser.add_argument('--adapter-checkpoint', help='Verified project checkpoint; experimental native testing only')
    parser.add_argument('--voice-reference', help='Approved original/synthetic mono16k reference for the native voice generator')
    parser.add_argument('--voice-reference-sha256', help='Exact SHA256 required for a custom reference')
    parser.add_argument('--speech-completion',action='store_true',help='Experimental bounded terminal acoustic continuation; requires paired native speech/control and latency qualification')
    parser.add_argument('--nonspoken-actions',action='store_true',help='Experimental native protocol filter; full speech/behavior qualification still required')
    parser.add_argument('--compare-foundation',action='store_true',help='Allow original/adapted comparison between isolated conversations')
    parser.add_argument('--tts-attention',choices=['released','sdpa'],default='released',help='Experimental TTS-only backend; native speed and audio validation required')
    parser.add_argument('--reasoning-proxy', help='Opt-in loopback Mac-only proxy URL over existing SSH; provider credentials never go on GPU')
    parser.add_argument('--reasoning-run-id', help='Explicit matching bounded proxy execution identity')
    parser.add_argument('--reasoning-audio-context',action='store_true',help='Experimental pinned local GPU ASR for native-selected tasks; requires contextual Mac proxy and GPU latency qualification')
    args = parser.parse_args()
    import uvicorn
    # Loopback and SSH tunnel only. No microphone or conversation recording.
    prompt=Path(args.system_prompt_file).read_text().strip() if args.system_prompt_file else 'Streaming Omni Conversation.'
    if not prompt or len(prompt)>4000:raise ValueError('Bounded nonempty system prompt required')
    if bool(args.reasoning_proxy)!=bool(args.reasoning_run_id):parser.error('Both reasoning proxy and execution identity are required')
    if args.reasoning_audio_context and not args.reasoning_proxy:parser.error('Audio context requires the explicit Mac reasoning proxy and identity')
    worker_factory=None
    reasoning_context=(args.reasoning_proxy,args.reasoning_run_id) if args.reasoning_audio_context else None
    if args.reasoning_proxy and not args.reasoning_audio_context:
        from .reasoning import LocalProxy,ReasoningWorker
        proxy=LocalProxy(args.reasoning_proxy,args.reasoning_run_id)
        worker_factory=lambda:ReasoningWorker(proxy)
    uvicorn.run(create_app(args.model_dir,prompt,args.adapter_checkpoint,args.voice_reference,args.voice_reference_sha256,args.nonspoken_actions,args.compare_foundation,args.tts_attention,worker_factory,reasoning_context,args.speech_completion), host='127.0.0.1', port=args.port)
