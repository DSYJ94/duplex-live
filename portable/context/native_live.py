"""Experimental live native-RET delegation; qualification remains separate.

Only the speech model's actual RET marker dispatches work. ASR and the existing
semantic worker run off the frame loop. Results are references, never forced
words/audio. Session retirement rejects late results without claiming that the
provider's remote computation was aborted. Conversation state stays in memory.
"""
import asyncio
import uuid

from evals.recovery_worker import RecordWorld, WorkerContractError, WorkerSuperseded, run_worker


class NativeLiveSession:
    def __init__(self, engine, transcribe, records=None, worker=run_worker, maximum_requests=6):
        if type(maximum_requests) is not int or not 1 <= maximum_requests <= 6:
            raise ValueError('One live session permits at most six native requests')
        self.engine, self.transcribe, self.worker = engine, transcribe, worker
        self.world = RecordWorld({} if records is None else records)
        self.maximum_requests = maximum_requests
        self.identity = uuid.uuid4().hex
        self.audio, self.tokens, self.pending = [], [], {}
        self.closed = False
        self.requests = self.delivered = self.rejected = 0

    async def _work(self, identity, revision, pcm, seconds, assistant):
        from evals.context_spanning_asr import request_context
        from evals.context_backend_failure import notice
        recognized = await asyncio.to_thread(self.transcribe, pcm, 24000)
        if self.closed:
            return None
        context, rejected = request_context(recognized, seconds, assistant, seconds, identity)
        if rejected is not None:
            return rejected
        with self.world.lock:
            self.world.tasks[identity]['context'] = context
        try:
            result = await asyncio.to_thread(self.worker, context, self.world, identity,
                                            expected_revision=revision, dispatch_protocol='state-aware',
                                            scope_read_tools=True)
        except WorkerContractError as error:
            result = notice(error, identity)
        except WorkerSuperseded:
            return None
        if self.closed:
            return None
        if result['route'] != 'error':
            for receipt in result['receipts']:
                if receipt['tool'] == 'lookup_record':
                    self.world.mark_ready(receipt)
        return result

    def _drain(self):
        from evals.context_backend_reference import reference
        events = []
        for identity, task in list(self.pending.items()):
            if not task.done():
                continue
            del self.pending[identity]
            try:
                result = task.result()
            except (WorkerSuperseded, asyncio.CancelledError):
                result = None
            except Exception as error:
                result = dict(route='error', receipts=[], task_id=identity,
                              error_type=type(error).__name__,
                              content='The background request failed. No verified answer is available. Ask the user whether to try again.')
            state = self.world.tasks[identity]
            if self.closed or result is None or state['cancelled']:
                self.rejected += 1
                events.append(dict(type='backend_retired', task_id=identity))
                continue
            text, _ = reference(result, self.world)
            self.engine.inject_context_span(text)
            state['phase'] = 'failed' if result['route'] == 'error' else 'delivered'
            self.delivered += 1
            events.append(dict(type='backend_result', task_id=identity,
                               route=result['route'], result_injected=True,
                               error_type=result.get('error_type')))
        return events

    def step(self, frame):
        """Process every actual arrived frame; never await ASR or a provider."""
        import numpy as np
        if self.closed:
            raise RuntimeError('Conversation is closed')
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.engine.frame_size,) or not np.isfinite(frame).all():
            raise ValueError('Exactly one finite native audio frame required')
        if (len(self.audio)+1)*self.engine.frame_size > 60*24000:
            raise ValueError('Experimental live sessions are bounded to sixty seconds')
        events = self._drain()
        self.audio.append(frame.copy())
        result = self.engine.step(frame)
        if result['text_token'] is not None:
            self.tokens.append(result['text_token'])
        if result['is_ret']:
            if self.requests >= self.maximum_requests:
                raise RuntimeError('Native model exceeded the bounded request budget')
            self.requests += 1
            identity = self.identity+'-ret-'+str(self.requests)
            revision = self.world.reserve(identity)
            task = asyncio.create_task(self._work(identity, revision, np.concatenate(self.audio),
                len(self.audio)*self.engine.frame_size/24000, self.engine.decode_text(self.tokens)))
            # Consume failures even after the owning websocket closes; never
            # deliver a reference to the next conversation sharing this engine.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            self.pending[identity] = task
            events.append(dict(type='ret', task_id=identity, native_trigger=True))
        return result, events

    def close(self):
        self.closed = True
        for identity, task in self.pending.items():
            with self.world.lock:
                state = self.world.tasks[identity]
                if not state['cancelled'] and not state.get('closed', False):
                    self.world.cancel(identity)
            task.cancel()


async def ws_handler(request):
    """Existing microphone wire protocol, with isolated per-connection work."""
    import numpy as np
    from aiohttp import web, WSMsgType
    from contextspan.model.sequence_convention import persona_text
    ws = web.WebSocketResponse(max_msg_size=1920*4)
    await ws.prepare(request)
    app = request.app
    if app['lock'].locked():
        await ws.send_json(dict(type='busy')); await ws.close(); return ws
    async with app['lock']:
        engine = app['engine']
        session = NativeLiveSession(engine, app['asr'].transcribe)
        engine.reset(); engine.set_persona(persona_text(app['persona']), app['voice'])
        shown = ''
        try:
            await ws.send_json(dict(type='ready', sample_rate=24000, frame_size=engine.frame_size,
                                   external_backend_enabled=True, actual_live_delegation_verified=False,
                                   session_maximum_seconds=60))
            async for message in ws:
                if message.type != WSMsgType.BINARY:
                    continue
                frame = np.frombuffer(message.data, dtype=np.float32)
                result, events = session.step(frame)
                if result['agent_pcm'] is not None:
                    pcm = np.asarray(result['agent_pcm'], dtype=np.float32)
                    if pcm.shape != (engine.frame_size,) or not np.isfinite(pcm).all():
                        raise RuntimeError('Invalid sampled native audio')
                    await ws.send_bytes(pcm.tobytes())
                for event in events:
                    await ws.send_json(event)
                full = engine.decode_text(session.tokens)
                if full != shown:
                    await ws.send_json(dict(type='text', delta=full[len(shown):])); shown = full
        except Exception:
            # No user words, audio, provider payload or credentials in the log.
            await ws.send_json(dict(type='error', message='The bounded native session ended; reconnect to start a fresh session.'))
        finally:
            session.close()
            await ws.close()
    return ws
