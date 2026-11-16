"""Explicit-backend native connection shared by diagnostics and serving.

No worker/API is chosen implicitly. Only an actual engine RET dispatches work;
the async frame boundary lets completed background work progress between frames.
Forced-token contract tests must never be reported as learned native behaviour.
"""
import asyncio
from .native_live import NativeLiveSession


class NativeConnection:
    def __init__(self,engine,transcribe,worker,*,records=None,maximum_seconds=60,context_provider=None,audit_sink=None):
        if not callable(transcribe)or not callable(worker):
            raise ValueError('Explicit local transcription and backend callables required')
        self.session=NativeLiveSession(engine,transcribe,records=records,worker=worker,
            maximum_seconds=maximum_seconds,context_provider=context_provider,audit_sink=audit_sink)

    @property
    def requests(self):return self.session.requests

    async def step(self,frame):
        result,events=self.session.step(frame)
        await asyncio.sleep(0)
        return result,events

    def close(self):self.session.close()
