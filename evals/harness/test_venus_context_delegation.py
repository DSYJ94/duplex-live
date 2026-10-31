import asyncio
import json
import threading
import time
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from portable.venus import native_app
from portable.venus.reasoning import ReasoningWorker
from portable.venus.task_context import AudioEvidence, CausalAudioBuffer
from evals.harness.test_venus_reasoning import reply


def test_streamed_request_freezes_at_opening_before_close_and_multiple_requests():
    b=CausalAudioBuffer();p=native_app.DelegateParser(b.freeze_at_request_opening)
    b.append(0,b'a'*32000)
    assert p.feed('<unit><delegate>look up')==[]
    b.append(1,b'b'*32000)
    assert p.feed(' cupboard</delegate></unit><delegate>second</delegate>')==['look up cupboard','second']
    first,second=p.last_contexts
    assert first.pcm16==b'a'*32000 and second.pcm16==b'a'*32000+b'b'*32000
    assert p.last_raw_queries==['look up cupboard','second']


def test_context_transport_cannot_silently_discard_or_forge_audio():
    with pytest.raises(ValueError):AudioEvidence(0,1,b'a'*32000)
    with pytest.raises(ValueError):AudioEvidence(0,0,bytearray(32000))
    w=ReasoningWorker(None,evidence_request=lambda *_:None)
    with pytest.raises(ValueError):w.submit('request without context')
    with pytest.raises(ValueError):ReasoningWorker(None).submit('request',audio_evidence=AudioEvidence(0,0,bytes(32000)))


def test_real_websocket_keeps_processing_audio_while_contextual_backend_is_pending(monkeypatch):
    gate=threading.Event();entered=threading.Event();captured=[];workers=[];feedback=[]
    async def contextual(value,evidence):
        captured.append((json.loads(value['input']),evidence));entered.set()
        await asyncio.to_thread(gate.wait,5)
        return reply('context-fixture-response','A synthetic answer.')
    async def forbidden(_):raise AssertionError('Description-only transport must not be used')
    def factory():
        w=ReasoningWorker(forbidden,evidence_request=contextual);workers.append(w);return w
    class LLM:
        def modules(self):return []
    class Engine:
        def __init__(self,*args):self.model=SimpleNamespace(llm=LLM());self.profile=dict(model='fixture-only',optimizer_updates=0)
        def prepare(self):self.n=0
        def step(self,pcm,backend):
            if backend:feedback.append(backend)
            text={0:'<delegate>look up',1:' cupboard</delegate>'}.get(self.n,'');self.n+=1
            return dict(raw_text=text,text='',is_listen=True,audio='')
    monkeypatch.setattr(native_app,'NativeEngine',Engine)
    try:
        with TestClient(native_app.create_app('fixture',worker_factory=factory)) as c:
            with c.websocket_connect('/ws',headers={'origin':'http://testserver'}) as ws:
                ready=ws.receive_json()
                assert ready['ready'] and ready['profile']['causal_task_audio_context']
                ws.send_bytes(b'a'*32000);ws.receive_json()
                ws.send_bytes(b'b'*32000);ws.receive_json();assert entered.wait(2)
                ws.send_bytes(b'c'*32000);assert ws.receive_json()['sequence']==2
                source,evidence=captured[0]
                assert source['native_selected_task']=='look up cupboard'
                assert evidence.pcm16==b'a'*32000 and evidence.last_sequence==0
                assert not feedback
                gate.set()
                for _ in range(30):
                    ws.send_bytes(b'd'*32000);ws.receive_json()
                    if feedback:break
                    time.sleep(.01)
                assert len(feedback)==1 and 'A synthetic answer.' in feedback[0]
                state=next(iter(workers[0].states.values()))
                assert state['audio_evidence_receipt']==evidence.receipt()
                assert state['task_input_sha256'] in feedback[0]
                assert 'audio_evidence' not in state
                ws.send_bytes(b'e'*32000);ws.receive_json();assert len(feedback)==1
    finally:gate.set()


@pytest.mark.anyio
async def test_disconnect_before_execution_clears_evidence_and_suppresses_transport():
    called=[]
    async def contextual(*args):called.append(args);return reply('unused','Unused.')
    w=ReasoningWorker(None,evidence_request=contextual)
    ident=w.submit('task',audio_evidence=AudioEvidence(0,0,bytes(32000)))
    await w.close();await w.tasks[ident]
    assert not called and w.take_results() is None and 'audio_evidence' not in w.states[ident]
