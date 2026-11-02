"""Actual WebSocket handler with fixture ASR/HTTP transport, never model proof."""
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from portable.venus import native_app, context_asr
from portable.venus.context_reasoning import ContextProxy, validate_contextual_payload
from evals.harness.test_venus_reasoning import reply


def test_opt_in_live_route_preserves_opening_audio_and_streams_during_provider_wait(monkeypatch):
    recognized, sent, feedback = [], [], []
    pending, release = threading.Event(), threading.Event()
    asr_started, asr_release = threading.Event(), threading.Event()
    executing = threading.Lock()

    class Engine:
        def __init__(self, *args):
            self.model = SimpleNamespace(llm=None)
            self.profile = dict(model='fixture-only', optimizer_updates=0)
        def prepare(self): self.n = 0
        def step(self, pcm, backend):
            assert executing.acquire(blocking=False), 'Native inference overlapped GPU ASR'
            try:
                if backend: feedback.append(backend)
                text = {0:'<delegate>Compare', 1:' these options</delegate>'}.get(self.n, '')
                self.n += 1
                return dict(raw_text=text, text='', is_listen=True, audio='')
            finally: executing.release()

    class ASR:
        def __call__(self, evidence):
            assert executing.acquire(blocking=False), 'GPU ASR overlapped native inference'
            try:
                recognized.append(evidence)
                asr_started.set()
                assert asr_release.wait(2)
                return dict(text='Compare the two loan terms I just gave.',
                    model=context_asr.MODEL, revision=context_asr.REVISION,
                    audio_receipt=evidence.receipt())
            finally: executing.release()

    def post(self, value):
        validate_contextual_payload(value)
        sent.append(value); pending.set()
        assert release.wait(3)
        return reply('fixture-context-route', 'Fixture comparison result.')

    monkeypatch.setattr(native_app, 'NativeEngine', Engine)
    monkeypatch.setattr(context_asr, 'GPUContextASR', ASR)
    monkeypatch.setattr(ContextProxy, '_post', post)
    try:
        with TestClient(native_app.create_app('fixture', reasoning_context=(
                'http://127.0.0.1:9999/v1/responses', 'fixture-context'))) as client:
            assert client.get('/health').json()['worker_backend']=='mac-context-reasoning-proxy'
            with client.websocket_connect('/ws', headers={'origin':'http://testserver'}) as ws:
                assert ws.receive_json()['profile']['causal_task_audio_context']
                ws.send_bytes(b'a'*32000); ws.receive_json()
                ws.send_bytes(b'b'*32000); ws.receive_json()
                assert asr_started.wait(2)
                # Queue the next native step while ASR holds the GPU slot.
                # The same route must finish it without simultaneous inference.
                ws.send_bytes(b'c'*32000)
                asr_release.set()
                assert ws.receive_json()['sequence']==2
                assert pending.wait(2)
                # The snapshot was taken when the native request opened, before
                # its closing frame. Later microphone bytes cannot enter it.
                assert recognized[0].pcm16==b'a'*32000
                source=json.loads(sent[0]['input'])
                assert source['native_selected_task']=='Compare these options'
                assert source['recognized_audio']['audio_receipt']==recognized[0].receipt()
                ws.send_bytes(b'd'*32000)
                assert ws.receive_json()['sequence']==3 and not feedback
                release.set()
                for _ in range(30):
                    ws.send_bytes(b'd'*32000); ws.receive_json()
                    if feedback: break
                    time.sleep(.01)
                assert len(feedback)==1 and 'Fixture comparison result.' in feedback[0]
                ws.send_bytes(b'e'*32000); ws.receive_json()
                assert len(feedback)==1 and len(sent)==1 and len(recognized)==1
    finally: asr_release.set(); release.set()


def test_invalid_context_configuration_fails_before_model_initialization():
    for config in (('http://external.example/v1/responses','id'), ('http://127.0.0.1:9/v1/responses',)):
        with pytest.raises(ValueError): native_app.create_app('unused',reasoning_context=config)
    with pytest.raises(ValueError):
        native_app.create_app('unused', worker_factory=lambda:None,
            reasoning_context=('http://127.0.0.1:9/v1/responses','id'))


def test_asr_refuses_cpu_before_loading_weights(monkeypatch):
    import torch
    import transformers
    monkeypatch.setattr(native_app,'ensure_guard',lambda:None)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def forbidden(*args,**kwargs): raise AssertionError('Must not load weights on CPU')
    monkeypatch.setattr(transformers.AutoModelForSpeechSeq2Seq,'from_pretrained',forbidden)
    with pytest.raises(RuntimeError,match='no Mac CPU fallback'): context_asr.GPUContextASR()
