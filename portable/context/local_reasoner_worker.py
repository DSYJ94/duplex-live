"""Explicit pinned local model behind the existing actual tool/state contract.

Reuses the previously tested local response protocol and Qwen XML parser.
Formatting an internal response is not a provider call or a tool receipt.
One executor owns one loaded model; do not also call actor.answer concurrently.
"""
import threading
from evals.qwen_response_executor import QwenResponseExecutor
from evals.qwen35_controller import NonThinkingTokenizer,parse_action
from evals.qwen_controller_probe import Generator
from evals.recovery_worker import run_worker
from training.personaplex.local_reasoner_reference import LocalReasoner,PIN_SHA,REVISION


class Local27BExecutor(QwenResponseExecutor):
    provider_name='local-pinned-qwen36-27b'

    def __init__(self,actor):
        if not isinstance(actor,LocalReasoner)or actor.identity['revision']!=REVISION:raise ValueError('Explicit verified local model required; no implicit API/model fallback')
        self.actor=actor;self.model=actor.model;self.tokenizer=NonThinkingTokenizer(actor.processor.tokenizer)
        self.generator=Generator(self.model,self.tokenizer);self.model_id=actor.identity['repository']
        self.prompt_profile='base';self.pending_speech_protocol='array';self.attention_backend='math'
        self.spoken_output_contract=True;self.compact_replies=False
        self.lock=threading.RLock();self.history={};self.counter=0

    def parse_generated_action(self,text,definitions):return parse_action(text,definitions)

    def tool_history_arguments(self,arguments):return dict(arguments)

    def __call__(self,payload):
        # Parent provides inference_mode, one GPU/lock, exact schema checks,
        # actual tool-return history, raw generation and honest timing.
        with self.lock:
            self.actor.guard(60)
            if self.counter>=12:raise RuntimeError('Bounded native session response limit reached')
            result=super().__call__(payload);self.actor.guard()
            result['local_inference'].update(shared_public_pin_sha256=PIN_SHA,revision=REVISION,external_API_calls=0)
            return result

    def worker(self,context,world,identity,**kwargs):
        return run_worker(context,world,identity,model=self.model_id,response_executor=self,**kwargs)
