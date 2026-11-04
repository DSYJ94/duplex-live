"""Opt-in terminal acoustic completion across native input frames.

The pinned native LLM still owns listen/speak/turn decisions. Only terminal TTS
sampling can remain pending after a frame. No text is rewritten or re-recognized.
Unqualified until paired native CUDA/audio/behavior and live-clock evaluation.
"""
import ast
import hashlib
import inspect
import textwrap
import time
from types import MethodType

SOURCE_HASHES=frozenset({'b7b4070cead9b88347f6e39d8333706ec007e2a832d6bcde5755063bf1453774'})
MAX_TERMINAL_TOKENS=200


def _source(method,name):
    from pathlib import Path
    if not hasattr(method,'__func__'):raise ValueError('Exact pinned native '+name+' required; wrapped composition is unqualified')
    raw=inspect.unwrap(method.__func__)
    if raw.__name__!=name or hashlib.sha256(Path(inspect.getfile(raw)).read_bytes()).hexdigest() not in SOURCE_HASHES:
        raise ValueError('Exact pinned native '+name+' required')
    return raw,ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(raw))))


def _compile(raw,source,extra=None):
    module=ast.parse(source);node=module.body[0];node.decorator_list=[]
    ns=dict(raw.__globals__);ns.update(extra or {})
    exec(compile(ast.fix_missing_locations(module),inspect.getfile(raw),'exec'),ns)
    return ns[node.name]


def _resumable(method):
    """Preserve native sampling math and its one-token lookahead at each yield."""
    raw,source=_source(method,'generate_chunk');module=ast.parse(source);node=module.body[0];node.decorator_list=[]
    loops=[n for n in node.body if isinstance(n,ast.For) and ast.unparse(n.target)=='t' and ast.unparse(n.iter)=='range(max_new_token)']
    assert len(loops)==1,'Pinned sampler loop changed'
    loop=loops[0];index=node.body.index(loop)
    node.body[index:index]=ast.parse('_completion_start = 0').body
    loop.body+=ast.parse('''if t > 0 and t % 25 == 0:
    yield new_tokens[:, _completion_start:t, :], past_key_values, False
    _completion_start = t
''').body
    tail=node.body.index(loop)+1
    assert len(node.body[tail:])==2 and isinstance(node.body[-1],ast.Return),'Pinned sampler return changed'
    node.body[tail:]=ast.parse('''if not bool(finish.all().item()):
    raise RuntimeError('Terminal acoustic EOS not reached within200 tokens; incomplete speech is not accepted')
yield new_tokens[:, _completion_start:t, :], past_key_values, True
''').body
    ns=dict(raw.__globals__)
    exec(compile(ast.fix_missing_locations(module),inspect.getfile(raw),'exec'),ns)
    return MethodType(ns[node.name],method.__self__)


def _duplex_generate(method,completion):
    raw,source=_source(method,'streaming_generate')
    edits=[
        ('tts_condition = self._convert_results_to_tts_input(total_hidden_in_unit)',
         '_completion.begin_native_speech(prompt_wav_path)\n    tts_condition = self._convert_results_to_tts_input(total_hidden_in_unit)'),
        ('new_tokens, old_kv = self.model.tts.generate_chunk(',
         'new_tokens, old_kv = _completion.sample(terminal=end_of_turn,'),
        ('if end_of_turn:\n        self.tts_text_start_pos = 0',
         'if end_of_turn and _completion.acoustic_complete:\n        self.tts_text_start_pos = 0'),
        ('new_tokens, prompt_wav_path, end_of_turn, force_flush=force_flush',
         'new_tokens, prompt_wav_path, end_of_turn and _completion.acoustic_complete, force_flush=force_flush'),
        ('if end_of_turn:\n        self._reset_token2wav_for_new_turn()',
         'if end_of_turn and _completion.acoustic_complete:\n        self._reset_token2wav_for_new_turn()'),
    ]
    for old,new in edits:
        if source.count(old)!=1:raise ValueError('Pinned native completion seam changed')
        source=source.replace(old,new)
    return MethodType(_compile(raw,source,{'_completion':completion}),method.__self__)


class Completion:
    """One bounded pending native acoustic stream; no persistent owner logging."""
    def __init__(self,duplex):
        self.duplex=duplex
        self.original_generate=duplex.streaming_generate
        self.original_prepare=duplex.prepare
        self.original_sampler=duplex.model.tts.generate_chunk
        self.resumable=_resumable(self.original_sampler)
        self.pending=None;self.events=[];self.acoustic_complete=False;self.used=False;self.prompt=None
        self.transformed=_duplex_generate(self.original_generate,self)

    def _event(self,kind,**details):
        self.events.append(dict(kind=kind,**details))

    def cancel(self,reason):
        if self.pending is None:return
        self.pending.close();self.pending=None;self.acoustic_complete=False
        d=self.duplex;d.tts_past_key_values=None;d.tts_text_start_pos=0;d.tts_current_turn_start_time=None
        d._reset_token2wav_for_new_turn()
        self._event('pending_acoustic_cancelled',reason=reason)

    def prepare(self,*args,**kwargs):
        self.cancel('session_reset');self.events=[];self.acoustic_complete=False
        return self.original_prepare(*args,**kwargs)

    def begin_native_speech(self,prompt):
        # This runs before native conditioning construction. A new native speak
        # unit following a linguistically ended turn supersedes that old tail.
        self.used=True;self.cancel('new_native_speaking_turn');self.prompt=prompt

    def _next(self):
        import torch
        try:
            with torch.inference_mode():tokens,cache,done=next(self.pending)
        except BaseException:
            self.cancel('sampling_failure');raise
        self.acoustic_complete=done
        if done:self.pending.close();self.pending=None
        self._event('terminal_acoustic_piece',audio_tokens=int(tokens.shape[1]),acoustic_complete=done,reason='acoustic_eos' if done else 'frame_yield',terminal_token_ceiling=MAX_TERMINAL_TOKENS)
        return tokens,cache

    def sample(self,terminal,**kwargs):
        if not terminal:
            self.acoustic_complete=False
            return self.original_sampler(**kwargs)
        if kwargs.get('max_new_token')!=26 or kwargs.get('min_new_tokens')!=0 or kwargs.get('force_no_stop'):
            raise ValueError('Pinned terminal sampler contract required')
        self.pending=self.resumable(**dict(kwargs,max_new_token=MAX_TERMINAL_TOKENS))
        return self._next()

    def _drain(self,result):
        d=self.duplex;started=time.monotonic();tokens,cache=self._next();sampled=time.monotonic()
        if self.acoustic_complete:
            d.tts_text_start_pos=0;d.tts_past_key_values=None;d.tts_current_turn_start_time=None
        else:
            d.tts_past_key_values=cache;d.tts_text_start_pos+=int(tokens.shape[1])
        result['audio_waveform']=d._generate_waveform_from_tokens(tokens,self.prompt,self.acoustic_complete,force_flush=False)
        if self.acoustic_complete:d._reset_token2wav_for_new_turn()
        result['n_tts_tokens']=int(tokens.numel())
        result['cost_tts']=sampled-started;result['cost_token2wav']=time.monotonic()-sampled
        result['cost_all']+=time.monotonic()-started

    def generate(self,*args,**kwargs):
        import torch
        d=self.duplex;self.events=[];self.used=False
        stopped=d.is_session_stop_set() or d.is_break_set()
        has_input=getattr(d,'pending_logits',None) is not None
        if stopped:self.cancel('native_session_stop_or_break')
        with torch.inference_mode():
            result=self.transformed(*args,**kwargs)
            if not stopped and has_input and self.pending is not None and not self.used:
                self._drain(result)
        result['speech_completion']=dict(version='native-terminal-resume-v1',pending=self.pending is not None,
            acoustic_complete=self.acoustic_complete,events=list(self.events),qualified=False)
        return result


def require_authorization():
    """Paid native entrypoint cannot borrow the consumed original-runtime grant."""
    import json,os
    from pathlib import Path
    lease_path=os.environ.get('DUPLEX_LIVE_LEASE_FILE')
    if not lease_path:raise ValueError('Explicit speech-completion native execution lease required')
    lease=json.loads(Path(lease_path).read_text())
    if lease.get('resource_key') not in ('venus-tts-completion-v57','venus-tts-completion-v58','venus-tts-completion-v59','venus-tts-completion-v60'):
        raise ValueError('Speech completion requires its own approved changed-runtime interval')
    from .guard import resource
    resource(lease)


def install(duplex):
    if getattr(duplex,'_speech_completion',None) is not None:raise ValueError('Completion already installed')
    if not duplex.generate_audio:raise ValueError('Completion requires native audio generation')
    # Construct and verify both pinned methods before mutating either entrypoint.
    handle=Completion(duplex)
    duplex.streaming_generate=handle.generate;duplex.prepare=handle.prepare;duplex._speech_completion=handle
    return handle
