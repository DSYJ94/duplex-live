"""Diagnostic-only observation of the pinned sampler's actual termination.

No sampling arguments or return values change. Thread-local Python profiling
and one scalar GPU read add overhead; this is not a live latency benchmark.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import hashlib
import inspect
import sys

SOURCE_SHA256='b7b4070cead9b88347f6e39d8333706ec007e2a832d6bcde5755063bf1453774'


@contextmanager
def observe(tts):
    original=tts.generate_chunk
    raw=inspect.unwrap(original.__func__)
    if raw.__name__!='generate_chunk' or hashlib.sha256(Path(inspect.getfile(raw)).read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError('Exact pinned unmodified TTS sampler required')
    records=[]
    def profile(frame,event,result):
        if event!='return' or frame.f_code is not raw.__code__ or not isinstance(result,tuple):return
        try:
            v=frame.f_locals;t=v['t'];maximum=v['max_new_token']
            eos=bool(v['new_tokens'][:,t].eq(v['eos_token']).any().item())
            records.append(dict(sampled_steps=t+1,returned_audio_tokens=int(result[0].shape[1]),
                maximum=maximum,minimum=v['min_new_tokens'],force_no_stop=bool(v['force_no_stop']),
                condition_tokens=int(v['inputs_embeds'].shape[1]),
                sampled_last_token_is_eos=eos,discarded_non_eos_last_sample=not eos,
                termination='eos' if eos else 'budget' if t+1==maximum else 'unknown',
                observer_only=True,native_speech_qualified=False))
        except Exception as error:
            records.append(dict(observer_error=type(error).__name__,native_speech_qualified=False))
    @wraps(original)
    def call(*args,**kwargs):
        previous=sys.getprofile()
        if previous is not None:raise RuntimeError('Do not replace an existing profiler')
        sys.setprofile(profile)
        try:return original(*args,**kwargs)
        finally:sys.setprofile(previous)
    tts.generate_chunk=call
    try:yield records
    finally:tts.generate_chunk=original


@contextmanager
def observe_duplex(duplex):
    """Capture conditioning identities, never hidden vectors or input audio."""
    original=duplex._convert_results_to_tts_input
    conditioning=[]
    def convert(rows):
        value=original(rows)
        conditioning.append(dict(native_token_ids=[int(r[0]) for r in rows],
            native_rows=len(rows),condition_tokens=int(value.shape[1])))
        return value
    duplex._convert_results_to_tts_input=convert
    try:
        with observe(duplex.model.tts) as records:
            yield dict(sampler=records,conditioning=conditioning)
    finally:duplex._convert_results_to_tts_input=original
