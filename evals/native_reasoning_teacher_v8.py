"""Versioned repair recorder: frozen quiescent transport, corrected readback.

Functions get isolated global dictionaries; no frozen module or running capture
is modified. Only diagnosed unclear acknowledgments use a generic clarity prompt.
No answers, task identifiers, forced delegation tokens or scoring fields are added.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType

from evals import native_reasoning_teacher_v7 as original
from evals import native_reasoning_stream_v6 as stream
from evals import native_reasoning_teacher_cutoff_readback as readback
from evals.native_reasoning_source import sha

CLARITY = ('\nSpeech clarity: Use complete, clearly spoken English words rather than '
           'murmurs, filled pauses or unfinished acknowledgment fragments. '
           'When acknowledging cancellation after its receipt, say "Stopped that search". '
           'When acknowledging replacement after its receipt, say "Switching tasks". '
           'Speak each arithmetic answer and its unit clearly. All delegation and '
           'interruption policies above still apply.')


def configuration(clarity):
    value = deepcopy(stream.configuration())
    if clarity: value['instructions'] += CLARITY
    return value


def code_hashes():
    return dict(original.code_hashes(), **{
        Path(readback.__file__).name: sha(Path(readback.__file__)),
        Path(__file__).name: sha(Path(__file__))})


def bind(function, **values):
    rebound = FunctionType(function.__code__, dict(function.__globals__, **values),
                           function.__name__, function.__defaults__, function.__closure__)
    rebound.__kwdefaults__ = function.__kwdefaults__
    return rebound


def implementation(clarity):
    if type(clarity) is not bool: raise ValueError('Explicit versioned clarity profile required')
    config = lambda: configuration(clarity)
    capture = bind(stream.capture, configuration=config)
    record = bind(original.capture_one, capture=capture, configuration=config, code_hashes=code_hashes)
    verify = bind(readback.verify_record, configuration=config, code_hashes=code_hashes)
    return record, verify
