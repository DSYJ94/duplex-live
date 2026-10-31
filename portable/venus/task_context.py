"""Bounded causal audio evidence for a native-selected task, kept in memory.

This captures what the frontend heard. It does not transcribe, route, solve,
modify a request, call a provider, or infer missing premises from test answers.
"""
from collections import deque
from dataclasses import dataclass, field
import hashlib


@dataclass(frozen=True)
class AudioEvidence:
    first_sequence: int
    last_sequence: int
    pcm16: bytes = field(repr=False)

    def __post_init__(self):
        if (type(self.first_sequence) is not int or type(self.last_sequence) is not int
                or not 0<=self.first_sequence<=self.last_sequence
                or not 1<=self.last_sequence-self.first_sequence+1<=30
                or type(self.pcm16) is not bytes
                or len(self.pcm16)!=(self.last_sequence-self.first_sequence+1)*32000):
            raise ValueError('Bounded immutable original audio with exact native sample clock required')

    def receipt(self):
        return dict(first_sequence=self.first_sequence,last_sequence=self.last_sequence,
                    start_sample=self.first_sequence*16000,end_sample=(self.last_sequence+1)*16000,
                    sample_rate=16000,channels=1,sample_width_bytes=2,
                    pcm_sha256=hashlib.sha256(self.pcm16).hexdigest(),
                    provenance='causal original user audio; not an ASR transcript',
                    after_opening_audio_included=False)


class CausalAudioBuffer:
    def __init__(self, maximum_frames=30):
        if type(maximum_frames) is not int or not 1<=maximum_frames<=30:
            raise ValueError('At most30seconds of in-memory evidence')
        self.frames=deque(maxlen=maximum_frames)
        self.next_sequence=0

    def append(self, sequence, pcm16):
        if type(sequence) is not int or sequence!=self.next_sequence:
            raise ValueError('Consecutive native input sequences required')
        if not isinstance(pcm16,(bytes,bytearray)) or len(pcm16)!=32000:
            raise ValueError('Exactly one second16kHz monoPCM16 required')
        self.frames.append((sequence,bytes(pcm16)))
        self.next_sequence+=1

    def freeze_at_request_opening(self):
        if not self.frames:raise ValueError('No observed audio evidence')
        return AudioEvidence(self.frames[0][0],self.frames[-1][0],b''.join(x[1] for x in self.frames))
