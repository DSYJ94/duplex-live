"""Explicit WAV-only I/O for the ARM TorchAudio2.9 runtime.

TorchAudio2.9 moved decoding to TorchCodec, whose matching release has no ARM
wheel. Native model math is untouched: the approved mono WAV references decode
to the same normalized float32 samples through SoundFile. Never normalize gain,
trim, resample or hide clipping here. Older runtimes remain unchanged.
"""
import os

def loader(torch):
    def load(uri,frame_offset=0,num_frames=-1,normalize=True,channels_first=True,format=None,buffer_size=4096,backend=None):
        import numpy as np
        import soundfile as sf
        if normalize is not True or format not in (None,'wav','WAV') or backend not in (None,'soundfile'):
            raise ValueError('Only the explicitly supported normalized WAV decoding contract is admitted')
        if frame_offset<0 or num_frames < -1:raise ValueError('Invalid WAV frame selection')
        with sf.SoundFile(uri) as stream:
            if stream.format not in ('WAV','WAVEX'):raise ValueError('Native reference must be an original WAV')
            stream.seek(frame_offset);data=stream.read(frames=num_frames,dtype='float32',always_2d=True);sr=stream.samplerate
        if channels_first:data=data.T
        return torch.from_numpy(np.ascontiguousarray(data)),sr
    return load

def install():
    if os.environ.get('DUPLEX_WAV_IO')!='soundfile-wav-v1':return False
    import torch,torchaudio
    if not torchaudio.__version__.startswith('2.9.0'):raise ValueError('Explicit TorchAudio2.9 WAV compatibility only')
    torchaudio.load=loader(torch)
    return True
