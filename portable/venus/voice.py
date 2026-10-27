"""Explicit, hash-bound native voice reference; no external speech fallback."""
import hashlib
from pathlib import Path

def reference(model_dir,override=None,expected_sha256=None):
    path=Path(override).resolve() if override else Path(model_dir).resolve()/'assets/HT_ref_audio.wav'
    # The cloud speech runtime is Python3.10 (file_digest arrived in3.11).
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if override and (not expected_sha256 or digest!=expected_sha256):
        raise ValueError('Custom voice reference requires its exact approved SHA256')
    import soundfile as sf
    import numpy as np
    audio,sr=sf.read(path,dtype='float32')
    if sr!=16000 or audio.ndim!=1 or not np.isfinite(audio).all():
        raise ValueError('Native reference must be finite mono16kHz audio')
    seconds=len(audio)/sr;peak=float(np.abs(audio).max()) if len(audio) else 0
    if not 2<=seconds<=30 or not 0<peak<=.995:
        raise ValueError('Reference must be2..30seconds, non-silent and unclipped; do not lower gain to pass')
    return dict(path=str(path),sha256=digest,seconds=seconds,peak=peak,
        mode='custom-native-reference' if override else 'released-native-reference',
        qualified=False,hosted_tts=False)
