"""User-only audio extraction for TRAIN failure collection, never teacher input."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .train_failure_inputs import RECORD_KEYS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(source, record, destination):
    """Verify the original stereo source and write only its complete user audio.

    The fixed response window is independent of the teacher's answer length.
    This prepares inference inputs; it performs no inference or correctness scoring.
    """
    source = Path(source).resolve()
    destination = Path(destination)
    if (set(record) != RECORD_KEYS or record['fold'] != 'TRAIN'
            or record['user_channel'] != 1
            or record['response_tail_seconds'] != 60
            or record['required_user_only_extraction'] is not True
            or type(record['user_frames']) is not int or record['user_frames'] < 1):
        raise ValueError('Exact TRAIN-only user input record required')
    relative = Path(record['source_wave'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Contained original source required')
    wave = (source / relative).resolve()
    side = wave.with_suffix('.json')
    if (not wave.is_relative_to(source) or not side.resolve().is_relative_to(source)
            or digest(wave) != record['source_wave_sha256']
            or digest(side) != record['source_sidecar_sha256']):
        raise ValueError('Exact original TRAIN waveform and sidecar required')
    if json.loads(side.read_text())['fold'] != 'TRAIN':
        raise ValueError('Original source fold must remain TRAIN')
    pcm, rate = sf.read(wave, dtype='float32', always_2d=True)
    length = record['user_frames']
    if (rate != 24000 or pcm.shape[1] != 2 or length > len(pcm)
            or not np.isfinite(pcm).all()):
        raise ValueError('Finite complete original stereo24k recording required')
    user = np.concatenate((pcm[:length, 1], np.zeros(60 * rate, dtype=np.float32)))
    # Exclusive creation: never overwrite a completed native input or rollout.
    with destination.open('xb') as stream:
        sf.write(stream, user, rate, format='WAV', subtype='FLOAT')
    saved, saved_rate = sf.read(destination, dtype='float32')
    if saved_rate != rate or not np.array_equal(saved, user):
        raise ValueError('User-only extraction changed PCM')
    return dict(id=record['id'], family='expanded-TRAIN', source_fold='TRAIN',
                folder=str(destination.parent), input_rate=rate,
                frames=(len(user) + rate - 1) // rate,
                input_sha256=digest(destination), user_frames=length,
                original_source_sha256=record['source_wave_sha256'],
                teacher_audio_in_model_input=False, response_tail_seconds=60,
                native_inference_started=False, new_optimizer_updates=0)
