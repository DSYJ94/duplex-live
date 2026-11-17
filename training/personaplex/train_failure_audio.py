"""User-only audio extraction for TRAIN failure collection, never teacher input."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .train_failure_inputs import RECORD_KEYS, build


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


def prepare(inputs, expected_sha256, out):
    """Prepare the entire pinned1008 cohort; publish no subset as complete."""
    inputs, out = Path(inputs), Path(out)
    if digest(inputs) != expected_sha256:
        raise ValueError('Pinned complete input manifest required')
    value = json.loads(inputs.read_text())
    source = Path(value['source_root'])
    manifest, semantic = source / 'manifest.json', source / 'qualification/plan.json'
    if (digest(manifest) != value['source_manifest_sha256']
            or digest(semantic) != value['semantic_parent_sha256']):
        raise ValueError('Original complete qualified source parents changed')
    expected = build(json.loads(manifest.read_text()), json.loads(semantic.read_text()))
    if (value.get('schema') != expected['schema'] or value.get('required') != 1008
            or value.get('original_DEV_excluded') != 64
            or value['records'] != expected['records']):
        raise ValueError('Exact complete original TRAIN membership required')
    if any(not isinstance(r['id'], str) or Path(r['id']).name != r['id']
           or r['id'] in ('', '.', '..') for r in value['records']):
        raise ValueError('Contained distinct case IDs required')
    out.mkdir(exist_ok=False)
    rows = []
    for record in value['records']:
        folder = out / record['id']
        folder.mkdir()
        rows.append(extract(source, record, folder / 'user.wav'))
    result = dict(schema='dl-native-TRAIN-user-audio-v1', complete=True,
                  required=1008, original_DEV_excluded=64, cases=rows,
                  input_manifest_sha256=expected_sha256,
                  native_inference_started=False, correctness_labels_created=False,
                  teacher_audio_in_model_input=False, new_optimizer_updates=0,
                  external_API_calls=0)
    with (out / 'inputs.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.inputs, args.inputs_sha256, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))
