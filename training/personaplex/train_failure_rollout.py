"""Compact, complete TRAIN-only native rollouts; no learning or answer injection.

All native frames, tokens and raw PCM are retained in two files per case rather
than thousands of small chunk files. This is failure collection, not acceptance.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(path, expected_sha256):
    path = Path(path)
    if digest(path) != expected_sha256:
        raise ValueError('Exact complete prepared TRAIN manifest required')
    value = json.loads(path.read_text())
    rows = value.get('cases', [])
    if (value.get('schema') != 'dl-native-TRAIN-user-audio-v1'
            or value.get('complete') is not True or value.get('required') != 1008
            or value.get('original_DEV_excluded') != 64 or len(rows) != 1008
            or len({r['id'] for r in rows}) != 1008
            or any(r.get('source_fold') != 'TRAIN' or r.get('family') != 'expanded-TRAIN'
                   or r.get('teacher_audio_in_model_input') is not False
                   or r.get('response_tail_seconds') != 60 for r in rows)):
        raise ValueError('Whole distinct1008 TRAIN-only cohort required')
    for row in rows:
        folder = Path(row['folder']).resolve()
        if (folder.parent != path.parent.resolve() or folder.name != row['id']
                or type(row['user_frames']) is not int or row['user_frames'] < 1):
            raise ValueError('Contained original user-only case required')
    return sorted(rows, key=lambda r: r['id'])


def shard(rows, rank, world_size):
    if (len(rows) != 1008 or len({r['id'] for r in rows}) != 1008
            or type(rank) is not int or world_size != 4 or not 0 <= rank < 4):
        raise ValueError('Four disjoint complete TRAIN shards required')
    return sorted(rows, key=lambda r: r['id'])[rank::world_size]


def collect_case(engine, row, out, guard):
    """Audio is the sole task input. Never read question/answer/scoring records."""
    wave = Path(row['folder']) / 'user.wav'
    if digest(wave) != row['input_sha256']:
        raise ValueError('Prepared source PCM changed')
    user, rate = sf.read(wave, dtype='float32')
    n = row['user_frames']
    if (rate != 24000 or row['input_rate'] != rate or user.ndim != 1
            or len(user) != n + 60 * rate or not np.isfinite(user).all()
            or np.any(user[n:]) or engine.frame_size != 1920):
        raise ValueError('Whole finite user-only24k input and fixed response tail required')
    padding = (-len(user)) % 1920
    user = np.pad(user, (0, padding))
    out = Path(out)
    out.mkdir(exist_ok=False)
    tokens, ret_count, sample_offset, frames = [], 0, 0, 0
    # Binary exclusive creation also protects partial outputs from overwrite.
    with (out / 'agent-raw.wav').open('xb') as raw, (out / 'trace.jsonl').open('x') as trace:
        with sf.SoundFile(raw, mode='w', samplerate=24000, channels=1,
                          format='WAV', subtype='FLOAT') as writer:
            for offset in range(0, len(user), 1920):
                guard()
                result = engine.step(user[offset:offset + 1920])
                pcm = np.asarray(result['agent_pcm'] if result['agent_pcm'] is not None else [], dtype='float32')
                if pcm.ndim != 1 or len(pcm) not in (0, 1920) or not np.isfinite(pcm).all():
                    raise ValueError('Unmodified finite native output frame required')
                writer.write(pcm)
                token = result['text_token']
                if token is not None:
                    tokens.append(token)
                ret_count += int(result['is_ret'])
                trace.write(json.dumps(dict(sequence=frames, text_token=token,
                    is_ret=result['is_ret'], raw_sample_offset=sample_offset,
                    audio_samples=len(pcm), pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest())) + '\n')
                frames += 1
                sample_offset += len(pcm)
    value = dict(id=row['id'], source_fold='TRAIN', complete=True,
                 input_sha256=row['input_sha256'], user_frames=n,
                 response_tail_seconds=60, input_padding_samples=padding,
                 native_frames=frames, output_samples=sample_offset,
                 generated_text=engine.decode_text(tokens), native_RET_events=ret_count,
                 raw_wav_sha256=digest(out / 'agent-raw.wav'),
                 trace_sha256=digest(out / 'trace.jsonl'),
                 teacher_audio_or_answer_provided=False, backend_connected=False,
                 paced_replay=False, live_latency_measured=False,
                 optimizer_updates=0, correctness_labels_created=False)
    with (out / 'result.json').open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return value


def run(args):
    import os
    import sys
    import torch
    from evals.context_execution import math_attention
    from .expanded_capture import PP, VENDOR, WEIGHTS, INSTRUCTIONS, REFERENCE_SHA
    from .expanded_evaluation import make_engine
    from .expanded_guard import check, ensure, TRAIN_FAILURE_AUTH
    from .native_process import process_identity

    lease = check(args.lease, 600)
    if lease.get('train_failure_authorization_sha256') != digest(TRAIN_FAILURE_AUTH):
        raise ValueError('Separate zero-update TRAIN collection admission required')
    plan = json.loads(args.plan.read_text())
    if (plan['source'] != args.source or digest(args.plan) != lease['collection_plan_sha256']
            or plan['input_manifest_sha256'] != '847084d665c2c848d46bd6ddcab0120ec6da2d5b6100730dc58dd6bca0bb1d25'):
        raise ValueError('Exact immutable collection source and plan required')
    rank = int(os.environ['LOCAL_RANK'])
    if os.environ.get('WORLD_SIZE') != '4' or not 0 <= rank < 4 or torch.cuda.device_count() != 4:
        raise ValueError('Four scheduler-owned independent native collectors required')
    torch.cuda.set_device(rank)
    os.environ.update(CS_BASE_DIR=str(PP), CS_WEIGHTS_DIR=str(WEIGHTS), CS_VOICES_DIR=str(WEIGHTS / 'voices'))
    sys.path.insert(0, str(VENDOR))
    from contextspan.model.engine import seed_all
    math_attention()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    rows = shard(inputs(plan['input_manifest'], plan['input_manifest_sha256']), rank, 4)
    selection = plan['selection']
    if selection['step'] != 196 or selection['checkpoint_sha256'] != '96cff747f42fff1c36572800debcee2d435f20efaebf2c8ccfe9720e9bec3c87':
        raise ValueError('Only the complete preserved196 checkpoint may be collected')
    args.base, args.capacity = 'personaplex', 'expanded105M'
    args.trainer_source = selection['trainer_source']
    seed_all(42)
    engine = make_engine(args, selection)
    capture = Path(plan['parent_root']) / 'capture/shard-00'
    reference = json.loads((capture / 'capture.json').read_text())
    if (reference['reference_source_sha256'] != REFERENCE_SHA or reference['instructions'] != INSTRUCTIONS
            or digest(capture / 'voice.pt') != reference['voice_sha256']):
        raise ValueError('Original RP reference/persona required')
    voice = torch.load(capture / 'voice.pt', map_location='cpu', weights_only=True)['codes']
    folder = args.root / ('rank-%02d' % rank)
    folder.mkdir(exist_ok=False)
    results = []
    def guard():
        # Keep exact per-frame provider/health bounds without spawning a Git
        # process for every80ms frame. Immutable source is rechecked per case.
        ensure(lease, json.loads(Path(os.environ['DUPLEX_LIVE_GUARD_FILE']).read_text()))
    with torch.inference_mode():
        for row in rows:
            check(args.lease, 180)
            seed_all(42)
            engine.reset()
            engine.set_persona(INSTRUCTIONS, voice)
            value = collect_case(engine, row, folder / row['id'], guard)
            results.append(value)
            print(json.dumps(dict(event='TRAIN_native_case_complete', rank=rank,
                                  completed=len(results), required=252)), flush=True)
    terminal = dict(complete=True, required=252, rank=rank, source=args.source,
                    actual_CUDA=True, process_identity=process_identity(),
                    input_manifest_sha256=plan['input_manifest_sha256'],
                    selection=selection, cases=results, optimizer_updates=0,
                    learned_RET_head_installed=False, teacher_forced=False,
                    backend_connected=False, correctness_labels_created=False,
                    product_accepted=False)
    with (folder / 'terminal.json').open('x') as stream:
        json.dump(terminal, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'plan', 'lease'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source', required=True)
    run(parser.parse_args())
