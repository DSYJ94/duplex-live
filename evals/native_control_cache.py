"""Approved control40 -> pinned native Mimi/text cache; no numerical/training approval.

The old24/25/49 loaders stay unchanged. All40 TRAIN scenes and all five families
are required. Every waveform sample is encoded, with integer80ms frame coverage.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

from data.synthetic.native_control_render import sha256
from evals.native_control_convert import verify as verify_conversion
from evals.personaplex_train import MODEL_HASHES
from evals.recovery_encode import native_word_timing

SCHEMA = 'dl-native-control3-codec-cache-v1'
CODEC_COMMIT = '8c6dfc101b7871baa428424bcdc583b74fb561d9'
INTERLEAVER_COMMIT = '2acc879fe7c48f885a18f6cc9548bccb2674d87b'
FAMILIES = {'no-release', 'release-reason', 'distractor-release', 'pause-resume', 'replace-cue'}
ASSETS = ('tokenizer-e351c8d8-checkpoint125.safetensors', 'tokenizer_spm_32k_3.model')


def inventory(conversion):
    rows = conversion['accepted']
    if (conversion['complete40'] is not True or conversion['training_approved'] is not False
            or conversion['channel_order'] != ['assistant', 'user']
            or len(rows) != 40 or len({r['scenario_id'] for r in rows}) != 40
            or any(r['split'] != 'train' or r['scenario_class'] != 'native_control3_teacher' for r in rows)
            or Counter(r['family'] for r in rows) != Counter({f: 8 for f in FAMILIES})):
        raise ValueError('Distinct complete40 TRAIN control corpus with original five families required')
    return rows


def frame_duration(frames):
    if type(frames) is not int or frames < 1:
        raise ValueError('Positive integer native frame count required')
    # Interleaver uses ceil(duration*12.5). One-ULP rounding toward zero avoids
    # inventing a nonexistent final frame; this never changes PCM or word times.
    duration = math.nextafter(frames / 12.5, 0.)
    if math.ceil(duration * 12.5) != frames:
        raise ValueError('Cannot represent exact integer native frame coverage')
    return duration


def source_pins(codec_source, interleaver_source, model):
    result = {}
    for label, source, commit, relative in (
            ('codec', codec_source, CODEC_COMMIT, 'moshi/moshi/models/loaders.py'),
            ('interleaver', interleaver_source, INTERLEAVER_COMMIT, 'finetune/data/interleaver.py')):
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
        if actual != commit or dirty:
            raise ValueError('Clean pinned native codec/interleaver source required')
        result[label] = dict(source=str(source), commit=commit, implementation_sha256=sha256(source / relative))
    result['assets'] = {name: sha256(model / name) for name in ASSETS}
    if any(result['assets'][name] != MODEL_HASHES[name] for name in ASSETS):
        raise ValueError('Pinned native codec or vocabulary changed')
    return result


def native_modules(codec_source, interleaver_source):
    sys.path[:0] = [str(codec_source / 'moshi'), str(interleaver_source)]
    from moshi.models import loaders
    from finetune.data import interleaver
    if (not Path(loaders.__file__).resolve().is_relative_to(codec_source.resolve())
            or not Path(interleaver.__file__).resolve().is_relative_to(interleaver_source.resolve())):
        raise ValueError('A different native codec/interleaver module was already imported')
    return loaders, interleaver.Interleaver


def text_builder(codec_source, interleaver_source, model, device):
    import sentencepiece
    loaders, Interleaver = native_modules(codec_source, interleaver_source)
    tokenizer = sentencepiece.SentencePieceProcessor(str(model / ASSETS[1]))
    builder = Interleaver(tokenizer, 12.5, text_padding=3, end_of_text_padding=0,
                          zero_padding=-1, keep_main_only=True, keep_and_shift=True, device=device)
    return loaders, tokenizer, builder


def runtime(codec_source, interleaver_source, model):
    loaders, tokenizer, builder = text_builder(codec_source, interleaver_source, model, 'cuda')
    mimi = loaders.get_mimi(model / ASSETS[0], device='cuda', num_codebooks=8).eval().requires_grad_(False)
    if mimi.sample_rate != 24000 or mimi.frame_rate != 12.5:
        raise ValueError('Native codec rate changed')
    return mimi, tokenizer, builder


def encode_item(audio, side, mimi, builder, tokenizer):
    import numpy as np
    import torch
    if (audio.ndim != 2 or audio.shape[1] != 2 or not len(audio) or len(audio) % 1920
            or not np.isfinite(audio).all() or float(np.abs(audio).max()) > .995
            or side['reference_arrivals'] or side['timeline']['meta']['split'] != 'train'
            or any(e['kind'] == 'marker' for e in side['timeline']['events'])):
        raise ValueError('Complete unclipped native assistant-left/user-right control PCM required')
    frames = len(audio) // 1920
    with torch.no_grad():
        acoustics = mimi.encode(torch.as_tensor(audio.T.copy(), dtype=torch.float32, device='cuda')[:, None])
        # No truncate/pad of acoustic codes: reject any unexpected geometry.
        if acoustics.shape != (2, 8, frames):
            raise ValueError('Native codec did not cover every original PCM frame exactly')
        text = builder.prepare_item(side['alignments'], frame_duration(frames))
        if text.shape != (1, 1, frames):
            raise ValueError('Native text interleaver did not preserve integer frame coverage')
        codes = torch.cat([text, acoustics.reshape(1, 16, frames)], dim=1).cpu()
    timing = check_codes(codes, frames, side, tokenizer)
    return codes, timing


def check_codes(codes, frames, side, tokenizer):
    import torch
    if (codes.dtype != torch.int64 or codes.shape != (1, 17, frames)
            or (codes[:, 1:] < 0).any() or (codes[:, 1:] >= 2048).any()
            or (codes[:, :1] < 0).any() or (codes[:, :1] >= tokenizer.vocab_size()).any()
            or (codes[:, :1] == 4).any()):
        raise ValueError('Invalid complete control codebook/text ownership or spurious delegation target')
    return native_word_timing(codes[0, 0].tolist(), side['alignments'], tokenizer.encode, 12.5)


def build(data, model, codec_source, interleaver_source, out):
    conversion = verify_conversion(data)  # Full quality+semantic chain before CUDA/import/output.
    rows = inventory(conversion)
    pins = source_pins(codec_source, interleaver_source, model)
    import soundfile as sf
    import torch
    mimi, tokenizer, builder = runtime(codec_source, interleaver_source, model)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, conversion=str(data), ingestion_sha256=sha256(data / 'ingestion.json'),
                  created_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha256(Path(__file__)),
                  source_pins=pins, model=str(model), codec_source=str(codec_source), interleaver_source=str(interleaver_source),
                  channel_order=['assistant', 'user'], frame_rate=12.5, sample_rate=24000, examples=[],
                  complete40=False, optimizer_updates=0, training_approved=False,
                  native_engine_input_and_numerical_approval_required=True,
                  acoustic_codes_cropped_or_padded=False, source_words_audio_receipts_folds_changed=False,
                  assistant_codebooks=list(range(1, 9)), user_codebooks=list(range(9, 17)),
                  execution=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name(), codec_dtype=str(next(mimi.parameters()).dtype),
                                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32))
    for row in rows:
        audio, rate = sf.read(row['path'], dtype='float32', always_2d=True)
        if rate != 24000 or len(audio) != row['frames']:
            raise ValueError('Approved control source rate/coverage changed')
        side = json.loads(Path(row['path']).with_suffix('.json').read_text())
        codes, timing = encode_item(audio, side, mimi, builder, tokenizer)
        name = row['scenario_id'] + '.pt'
        torch.save(dict(codes=codes, markers=[], arrivals=[]), out / name)
        report['examples'].append(dict(id=row['scenario_id'], split='train', family=row['family'], file=name,
                                       sha256=sha256(out / name), frames=len(audio) // 1920,
                                       source_artifacts=row['artifacts'], native_word_timing=timing))
        (out / 'cache.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(dict(id=row['scenario_id'], complete=len(report['examples']), frames=codes.shape[-1])), flush=True)
    report['complete40'] = True
    (out / 'cache.json').write_text(json.dumps(report, indent=2) + '\n')
    verify(out)
    return report


def verify(cache):
    import torch
    report = json.loads((cache / 'cache.json').read_text())
    data = Path(report['conversion'])
    conversion = verify_conversion(data)
    rows = inventory(conversion)
    pins = source_pins(Path(report['codec_source']), Path(report['interleaver_source']), Path(report['model']))
    if (report['schema'] != SCHEMA or report['code_sha256'] != sha256(Path(__file__))
            or report['ingestion_sha256'] != sha256(data / 'ingestion.json') or report['source_pins'] != pins
            or report['channel_order'] != ['assistant', 'user'] or report['frame_rate'] != 12.5
            or report['sample_rate'] != 24000 or report['complete40'] is not True
            or report['training_approved'] is not False or report['optimizer_updates'] != 0
            or report['acoustic_codes_cropped_or_padded'] is not False
            or report['source_words_audio_receipts_folds_changed'] is not False
            or report['native_engine_input_and_numerical_approval_required'] is not True
            or report['assistant_codebooks'] != list(range(1, 9)) or report['user_codebooks'] != list(range(9, 17))
            or [r['id'] for r in report['examples']] != [r['scenario_id'] for r in rows]):
        raise ValueError('Distinct unchanged complete40 native cache required')
    _, tokenizer, builder = text_builder(Path(report['codec_source']), Path(report['interleaver_source']),
                                         Path(report['model']), 'cpu')
    for entry, row in zip(report['examples'], rows):
        path = cache / entry['file']
        if path.name != row['scenario_id'] + '.pt' or sha256(path) != entry['sha256'] or path.is_symlink():
            raise ValueError('Native control cache artifact changed')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if set(payload) != {'codes', 'markers', 'arrivals'} or payload['markers'] or payload['arrivals']:
            raise ValueError('Control-only cache acquired requests or supplied context')
        side = json.loads(Path(row['path']).with_suffix('.json').read_text())
        frames = row['frames'] // 1920
        timing = check_codes(payload['codes'], frames, side, tokenizer)
        expected_text = builder.prepare_item(side['alignments'], frame_duration(frames))
        if not torch.equal(expected_text, payload['codes'][:, :1]):
            raise ValueError('Native text/PAD/EPAD lane differs from independent pinned interleaving')
        if entry != dict(id=row['scenario_id'], split='train', family=row['family'], file=path.name,
                         sha256=sha256(path), frames=frames, source_artifacts=row['artifacts'], native_word_timing=timing):
            raise ValueError('Native control cache metadata/timing differs from approved conversion')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify', type=Path)
    for name in ('data', 'model', 'codec-source', 'interleaver-source', 'out'):
        ap.add_argument('--' + name, type=Path)
    args = ap.parse_args()
    if args.verify:
        report = verify(args.verify)
    elif all((args.data, args.model, args.codec_source, args.interleaver_source, args.out)):
        report = build(args.data, args.model, args.codec_source, args.interleaver_source, args.out)
    else:
        ap.error('Use --verify or all data/model/source/out paths')
    print(json.dumps(dict(complete=len(report['examples']), training_approved=False, optimizer_updates=0)))


if __name__ == '__main__':
    main()
