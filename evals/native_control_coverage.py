"""Read-only audit of the exact saved native49 source coverage and quiet PCM.

This checks saved recipe bindings, not tensor mask contents or model capability.
It never edits an approval, recording, cache, model, or training recipe.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_path(archive, remote):
    relative = Path(remote).relative_to('/projects/u6si/duplex-live')
    if '..' in relative.parts:
        raise ValueError('Archive path traversal')
    result = (archive / relative).resolve()
    result.relative_to(archive.resolve())
    return result


def check_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'Saved recipe hash mismatch: {path.name}')
    return actual


def quiet_windows(sidecar):
    """Use actual complete utterance bounds, never one merged assistant event."""
    timeline = sidecar['timeline']
    meta = timeline['meta']
    users = [e for e in timeline['events']
             if e['ch'] == 'user' and e.get('kind') == 'utterance']
    if 'quiet_until_user_ordinal' in meta:
        start, stop = 0, meta['quiet_until_user_ordinal']
    elif 'quiet_between_user_ordinals' in meta:
        start, stop = meta['quiet_between_user_ordinals']
    else:
        return []
    if not 0 <= start < stop < len(users):
        raise ValueError('Invalid approved quiet ordinals')
    begin = users[start]['actual_end_ms'] / 1000
    end = users[stop]['actual_end_ms'] / 1000
    if end <= begin:
        raise ValueError('Invalid actual quiet interval')
    words = [word for word, (a, b), _ in sidecar['alignments']
             if a < end and b > begin]
    return [dict(start_s=begin, end_s=end, seconds=end - begin,
                 complete_release_utterance=users[stop]['text'],
                 aligned_words_during_quiet=words)]


def audit(recipe_path, archive):
    import numpy as np
    import soundfile as sf
    archive = archive.resolve()
    recipe = json.loads(recipe_path.read_text())
    policy = recipe['combined_recipe']
    old = policy['retention_teacher_text']['teachers']
    new = policy['additional_native_v2']['teacher']['teachers']
    teachers = old + new
    expected = policy['assistant_audio_auxiliary']['teacher_ids']
    ids = [t['id'] for t in teachers]
    if len(old) != 25 or len(new) != 24 or len(set(ids)) != 49 or ids != expected:
        raise ValueError('Exact saved ordered old25+new24 required')
    rows = []
    bound_files = {}
    for teacher in teachers:
        cache_dir = local_path(archive, teacher['cache'])
        cache_file = cache_dir / 'cache.json'
        bound_files[str(cache_file.relative_to(archive))] = check_hash(cache_file, teacher['cache_sha256'])
        cache = json.loads(cache_file.read_text())
        conversion = local_path(archive, cache['conversion'])
        ingestion_file = conversion / 'ingestion.json'
        bound_files[str(ingestion_file.relative_to(archive))] = check_hash(ingestion_file, teacher['ingestion_sha256'])
        code_file = cache_dir / (teacher['id'] + '.pt')
        bound_files[str(code_file.relative_to(archive))] = check_hash(code_file, teacher['codes_sha256'])
        folder = conversion / teacher['id']
        for name, key in [('conversation.wav', 'wav_sha256'), ('conversation.json', 'sidecar_sha256')]:
            path = folder / name
            bound_files[str(path.relative_to(archive))] = check_hash(path, teacher[key])
        ingestion = json.loads(ingestion_file.read_text())
        accepted = [r for r in ingestion['accepted'] if r['scenario_id'] == teacher['id']]
        if len(accepted) != 1 or accepted[0]['split'] != 'train' or teacher['source_split'] != 'train':
            raise ValueError('Teacher is not uniquely approved TRAIN source')
        sidecar = json.loads((folder / 'conversation.json').read_text())
        meta = sidecar['timeline']['meta']
        users = [e for e in sidecar['timeline']['events'] if e['ch'] == 'user' and e.get('kind') == 'utterance']
        windows = quiet_windows(sidecar)
        for window in windows:
            pcm, rate = sf.read(folder / 'conversation.wav', always_2d=True, dtype='float64')
            if rate != 24000 or pcm.shape[1] != 2 or not np.isfinite(pcm).all():
                raise ValueError('Invalid approved stereo native waveform')
            # recovery_teacher_convert.py explicitly stores [assistant, user].
            samples = pcm[round(window['start_s'] * rate):round(window['end_s'] * rate), 0]
            if not len(samples) or round(window['end_s'] * rate) > len(pcm):
                raise ValueError('Empty actual quiet interval')
            window.update(assistant_peak=float(np.abs(samples).max()),
                          assistant_rms=float(np.sqrt(np.mean(samples ** 2))),
                          assistant_nonzero_samples=int(np.count_nonzero(samples)),
                          native_80ms_frames_fully_inside=int(round(window['end_s'] * 1000) // 80
                                                            - math.ceil(round(window['start_s'] * 1000) / 80)))
            if window['aligned_words_during_quiet'] or window['assistant_nonzero_samples']:
                raise ValueError('Approved teacher contains speech/PCM in declared quiet interval')
        rows.append(dict(id=teacher['id'], kind=teacher['kind'],
                         behavior=meta.get('behavior'), user_voice=meta['user_voice'],
                         user_utterance_count=len(users),
                         user_text=[e['text'] for e in users],
                         quiet_windows=windows,
                         text_targets=teacher['text_targets'],
                         spoken_targets=teacher['spoken_targets'],
                         other_text_targets=teacher['text_targets'] - teacher['spoken_targets'],
                         mask_sha256=teacher['mask_sha256'],
                         targets_sha256=teacher['targets_sha256'],
                         prefix_loss_targets=teacher['prefix_loss_targets'],
                         span_loss_targets=teacher['span_loss_targets'],
                         user_loss_targets=teacher['user_loss_targets']))
    controls = [r for r in rows if r['quiet_windows']]
    return dict(schema='dl-native49-control-source-coverage-v1',
                at_utc=datetime.now(timezone.utc).isoformat(),
                source_recipe_sha256=digest(recipe_path), source_payload=recipe['repo_commit'],
                implementation_sha256=digest(Path(__file__)),
                source_integrity_passed=True, all49_complete=True,
                bound_file_count=len(bound_files), bound_files=bound_files,
                counts=dict(teachers=49, old=25, new=24,
                            explicit_quiet_control_teachers=len(controls),
                            behaviors=dict(Counter(r['behavior'] or 'no-explicit-quiet-metadata' for r in rows)),
                            control_user_voices=sorted({r['user_voice'] for r in controls})),
                controls=controls, teachers=rows,
                limits=['Explicit source metadata coverage; absence does not prove zero implicit control learning.',
                        'Hash-verified cached tensors, but tensor mask contents are NOT decoded or independently reapproved here.',
                        'Other text targets include all non-spoken clock/control labels, not only deliberate silence.',
                        'Quiet PCM checks apply to declared windows in approved teacher audio, not candidate output.',
                        'No causal diagnosis or behavioral qualification is claimed.'],
                source_or_approval_changes=False, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Audit outputs must be new versioned files')
    result = audit(args.recipe, args.archive)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('source_integrity_passed', 'bound_file_count', 'counts')}))


if __name__ == '__main__':
    main()
