"""Assemble the complete new control source from immutable, receipted clips.

No generation, resampling, gain adjustment, source approval, or model inference.
Terminal quiet is real PCM, never an empty/fake assistant utterance. Outputs are
raw synthetic source material and cannot bypass teacher/quality approval.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from data.synthetic.native_control_repair import build, validate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_clip(path, expected):
    import numpy as np
    import soundfile as sf
    receipt_path = path.with_suffix('.receipt.json')
    receipt = json.loads(receipt_path.read_text())
    allowed = {'sha256', 'input_text', 'voice', 'source_model', 'request_id',
               'attempt', 'sample_rate', 'seconds', 'instructions'}
    if set(receipt) - allowed:
        raise ValueError('Unexpected receipt fields: do not archive headers or credentials')
    if (receipt.get('sha256') != sha256(path)
            or receipt.get('input_text') != expected['text']
            or receipt.get('voice') != expected['voice']
            or not receipt.get('source_model')):
        raise ValueError('Unbound source words, voice, model or waveform receipt')
    pcm, rate = sf.read(path, dtype='float32')
    if rate != 24000 or pcm.ndim != 1 or not .1 * rate <= len(pcm) <= 30 * rate:
        raise ValueError('Require nonempty 24kHz mono source audio, without resampling')
    if not np.isfinite(pcm).all() or not .001 < float(np.abs(pcm).max()) <= .995:
        raise ValueError('Empty, nonfinite or clipped input: preserve and reject, never lower gain')
    return pcm, dict(source_model=receipt['source_model'], voice=receipt['voice'],
                     input_text=receipt['input_text'], waveform_sha256=receipt['sha256'],
                     receipt_sha256=sha256(receipt_path),
                     request_id=receipt.get('request_id'), peak=float(np.abs(pcm).max()),
                     frames=len(pcm), sample_rate=rate)


def assemble_scene(scene, clips, folder, plan_sha256):
    import numpy as np
    import soundfile as sf
    events, pieces, receipts = [], [], []
    end = 0
    for index, turn in enumerate(scene['turns']):
        voice = scene['user_voice'] if turn['role'] == 'user' else scene['assistant_voice']
        clip = clips / scene['id'] / f'{index:02d}_{turn["role"]}.wav'
        pcm, receipt = load_clip(clip, dict(text=turn['text'], voice=voice))
        # Every scheduled utterance begins on the unchanged native 80ms clock.
        start = 19200 if index == 0 else math.ceil((end + round(turn['gap_before_s'] * 24000)) / 1920) * 1920
        end = start + len(pcm)
        events.append(dict(ch=turn['role'], text=turn['text'], kind='utterance',
                           source_turn_index=index, actual_start_sample=start,
                           actual_end_sample=end, actual_start_ms=start / 24,
                           actual_end_ms=end / 24,
                           wav=f'{scene["id"]}/utterances/{index:02d}_{turn["role"]}.wav'))
        pieces.append((turn['role'], start, pcm, clip))
        receipts.append(dict(turn_index=index, **receipt))
    frames = math.ceil((end + round(scene['trailing_observation_s'] * 24000)) / 1920) * 1920
    if not 0 < frames <= 90 * 24000:
        raise ValueError('Source exceeds the existing bounded90-second teacher input')
    stems = {role: np.zeros(frames, dtype=np.float32) for role in ('assistant', 'user')}
    for role, start, pcm, _ in pieces:
        stems[role][start:start + len(pcm)] = pcm
    quiet_start = events[scene['quiet_start_after_turn']]['actual_end_sample']
    release = scene['release_turn']
    quiet_end = frames if release is None else events[release]['actual_end_sample']
    if quiet_end <= quiet_start or np.count_nonzero(stems['assistant'][quiet_start:quiet_end]):
        raise ValueError('Source assistant violates the full quiet contract')
    folder.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    (folder / 'utterances').mkdir()
    for _, _, _, original in pieces:
        # Keep the provider WAV and receipt byte for byte; derived stems are separate.
        for source in (original, original.with_suffix('.receipt.json')):
            destination = folder / 'utterances' / source.name
            destination.write_bytes(source.read_bytes())
            artifacts[str(destination.relative_to(folder))] = sha256(destination)
    for name, pcm in [('stem_user.wav', stems['user']), ('stem_assistant.wav', stems['assistant']),
                      ('conversation.wav', np.stack([stems['assistant'], stems['user']], axis=1))]:
        sf.write(folder / name, pcm, 24000, subtype='FLOAT')
        artifacts[name] = sha256(folder / name)
    timeline = dict(scenario_id=scene['id'], scenario_class='native_control3_raw_source',
                    events=events, meta=dict(scene, synthetic=True, provenance_class='closed-content',
                                            plan_sha256=plan_sha256, source_clip_receipts=receipts,
                                            verified=False, training_approved=False),
                    quiet_contract=dict(start_sample=quiet_start, end_sample=quiet_end,
                                        assistant_nonzero_samples=0,
                                        release_is_later_user_turn=release is not None,
                                        terminal_no_release=release is None))
    timeline_path = folder / 'timeline.actual.json'
    timeline_path.write_text(json.dumps(timeline, indent=2) + '\n')
    artifacts[timeline_path.name] = sha256(timeline_path)
    return dict(id=scene['id'], split='train', frames=frames, sample_rate=24000,
                artifacts=artifacts, quiet_contract=timeline['quiet_contract'],
                verified=False, training_approved=False)


def render(plan_path, clips, out):
    plan = json.loads(plan_path.read_text())
    summary = validate(plan)
    if plan != build():
        raise ValueError('Only the exact independently authored source version is supported')
    if out.exists():
        raise ValueError('Use a NEW immutable raw-source directory, including after failure')
    out.mkdir(parents=True)
    plan_hash = sha256(plan_path)
    (out / 'source-plan.json').write_bytes(plan_path.read_bytes())
    rows = [assemble_scene(scene, clips, out / scene['id'], plan_hash) for scene in plan['cases']]
    result = dict(schema='dl-native-control3-raw-source-v2', plan_sha256=plan_hash,
                  implementation_sha256=sha256(Path(__file__)),
                  complete40=True, source_coverage=summary, cases=rows,
                  input_clips_unchanged=True, gain_changed=False, resampled=False,
                  verified=False, training_approved=False,
                  required_next='Independent source word/timestamp/audio qualification of ALL40, actual teacher '
                                'capture, independent quality/semantic, conversion and numerical approval')
    (out / 'raw-source.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def validate_raw(root):
    """Rebuild every raw source from its portable clips; never trust a pass flag."""
    import numpy as np
    import soundfile as sf
    root = Path(root)
    manifest = json.loads((root / 'raw-source.json').read_text())
    plan = json.loads((root / 'source-plan.json').read_text())
    summary = validate(plan)
    if (plan != build() or manifest['schema'] != 'dl-native-control3-raw-source-v2'
            or manifest['plan_sha256'] != sha256(root / 'source-plan.json')
            or manifest['source_coverage'] != summary or manifest['complete40'] is not True
            or manifest['verified'] is not False or manifest['training_approved'] is not False
            or manifest['gain_changed'] is not False or manifest['resampled'] is not False
            or manifest['input_clips_unchanged'] is not True
            or [r['id'] for r in manifest['cases']] != [s['id'] for s in plan['cases']]):
        raise ValueError('Need the exact complete40 unapproved raw-source version')
    for scene, row in zip(plan['cases'], manifest['cases']):
        folder = root / scene['id']
        if type(row['frames']) is not int or not 0 < row['frames'] <= 90 * 24000:
            raise ValueError('Invalid bounded raw-source length')
        expected_names = {'timeline.actual.json', 'conversation.wav', 'stem_user.wav', 'stem_assistant.wav'}
        for i, turn in enumerate(scene['turns']):
            name = f'utterances/{i:02d}_{turn["role"]}.wav'
            expected_names.update((name, name[:-4] + '.receipt.json'))
        if set(row['artifacts']) != expected_names:
            raise ValueError('Missing or unexpected source artifact inventory')
        for name, digest in row['artifacts'].items():
            if sha256(folder / name) != digest:
                raise ValueError('Raw source artifact hash changed')
        timeline = json.loads((folder / 'timeline.actual.json').read_text())
        if (timeline['scenario_id'] != scene['id']
                or timeline['scenario_class'] != 'native_control3_raw_source'
                or len(timeline['events']) != len(scene['turns'])
                or row['split'] != 'train' or row['sample_rate'] != 24000
                or row['verified'] is not False or row['training_approved'] is not False):
            raise ValueError('Raw source scene/fold changed')
        stems = {role: np.zeros(row['frames'], dtype='float32') for role in ('user', 'assistant')}
        receipts, end = [], 0
        for i, (turn, event) in enumerate(zip(scene['turns'], timeline['events'])):
            relative = f'{scene["id"]}/utterances/{i:02d}_{turn["role"]}.wav'
            voice = scene['user_voice'] if turn['role'] == 'user' else scene['assistant_voice']
            pcm, receipt = load_clip(root / relative, dict(text=turn['text'], voice=voice))
            start = 19200 if i == 0 else math.ceil((end + round(turn['gap_before_s'] * 24000)) / 1920) * 1920
            end = start + len(pcm)
            expected = dict(ch=turn['role'], text=turn['text'], kind='utterance', source_turn_index=i,
                            actual_start_sample=start, actual_end_sample=end,
                            actual_start_ms=start / 24, actual_end_ms=end / 24, wav=relative)
            if event != expected:
                raise ValueError('Source words, schedule or portable clip path changed')
            stems[turn['role']][start:end] = pcm
            receipts.append(dict(turn_index=i, **receipt))
        frames = math.ceil((end + round(scene['trailing_observation_s'] * 24000)) / 1920) * 1920
        if frames != row['frames'] or timeline['meta'] != dict(
                scene, synthetic=True, provenance_class='closed-content',
                plan_sha256=manifest['plan_sha256'], source_clip_receipts=receipts,
                verified=False, training_approved=False):
            raise ValueError('Source duration, receipts or proposal state changed')
        for role in ('user', 'assistant'):
            observed, rate = sf.read(folder / f'stem_{role}.wav', dtype='float32')
            if rate != 24000 or not np.array_equal(observed, stems[role]):
                raise ValueError('Source stem differs from immutable original clips')
        stereo, rate = sf.read(folder / 'conversation.wav', dtype='float32', always_2d=True)
        if rate != 24000 or not np.array_equal(stereo, np.stack([stems['assistant'], stems['user']], axis=1)):
            raise ValueError('Source stereo differs from exact assistant/user stems')
        begin = timeline['events'][scene['quiet_start_after_turn']]['actual_end_sample']
        release = scene['release_turn']
        stop = frames if release is None else timeline['events'][release]['actual_end_sample']
        contract = dict(start_sample=begin, end_sample=stop, assistant_nonzero_samples=0,
                        release_is_later_user_turn=release is not None, terminal_no_release=release is None)
        if (timeline['quiet_contract'] != contract or row['quiet_contract'] != contract
                or stop <= begin or np.count_nonzero(stems['assistant'][begin:stop])):
            raise ValueError('Quiet interval or terminal observation changed')
    return manifest, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--clips', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = render(args.plan, args.clips, args.out)
    print(json.dumps({k: result[k] for k in ('complete40', 'gain_changed', 'verified', 'training_approved')}))


if __name__ == '__main__':
    main()
