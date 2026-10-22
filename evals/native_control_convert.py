"""Convert only the complete independently approved control40 native histories.

Source PCM and receipts remain immutable. Derived FLOAT stereo is explicitly
assistant-left/user-right, preserving every sample and adding only recorded
observation/grid-tail silence. This is data conversion, never optimizer approval.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from data.synthetic.native_control_render import sha256, validate_raw
from evals.native_control_semantic import verify as verify_semantic
from evals.native_control_teacher_reaudit import verify as verify_quality

SCHEMA = 'dl-native-control3-measured-conversion-v1'
FILES = ('stem_user.wav', 'stem_assistant.wav', 'conversation.wav', 'conversation.json', 'timeline.actual.json')


def padded_stems(user_path, assistant_path, observed_s):
    import numpy as np
    import soundfile as sf
    user, rate = sf.read(user_path, dtype='float32')
    assistant, assistant_rate = sf.read(assistant_path, dtype='float32')
    if (rate != 24000 or assistant_rate != rate or user.ndim != 1 or assistant.ndim != 1
            or not len(user) or not len(assistant) or not np.isfinite(user).all() or not np.isfinite(assistant).all()
            or not math.isfinite(observed_s) or observed_s < len(user) / rate
            or len(assistant) / rate > observed_s + .08
            or float(np.abs(user).max()) > .995 or float(np.abs(assistant).max()) > .995):
        raise ValueError('Complete valid unclipped original PCM and session observation required')
    frames = math.ceil(max(len(user), len(assistant), math.ceil(observed_s * rate)) / 1920) * 1920
    return (np.pad(user, (0, frames - len(user))), np.pad(assistant, (0, frames - len(assistant))),
            dict(sample_rate=rate, frames=frames, source_user_samples=len(user), source_assistant_samples=len(assistant),
                 observed_until_s=observed_s, observed_end_grid_tail_samples=frames - math.ceil(observed_s * rate),
                 channels=['assistant', 'user'], original_pcm_values_changed=False,
                 resampled=False, gain_changed=False, internal_silence_removed=False))


def measured_timeline(scene, source, source_checks, teacher):
    from data.synthetic.timeline import Event, Timeline
    from data.synthetic.timing_audit import grid_words_by_event
    sid = scene['id']
    events, user_words = [], []
    for original, check in zip(source['events'], source_checks['clips']):
        if original['ch'] != 'user':
            continue
        offset = original['actual_start_sample'] / 24000
        user_words.extend(dict(text=w['text'].strip(), start_s=offset + w['start'], end_s=offset + w['end'])
                          for w in check['measured_words'])
        events.append(Event('user', original['actual_start_ms'], original['text'],
                            actual_start_ms=original['actual_start_ms'], actual_end_ms=original['actual_end_ms'],
                            note='Original sealed user input PCM clock; actual send availability separately retained'))
    words = [w for segment in teacher['speech']['asr_segments'] for w in segment['words']]
    phases = teacher['checks']['answer_phases']
    covered = []
    for phase in phases:
        selected = [w for w in words if phase['start_s'] <= w['start'] and w['end'] <= phase['end_s']]
        if not phase['passed'] or not selected:
            raise ValueError('Complete approved measured answer phases required')
        covered.extend(selected)
        start, end = round(selected[0]['start'] * 1000), round(selected[-1]['end'] * 1000)
        events.append(Event('assistant', start, ' '.join(w['text'].strip() for w in selected),
                            actual_start_ms=start, actual_end_ms=end,
                            word_onsets_ms=[round(w['start'] * 1000) - start for w in selected],
                            note='Independent measured native teacher answer phase; no output forced'))
    if covered != words:
        raise ValueError('Measured conversion omitted or duplicated assistant words')
    timeline = Timeline(sid, 'native_control3_teacher', sorted(events, key=lambda e: e.actual_start_ms),
                        dict(split='train', family=scene['family'], source_model='gpt-live-1',
                             teacher_reference_conditioning='authored-control3-TRAIN-only',
                             quiet_start_s=teacher['checks']['quiet_start_s'], quiet_end_s=teacher['checks']['quiet_end_s'],
                             quiet_active_20ms_frames=teacher['checks']['quiet_active_20ms_frames'],
                             source_user_clock='Original actually submitted sequential PCM',
                             assistant_clock='Original packet availability and unbuffered playout',
                             actual_input_packet_availability_required=True,
                             no_external_requests=True, teacher_audio=True, verified=False))
    _, grid = grid_words_by_event(timeline)
    if len(grid) != len(words) or [text for _, _, text in grid] != [w['text'].strip() for w in words]:
        raise ValueError('Measured answer words lost or reordered on native80ms grid')
    maximum = max(abs(tick * 80 - word['start'] * 1000) for (_, tick, _), word in zip(grid, words))
    if maximum > 200:
        raise ValueError('Measured word grid exceeds unchanged200ms timing bar')
    side = dict(alignments=[[w['text'].strip(), [w['start'], w['end']], 'SPEAKER_MAIN'] for w in words],
                user_words=user_words, reference_arrivals=[], timeline=timeline.to_dict(),
                actual_input_packet_availability=teacher['actual_input_packet_availability'],
                original_model_transcript=teacher['speech']['model_text'],
                independent_asr_text=teacher['speech']['asr_text'])
    return timeline, side, maximum


def approvals(raw, source_audit, batch, quality, semantic):
    manifest = json.loads((batch / 'teacher.json').read_text())
    if 'repair_followup' in manifest:
        from evals.native_control_teacher_followup import verify_followup
        verify_followup(raw, source_audit, batch, manifest)
    verify_semantic(raw, source_audit, batch, quality, semantic)
    report = verify_quality(raw, source_audit, batch, quality)
    source, plan = validate_raw(raw)
    if len(report['results']) != 40 or len(plan['cases']) != 40:
        raise ValueError('Complete40 quality and semantic approval required')
    return source, plan, report


def derived(raw, source_audit, batch, source_row, scene, teacher):
    from evals.native_control_teacher_artifacts import input_availability
    folder = batch / scene['id']
    session = json.loads((folder / 'report.json').read_text())
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    teacher = dict(teacher, actual_input_packet_availability=input_availability(events, session['sent_input_samples']))
    checks = json.loads(source_audit.read_text())
    source_checks = next(r for r in checks['results'] if r['id'] == scene['id'])
    source_timeline = json.loads((raw / scene['id'] / 'timeline.actual.json').read_text())
    timeline, side, maximum = measured_timeline(scene, source_timeline, source_checks, teacher)
    user, assistant, clock = padded_stems(folder / 'user_in.wav', folder / 'model_out.wav', session['observed_until_s'])
    side['pcm_clock'] = clock
    return timeline, side, maximum, user, assistant, clock


def convert(raw, source_audit, batch, quality, semantic, out):
    import numpy as np
    import soundfile as sf
    source, plan, report = approvals(raw, source_audit, batch, quality, semantic)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for source_row, scene, teacher in zip(source['cases'], plan['cases'], report['results']):
        folder = out / scene['id']
        folder.mkdir()
        timeline, side, maximum, user, assistant, clock = derived(raw, source_audit, batch, source_row, scene, teacher)
        for name, pcm in (('stem_user.wav', user), ('stem_assistant.wav', assistant),
                          ('conversation.wav', np.stack([assistant, user], axis=1))):
            sf.write(folder / name, pcm, 24000, subtype='FLOAT')
        timeline.save(folder / 'timeline.actual.json')
        (folder / 'conversation.json').write_text(json.dumps(side, indent=2) + '\n')
        rows.append(dict(scenario_id=scene['id'], dir=scene['id'], split='train', family=scene['family'],
                         scenario_class='native_control3_teacher', path=str(folder / 'conversation.wav'),
                         frames=clock['frames'], duration=clock['frames'] / 24000, max_grid_offset_ms=maximum,
                         artifacts={name: sha256(folder / name) for name in FILES},
                         training_approved=False, native_numerical_approval_required=True))
    for fold in ('train', 'validation', 'test'):
        (out / (fold + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows if r['split'] == fold))
    receipt = dict(schema=SCHEMA, at_utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha256(Path(__file__)),
                   raw=str(raw), source_audit=str(source_audit), batch=str(batch), quality=str(quality), semantic=str(semantic),
                   source_raw_manifest_sha256=sha256(raw / 'raw-source.json'), source_audit_sha256=sha256(source_audit),
                   teacher_manifest_sha256=sha256(batch / 'teacher.json'), quality_sha256=sha256(quality),
                   semantic_sha256=sha256(semantic), accepted=rows, complete40=True, optimizer_updates=0,
                   training_approved=False, native_numerical_approval_required=True,
                   original_data_retention_losses_or_heldout_changed=False,
                   channel_order=['assistant', 'user'])
    (out / 'ingestion.json').write_text(json.dumps(receipt, indent=2) + '\n')
    verify(out)
    return receipt


def verify(data):
    import numpy as np
    import soundfile as sf
    receipt = json.loads((data / 'ingestion.json').read_text())
    raw, source_audit, batch, quality, semantic = (Path(receipt[k]) for k in ('raw', 'source_audit', 'batch', 'quality', 'semantic'))
    source, plan, report = approvals(raw, source_audit, batch, quality, semantic)
    if (receipt['schema'] != SCHEMA or receipt['code_sha256'] != sha256(Path(__file__))
            or receipt['complete40'] is not True or len(receipt['accepted']) != 40
            or receipt['optimizer_updates'] != 0 or receipt['training_approved'] is not False
            or receipt['native_numerical_approval_required'] is not True
            or receipt['original_data_retention_losses_or_heldout_changed'] is not False
            or receipt['channel_order'] != ['assistant', 'user']
            or receipt['source_raw_manifest_sha256'] != sha256(raw / 'raw-source.json')
            or receipt['source_audit_sha256'] != sha256(source_audit)
            or receipt['teacher_manifest_sha256'] != sha256(batch / 'teacher.json')
            or receipt['quality_sha256'] != sha256(quality) or receipt['semantic_sha256'] != sha256(semantic)):
        raise ValueError('Complete unchanged40 source/quality/semantic conversion required')
    for row, source_row, scene, teacher in zip(receipt['accepted'], source['cases'], plan['cases'], report['results']):
        folder = data / scene['id']
        timeline, side, maximum, user, assistant, clock = derived(raw, source_audit, batch, source_row, scene, teacher)
        expected = dict(scenario_id=scene['id'], dir=scene['id'], split='train', family=scene['family'],
                        scenario_class='native_control3_teacher', path=str(folder / 'conversation.wav'),
                        frames=clock['frames'], duration=clock['frames'] / 24000, max_grid_offset_ms=maximum,
                        artifacts={name: sha256(folder / name) for name in FILES},
                        training_approved=False, native_numerical_approval_required=True)
        if row != expected or json.loads((folder / 'conversation.json').read_text()) != side or json.loads((folder / 'timeline.actual.json').read_text()) != timeline.to_dict():
            raise ValueError('Converted full history, fold, word timing or receipt changed')
        for name, wanted in (('stem_user.wav', user), ('stem_assistant.wav', assistant),
                             ('conversation.wav', np.stack([assistant, user], axis=1))):
            actual, rate = sf.read(folder / name, dtype='float32')
            if rate != 24000 or not np.array_equal(actual, wanted):
                raise ValueError('Original PCM was changed, truncated or assigned to the wrong channel')
    for fold in ('train', 'validation', 'test'):
        rows = [json.loads(line) for line in (data / (fold + '.jsonl')).read_text().splitlines()]
        if rows != [r for r in receipt['accepted'] if r['split'] == fold]:
            raise ValueError('Complete40 TRAIN-only fold inventory changed')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'source-audit', 'batch', 'quality', 'semantic', 'out', 'verify'):
        ap.add_argument('--' + name, type=Path)
    args = ap.parse_args()
    if args.verify:
        result = verify(args.verify)
    elif all(getattr(args, name) for name in ('raw', 'source_audit', 'batch', 'quality', 'semantic', 'out')):
        result = convert(args.raw, args.source_audit, args.batch, args.quality, args.semantic, args.out)
    else:
        ap.error('Use --verify or all complete quality/semantic conversion inputs')
    print(json.dumps(dict(complete=len(result['accepted']), training_approved=False, optimizer_updates=0)))


if __name__ == '__main__':
    main()
