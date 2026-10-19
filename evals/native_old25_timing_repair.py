"""Versioned timing-only repair; all 25 original native teachers remain mandatory.

Only the four SHA-bound diagnosed histories receive new word intervals. This
does not alter raw audio, words, tool receipts or the 28 student-history inputs.
The resulting conversion still needs native cache/input/numerical approval.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from evals.moshi_model_download import sha256
from evals.recovery_prepare_data import spoken_words

ROOT = Path('/projects/u6si/duplex-live')
PARENT = ROOT / 'runs/context-training-proof-7510331/report.json'
PARENT_SHA256 = '7f69a321b87613b44c7d5a19b783c3c6ddcbb479bb2fc7bb94e96fe210119778'
REPAIRED = frozenset(('teacher-cf-recovery-train-001-0',
                     'teacher-cf-recovery-train-002-0',
                     'teacher-cf-recovery-train-007-0',
                     'teacher-recovery-train-004'))
SCHEMA = 'dl-native-old25-timing-repair-v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def inventory():
    from evals.recovery_data import validate_cache, validate_conversion
    if sha256(PARENT) != PARENT_SHA256:
        raise RuntimeError('Frozen native520 parent changed')
    policy = read(PARENT)['retention_teacher_text']
    teachers = policy['teachers']
    if len(teachers) != 25 or len({t['id'] for t in teachers}) != 25:
        raise RuntimeError('Original native520 teacher inventory changed')
    result = []
    for teacher in teachers:
        cache = Path(teacher['cache'])
        if sha256(cache / 'cache.json') != teacher['cache_sha256']:
            raise RuntimeError('Original teacher cache changed')
        receipt = validate_cache(cache)
        converted = validate_conversion(Path(receipt['conversion']))
        row = next(r for r in converted['accepted'] if r['scenario_id'] == teacher['id'])
        origin = Path(row['path']).parent.parent / 'ingestion.json'
        original = validate_conversion(origin.parent)
        checks_path = Path(original['teacher_checks'])
        checks = read(checks_path)
        decision = next(r for r in checks['results'] if 'teacher-' + r['scenario_id'] == teacher['id'])
        if not decision['speech_and_fact_checks_pass']:
            raise RuntimeError('An original teacher was not approved')
        result.append(dict(teacher=teacher, original_row=row, origin=str(origin),
                           origin_sha256=sha256(origin), checks=str(checks_path),
                           checks_sha256=sha256(checks_path), decision=decision,
                           raw_batch=original['raw_batch']))
    if not REPAIRED.issubset({r['teacher']['id'] for r in result}):
        raise RuntimeError('Repair scope is not contained in the complete original 25')
    return result


def timing_only_words(original, measured):
    """Keep every original word verbatim; substitute independently measured times."""
    if (len(original) != len(measured) or not original
            or any(spoken_words(a['text']) != spoken_words(b['text'])
                   for a, b in zip(original, measured))):
        raise RuntimeError('Timing repair cannot add, omit, substitute or reorder original words')
    return [dict(a, start=b['start'], end=b['end']) for a, b in zip(original, measured)]


def review(items, speech_report):
    from evals.recovery_teacher_quality import assess, validate_teacher
    from evals.teacher_world import teacher_metadata
    fresh = {Path(r['run']).name: r for r in read(speech_report)['results']}
    expected = {r['decision']['raw_record']['dir'] for r in items if r['teacher']['id'] in REPAIRED}
    if len(fresh) != 4 or set(fresh) != expected:
        raise RuntimeError('Fresh measurement must cover exactly the four diagnosed recordings')
    reviewed = []
    for item in items:
        sid = item['teacher']['id']
        raw = Path(item['raw_batch'])
        manifest = validate_teacher(raw)
        prior = item['decision']
        record = prior['raw_record']
        folder = raw / record['dir']
        original_speech = read(Path(item['checks']).parent / 'report.json')
        speech = next(r for r in original_speech['results'] if Path(r['run']).name == record['dir'])
        if sid in REPAIRED:
            measured = fresh[record['dir']]
            if (measured['wav_sha256'] != record['audio_sha256']
                    or measured['wav_sha256'] != speech['wav_sha256']
                    or measured['model_text'] != speech['model_text']
                    or not measured.get('timestamps_restored_to_original_wav')
                    or measured['internal_silence_removed']
                    or not measured['word_starts_within_recorded_audio']
                    or not measured['asr_time_bounds_valid'] or not measured['asr_word_bounds_valid']):
                raise RuntimeError('Fresh timing lost immutable audio/text or original receipt clock')
            speech = measured
        timeline = read(Path(record['source_row']['path']).with_suffix('.json'))['timeline']
        metadata = dict(teacher_metadata(record, timeline['meta'], manifest['source_ingestion_sha256']),
                        source_events=timeline['events'])
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
        checked = assess(speech, events, metadata, item['teacher']['kind'], read(folder / 'report.json'))
        if checked['speech_and_fact_checks_pass']:
            checked['measured_words'] = timing_only_words(prior['measured_words'], checked['measured_words'])
            if sid not in REPAIRED and checked['measured_words'] != prior['measured_words']:
                raise RuntimeError('Undiagnosed teacher intervals changed')
        # Every original waveform also receives the physical-onset check, including unchanged 21.
        import numpy as np
        import soundfile as sf
        pcm, rate = sf.read(folder / 'model_out.wav', always_2d=True)
        nz = np.flatnonzero(pcm[:, 0])
        words = checked['measured_words']
        physical = bool(len(nz) and words and words[0]['start'] >= nz[0]/rate - .2)
        if not physical:
            checked['speech_and_fact_checks_pass'] = False
            checked['reasons'].append('Word onset precedes original nonzero PCM beyond 200ms')
        reviewed.append(dict(item, checked=checked, timing_repaired=sid in REPAIRED,
                             physical_onset_pass=physical))
    return dict(schema=SCHEMA, parent=str(PARENT), parent_sha256=sha256(PARENT),
                original_policy=read(PARENT)['retention_teacher_text'],
                fresh_speech_report=str(speech_report), fresh_speech_sha256=sha256(speech_report),
                repaired_ids=sorted(REPAIRED), cases=reviewed,
                passed=len(reviewed) == 25 and all(r['checked']['speech_and_fact_checks_pass'] for r in reviewed),
                optimizer_steps=0, training_approved=False, original_words_audio_receipts_preserved=True,
                actual_history_components_unchanged=28, reference_kl_weight=1., reference_hidden_mse_weight=10.,
                code_sha256=sha256(Path(__file__)))


def convert(approval_file, out):
    from evals.recovery_teacher_convert import measured_timeline
    approval = read(approval_file)
    expected = review(inventory(), Path(approval['fresh_speech_report']))
    if approval != expected or not approval['passed']:
        raise RuntimeError('All 25 timing/word/fact/receipt checks must pass before conversion')
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in approval['cases']:
        original = case['original_row']
        folder = out / original['scenario_id']
        shutil.copytree(Path(original['path']).parent, folder)
        if case['timing_repaired']:
            side = read(folder / 'conversation.json')
            words = case['checked']['measured_words']
            timeline, _ = measured_timeline(original['scenario_id'], words, side['timeline'],
                case['checked']['delegation_events'], case['checked']['reference_arrivals'])
            timeline.save(folder / 'timeline.actual.json')
            side['alignments'] = [[w['text'].strip(), [w['start'], w['end']], 'SPEAKER_MAIN'] for w in words]
            side['timeline'] = timeline.to_dict()
            write(folder / 'conversation.json', side)
        row = dict(original, path=str(folder / 'conversation.wav'), dir=folder.name)
        for name, key in (('conversation.json', 'sidecar_sha256'), ('timeline.actual.json', 'source_timeline_sha256')):
            row[key] = sha256(folder / name)
        rows.append(row)
    for fold in ('train', 'validation', 'test'):
        (out / (fold + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows if r['split'] == fold))
    receipt = dict(schema=SCHEMA, approval=str(approval_file), approval_sha256=sha256(approval_file),
                   accepted=rows, repaired_ids=sorted(REPAIRED), source_batch=str(out),
                   original_words_audio_receipts_preserved=True, optimizer_steps=0, training_approved=False)
    write(out / 'ingestion.json', receipt)
    validate_conversion(out, receipt)


def validate_conversion(data, receipt):
    """Full inventory and byte-level invariants; this schema never accepts a subset."""
    approval_path = Path(receipt['approval'])
    approval = read(approval_path)
    if (sha256(approval_path) != receipt['approval_sha256'] or not approval['passed']
            or approval != review(inventory(), Path(approval['fresh_speech_report']))
            or receipt['repaired_ids'] != sorted(REPAIRED)
            or len(receipt['accepted']) != 25
            or [r['scenario_id'] for r in receipt['accepted']] != [c['teacher']['id'] for c in approval['cases']]):
        raise RuntimeError('Repaired native teachers lack complete unchanged-parent approval')
    for row, case in zip(receipt['accepted'], approval['cases']):
        original = case['original_row']
        source = Path(original['path']).parent
        folder = data / row['dir']
        expected_row = dict(original, path=str(folder / 'conversation.wav'), dir=folder.name)
        for name, key in (('conversation.json', 'sidecar_sha256'), ('timeline.actual.json', 'source_timeline_sha256')):
            expected_row[key] = sha256(folder / name)
        if row != expected_row:
            raise RuntimeError('Timing repair changed an undeclared source/fold/receipt field')
        for name in ('conversation.wav', 'stem_user.wav', 'stem_assistant.wav'):
            if sha256(folder / name) != sha256(source / name):
                raise RuntimeError('Timing repair changed original audio')
        old = read(source / 'conversation.json')
        side = read(folder / 'conversation.json')
        if case['timing_repaired']:
            words = case['checked']['measured_words']
            if side['alignments'] != [[w['text'].strip(), [w['start'], w['end']], 'SPEAKER_MAIN'] for w in words]:
                raise RuntimeError('Repaired alignment differs from independent measurement')
            if ([w[0] for w in side['alignments']] != [w[0] for w in old['alignments']]
                    or side['user_words'] != old['user_words']
                    or side['reference_arrivals'] != old['reference_arrivals']):
                raise RuntimeError('Timing repair changed words/user inputs/worker receipts')
            from evals.recovery_teacher_convert import measured_timeline
            timeline, _ = measured_timeline(row['scenario_id'], words, old['timeline'],
                case['checked']['delegation_events'], case['checked']['reference_arrivals'])
            expected_side = dict(old, alignments=side['alignments'], timeline=timeline.to_dict())
            if side != expected_side or read(folder / 'timeline.actual.json') != timeline.to_dict():
                raise RuntimeError('Repaired timeline changed beyond measured timing')
        elif any(sha256(folder / name) != sha256(source / name)
                 for name in ('conversation.json', 'timeline.actual.json')):
            raise RuntimeError('An undiagnosed original teacher was relabelled')
    for fold in ('train', 'validation', 'test'):
        rows = [json.loads(line) for line in (data / (fold + '.jsonl')).read_text().splitlines()]
        if rows != [r for r in receipt['accepted'] if r['split'] == fold]:
            raise RuntimeError('Timing repair changed fold membership')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--convert-approval', type=Path)
    args = ap.parse_args()
    if args.convert_approval:
        convert(args.convert_approval, args.out)
        return
    items = inventory()
    args.out.mkdir(parents=True, exist_ok=False)
    runs = [str(Path(r['raw_batch']) / r['decision']['raw_record']['dir'])
            for r in items if r['teacher']['id'] in REPAIRED]
    subprocess.run([sys.executable, '-u', '-m', 'evals.moshi_speech_audit', '--out', str(args.out / 'speech'),
                    '--device', 'cpu', '--trim-leading-silence', '--trim-trailing-silence', '--live-runs', *runs], check=True)
    report = review(items, args.out / 'speech/report.json')
    write(args.out / 'report.json', report)
    print(json.dumps(dict(passed=report['passed'], total=len(report['cases']),
        failed=[r['teacher']['id'] for r in report['cases'] if not r['checked']['speech_and_fact_checks_pass']])), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
