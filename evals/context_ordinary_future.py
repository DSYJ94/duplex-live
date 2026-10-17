"""Approve a candidate ordinary reply without private receipts or backend hints.

This plans a full measured TRAIN teacher continuation. It creates no labels and
does not replace the independent raw-history, restart or component audits.
"""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def scope(record, report, events, actual, teacher, side):
    if (actual['split'] != 'train' or actual['kind'] != 'simple'
            or record['source']['split'] != 'train' or record['source']['kind'] != 'simple'
            or record['source']['declared_counterfactual_world'] != {}
            or report['source_split'] != 'train' or report.get('backend_execution_disabled') is not True
            or report.get('conditioning_only') is not True or report['training_approved'] is not False
            or report['native_foreground_backend_request'] is not False
            or events or report['worker_events']):
        raise RuntimeError('Ordinary targets require simple TRAIN observation without backend context')
    if (teacher['kind'] != 'simple' or teacher['source_split'] != 'train'
            or teacher['source_model'] != 'gpt-live-1' or side['reference_arrivals']
            or any(e['kind'] == 'marker' for e in side['timeline']['events'])):
        raise RuntimeError('Ordinary teacher must be approved TRAIN speech without backend results/requests')
    user = [e for e in side['timeline']['events'] if e['ch'] == 'user' and e['kind'] == 'utterance']
    words = [a for a in side['alignments'] if a[2] == 'SPEAKER_MAIN']
    last_user = max(w['end_s'] for w in side['user_words'])
    onset = min(w[1][0] for w in words)
    if len(user) != 1 or not math.isfinite(onset) or not math.isfinite(last_user) or onset < last_user:
        raise RuntimeError('Ordinary reply must follow the complete single user question')


def plan(history, data, examples, spm, teacher_id):
    import soundfile as sf
    from evals.recovery_data import validate_conversion
    from evals.context_actual_targets import matching_user_stimulus, captured_packets
    record = json.loads((history/'student-history.json').read_text())
    report = json.loads((history/'report.json').read_text())
    events = json.loads((history/'worker-events.json').read_text())
    actual = next(r for r in validate_conversion(data)['accepted']
        if r['scenario_id'] == record['source']['scenario'])
    source_side = json.loads(Path(actual['path']).with_suffix('.json').read_text())
    source_audio, rate = sf.read(actual['path'], dtype='float32', always_2d=True)
    candidates = []; conversions = {}
    for example in examples:
        r = example['record']
        if (r['kind'] != 'simple' or r['source_model'] != 'gpt-live-1'
                or (teacher_id is not None and r['id'] != teacher_id)):
            continue
        cache = Path(r['cache'])
        if cache not in conversions:
            receipt = json.loads((cache/'cache.json').read_text())
            conversions[cache] = {t['scenario_id']:t for t in
                validate_conversion(Path(receipt['conversion']))['accepted']}
        teacher = conversions[cache][r['id']]
        side = json.loads(Path(teacher['path']).with_suffix('.json').read_text())
        teacher_audio, teacher_rate = sf.read(teacher['path'], dtype='float32', always_2d=True)
        if teacher_rate != rate or not matching_user_stimulus(source_audio[:,1], teacher_audio[:,1],
                source_side['user_words'], side['user_words']):
            continue
        scope(record, report, events, actual, r, side)
        clean = example['clean_codes']; positions = (clean[0] >= 5).nonzero().flatten()
        if not len(positions) or (clean[0] == 4).any():
            raise RuntimeError('Ordinary teacher has no measured words or includes a retrieval target')
        words = [a for a in side['alignments'] if a[2] == 'SPEAKER_MAIN']
        left = int(positions[0])-1
        right = max(int(positions[-1])+1, math.ceil(max(w[1][1] for w in words)*12.5))+2
        if left < 0 or right > clean.shape[1] or (clean[:9,left:right] < 0).any():
            raise RuntimeError('Ordinary complete reply exceeds valid approved native targets')
        selection = {'source_start_frame':left, 'source_end_frame':right,
            'source_first_word_index':0, 'source_end_word_index':len(words)}
        candidates.append((r['id'], example, selection))
    if not candidates:
        raise RuntimeError('No approved ordinary teacher with identical user PCM and words')
    _, example, selection = sorted(candidates, key=lambda c:c[0])[0]
    rows = [json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    if sha256(history/'model.wav') != report['audio_sha256']:
        raise RuntimeError('Observed generated recording changed')
    audio, recording_rate = sf.read(history/'model.wav', dtype='float32')
    if recording_rate != rate:
        raise RuntimeError('Ordinary observation has a different audio clock')
    captured_packets(audio, report['playout']['packets'], rows)
    result = {'schema':'dl-context-actual-future-plan-v1', 'target_kind':'ordinary_answer',
        'ordinary_plan_code_sha256':sha256(Path(__file__)),
        'actual_history':str(history), 'actual_history_sha256':record['sha256'],
        'actual_history_manifest_sha256':sha256(history/'student-history.json'),
        'actual_report_sha256':sha256(history/'report.json'), 'teacher':example['record'],
        'selection':selection, 'matching_approved_teacher_count':len(candidates),
        'world':{}, 'actual_private_receipt':None, 'all_past_labels_masked':True,
        'training_approved':False, 'gold_targets_created':False}
    clean = example['clean_codes']
    return result, clean[:9,selection['source_start_frame']:selection['source_end_frame']].clone()
