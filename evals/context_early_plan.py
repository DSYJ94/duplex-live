"""Plan an approved full continuation before any student reply or request token.

This is an offline composition plan, not a live dispatch or training approval.
User PCM and teacher word times retain their original absolute real-frame clock.
"""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def plan(late, examples):
    from evals.recovery_data import validate_conversion
    history = Path(late['actual_history'])
    record = json.loads((history/'student-history.json').read_text())
    rows = [json.loads(line) for line in (history/'events.jsonl').read_text().splitlines()]
    example = next(e for e in examples if e['record'] == late['teacher'])
    clean = example['clean_codes']
    actual_first = next(i for i,r in enumerate(rows) if r['text_token'] >= 4)
    teacher_first = int((clean[0]>=4).nonzero()[0])
    begin = min(actual_first,teacher_first)-2
    # Restart checks through cut+2; no old token is replaced by teacher gold.
    cut = begin-3
    if cut < 0 or any(r['text_token'] not in (0,3) for r in rows[:begin]):
        raise RuntimeError('No early actual cut before every sampled reply/request token')
    if (clean[0,:begin]>=4).any():
        raise RuntimeError('Early window would omit an approved teacher reply/request token')
    if any(abs(row['input_start_s']-i*.08)>1e-7 for i,row in enumerate(rows)):
        raise RuntimeError('Early composition needs the unchanged absolute real-frame clock')
    teacher = late['teacher']; cache = Path(teacher['cache'])
    receipt = json.loads((cache/'cache.json').read_text())
    source = next(r for r in validate_conversion(Path(receipt['conversion']))['accepted']
        if r['scenario_id'] == teacher['id'])
    side = json.loads(Path(source['path']).with_suffix('.json').read_text())
    words = [a for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    last_word = int((clean[0]>=5).nonzero()[-1])
    end = max(last_word+1,math.ceil(max(w[1][1] for w in words)*12.5))+2
    if end > min(len(rows),clean.shape[1]):
        raise RuntimeError('Complete approved teacher reply exceeds the recorded actual input; no silent crop')
    if (clean[:9,begin:end]<0).any():
        raise RuntimeError('Early continuation includes an invalid assistant target')
    events = json.loads((history/'worker-events.json').read_text())
    spans = []
    for event in events:
        if event.get('applied_at_s') is None or 'virtual_span_frames' not in event:continue
        boundary = next(i for i,r in enumerate(rows)
            if r['history_virtual_frames_before_step'] == event['history_virtual_frames_after_span'])
        if boundary < begin:
            raise RuntimeError('Early prefix unexpectedly contains an already delivered backend span')
        if boundary >= end:continue
        spans.append({'real_frame':boundary,'event':event})
    private = [s for s in spans if s['event'] == late['actual_private_receipt']]
    if len(private) != 1 or late['selection']['source_start_frame'] <= private[0]['real_frame']:
        raise RuntimeError('Approved private sentence would start before its actual delivered context')
    # No marker was consumed in the actual prefix. A target request must still
    # precede the provided private receipt on this newly composed future clock.
    requests = (clean[0,begin:private[0]['real_frame']] == 4).nonzero()
    if not len(requests):
        raise RuntimeError('No approved future request precedes the actual private receipt')
    result = {'schema':'dl-context-early-future-plan-v1',
        'actual_history':str(history),'actual_history_sha256':record['sha256'],
        'actual_history_manifest_sha256':sha256(history/'student-history.json'),
        'actual_report_sha256':sha256(history/'report.json'),
        'teacher':teacher,'world':late['world'],'actual_private_receipt':late['actual_private_receipt'],
        'replay_cut_index':cut,'future_first_real_frame':begin,'future_end_real_frame':end,
        'future_real_frames':end-begin,'source_start_frame':begin,'source_end_frame':end,
        'actual_prefix_has_no_reply_or_request_tokens':True,
        'all_teacher_reply_and_request_tokens_retained':True,
        'user_pcm_and_word_times_retimed':False,'teacher_word_times_retimed':False,
        'future_user_pcm_is_allowed_to_contain_real_speech':True,
        'future_recorded_receipts':spans,'future_receipts_replayed_not_dispatched':True,
        'first_private_sentence_source_frame':late['selection']['source_start_frame'],
        'past_loss_targets':0,'audio_loss_targets':0,'gold_targets_created':False,
        'training_approved':False,'free_running_delegation_qualified':False,
        'composition':'actual silent prefix, approved teacher future on original real clock, observed source user PCM and actual receipt boundaries'}
    return result,clean[:9,begin:end].clone()
