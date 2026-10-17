"""Retain a complete approved ordinary answer at an observed native speaking cue."""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def boundary(rows, text, source_start, source_end, last_user, first_onset):
    if (not 0<=source_start<source_end<=len(text) or source_end-source_start<2
            or not all(math.isfinite(t) and t>=0 for t in (last_user,first_onset))
            or any(t>=4 for t in text[:source_start])
            or any(t>=4 for t in text[source_end:])):
        raise RuntimeError('Response composition must retain the complete finite-clock teacher reply')
    first=next((i for i,r in enumerate(rows) if r['text_token']>=4),None)
    if (first is None or first<4 or rows[first]['text_token']!=4
            or rows[first-1]['text_token']!=0
            or any(r['text_token'] not in (0,3) for r in rows[:first-1])
            or any(abs(r['input_start_s']-i*.08)>1e-7 for i,r in enumerate(rows))):
        raise RuntimeError('No observed EPAD/RET response boundary before all sampled words')
    begin=first-1;end=begin+source_end-source_start;delta=begin-source_start
    if (source_start<0 or source_end>len(text) or text[source_start]!=0
            or text[source_start+1]<5 or any(t==4 for t in text)
            or begin*.08<last_user or first_onset+delta*.08<last_user+.24
            or end>len(rows) or end<=begin):
        raise RuntimeError('Complete retimed answer would precede arrived speech or exceed recorded input')
    return begin,end,delta


def plan(late,examples):
    from evals.recovery_data import validate_conversion
    if (late.get('target_kind')!='ordinary_answer' or late['world']!={}
            or late['actual_private_receipt'] is not None):
        raise RuntimeError('Response-boundary composition is only for separately approved ordinary TRAIN answers')
    history=Path(late['actual_history']);record=json.loads((history/'student-history.json').read_text())
    report=json.loads((history/'report.json').read_text())
    if (record['source']['split']!='train' or record['source']['kind']!='simple'
            or report.get('backend_execution_disabled') is not True
            or json.loads((history/'worker-events.json').read_text())):
        raise RuntimeError('Response-boundary future cannot remove actual backend context or use held-out input')
    example=next(e for e in examples if e['record']==late['teacher']);clean=example['clean_codes']
    cache=json.loads((Path(late['teacher']['cache'])/'cache.json').read_text())
    source=next(r for r in validate_conversion(Path(cache['conversion']))['accepted'] if r['scenario_id']==late['teacher']['id'])
    side=json.loads(Path(source['path']).with_suffix('.json').read_text())
    words=[a for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    rows=[json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    selection=late['selection'];left,right=selection['source_start_frame'],selection['source_end_frame']
    onset=min(w[1][0] for w in words);last_user=max(w['end_s'] for w in side['user_words'])
    begin,end,delta=boundary(rows,clean[0].tolist(),left,right,last_user,onset)
    if (clean[:9,left:right]<0).any():raise RuntimeError('Retimed ordinary answer has invalid native targets')
    result={'schema':'dl-context-early-future-plan-v1','target_kind':'ordinary_answer',
        'actual_history':str(history),'actual_history_sha256':record['sha256'],
        'actual_history_manifest_sha256':sha256(history/'student-history.json'),
        'actual_report_sha256':sha256(history/'report.json'),'teacher':late['teacher'],
        'world':{},'actual_private_receipt':None,'replay_cut_index':begin-3,
        'future_first_real_frame':begin,'future_end_real_frame':end,'future_real_frames':end-begin,
        'source_start_frame':left,'source_end_frame':right,
        'actual_prefix_has_no_reply_or_request_tokens':True,
        'source_first_assistant_measured_onset_s':onset,
        'target_first_assistant_onset_s':onset+delta*.08,'retiming_offset_frames':delta,
        'all_teacher_reply_and_request_tokens_retained':True,
        'user_pcm_and_word_times_retimed':False,'teacher_word_times_retimed':True,
        'measured_word_spacing_retained':True,'native_response_boundary':True,
        'response_boundary_plan_code_sha256':sha256(Path(__file__)),
        'ordinary_plan_code_sha256':late['ordinary_plan_code_sha256'],
        'ordinary_answer_without_backend_context':True,'future_user_pcm_is_allowed_to_contain_real_speech':True,
        'future_recorded_receipts':[],'future_receipts_replayed_not_dispatched':False,
        'past_loss_targets':0,'audio_loss_targets':0,'gold_targets_created':False,
        'training_approved':False,'free_running_delegation_qualified':False,
        'composition':'complete measured TRAIN ordinary reply retimed to observed EPAD/RET speaking boundary; no backend context or altered past'}
    return result,clean[:9,left:right].clone()
