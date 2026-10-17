"""Correct an observed trained model's first response without changing its past."""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def boundary(rows,text,left,right,last_user,onset,deadline=None):
    if (not 1<=left<right<=len(text) or text[left]<5 or text[left-1]!=0
            or not all(math.isfinite(v) and v>=0 for v in (last_user,onset))
            or any(t==4 for t in text[left:right])):
        raise RuntimeError('Trained response requires a complete independently selected word interval')
    first=next((i for i,r in enumerate(rows) if r['text_token']>=4),None)
    if (first is None or first<4 or rows[first-1]['text_token']!=0
            or any(r['text_token'] not in (0,3) for r in rows[:first-1])
            or any(abs(r['input_start_s']-i*.08)>1e-7 for i,r in enumerate(rows))):
        raise RuntimeError('No sampled trained EPAD followed by a first word or request')
    begin=first-1;end=begin+right-left;delta=begin-left
    if (end>len(rows) or begin*.08<last_user or onset+(delta+1)*.08<last_user+.24
            or (deadline is not None and (not math.isfinite(deadline) or end*.08+.24>deadline))):
        raise RuntimeError('Trained correction would precede arrived speech or cross its question boundary')
    return begin,end,delta,rows[first]['text_token']


def plan(late,examples):
    import torch
    from evals.recovery_data import validate_conversion
    from evals.context_rollout_conditioning import verify
    kind=late.get('target_kind')
    if kind not in ('ordinary_answer','clarification_question') or late['actual_private_receipt'] is not None:
        raise RuntimeError('Trained first-response correction is only ordinary speech or the first clarification question')
    history=Path(late['actual_history']);record=json.loads((history/'student-history.json').read_text())
    report=json.loads((history/'report.json').read_text());collector=verify(report['loading'])
    if (record['source']['split']!='train' or record['source']['kind']!=('simple' if kind=='ordinary_answer' else 'clarify')
            or report.get('backend_execution_disabled') is not True or report.get('conditioning_only') is not True
            or json.loads((history/'worker-events.json').read_text())):
        raise RuntimeError('Trained correction requires raw TRAIN observation with no backend context')
    example=next(e for e in examples if e['record']==late['teacher']);clean=example['clean_codes']
    cache=json.loads((Path(late['teacher']['cache'])/'cache.json').read_text())
    source=next(r for r in validate_conversion(Path(cache['conversion']))['accepted'] if r['scenario_id']==late['teacher']['id'])
    side=json.loads(Path(source['path']).with_suffix('.json').read_text())
    selection=late['selection'];left,right=selection['source_start_frame']+1,selection['source_end_frame']
    words=[a for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    selected=words[selection['source_first_word_index']:selection['source_end_word_index']]
    users=[e for e in side['timeline']['events'] if e['ch']=='user' and e['kind']=='utterance']
    question=kind=='clarification_question'
    if not question and (any(t>=4 for t in clean[0,:left].tolist()) or (clean[0,right:]>=4).any()):
        raise RuntimeError('Ordinary correction cannot discard any approved teacher words or requests')
    # The immediately preceding source EPAD is deliberately reused from the
    # actual cache, rather than regenerated. It is not an omitted spoken word.
    if not question and (clean[0,:left]>=5).any():raise RuntimeError('Ordinary correction omitted a word')
    initial_words=[w for w in side['user_words'] if w['end_s']<=users[0]['actual_end_ms']/1000+.04]
    last_user=max(w['end_s'] for w in (initial_words if question else side['user_words']))
    deadline=users[1]['actual_start_ms']/1000 if question else None
    rows=[json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    onset=min(w[1][0] for w in selected)
    begin,end,delta,replaced=boundary(rows,clean[0].tolist(),left,right,last_user,onset,deadline)
    saved=torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)
    for frame,token in [(begin-1,0),(begin,replaced)]:
        state=saved['states'][frame];position=(state['offset']-1)%state['cache'].shape[2]
        if (int(state['cache'][0,0,position])!=token or bool(state['provided'][0,0,position])
                or state['pending_exit_cb0'] is not None):raise RuntimeError('Trained boundary is not the actual sampled prediction cache')
    if (clean[:9,left:right]<0).any():raise RuntimeError('Trained correction has invalid native targets')
    details=({'clarification_plan_code_sha256':late['clarification_plan_code_sha256'],
        'clarification_question_without_backend_context':True,'question_only_selection':selection,
        'later_teacher_reply_and_request_tokens_deliberately_excluded':True,
        'question_retiming_before_project_verified':True,'question_deadline_s':deadline} if question else
        {'ordinary_plan_code_sha256':late['ordinary_plan_code_sha256'],'ordinary_answer_without_backend_context':True})
    result={'schema':'dl-context-early-future-plan-v1','target_kind':kind,
        'actual_history':str(history),'actual_history_sha256':record['sha256'],
        'actual_history_manifest_sha256':sha256(history/'student-history.json'),'actual_report_sha256':sha256(history/'report.json'),
        'teacher':late['teacher'],'world':late['world'],'actual_private_receipt':None,
        'trained_collector':collector,'conditioning_policy':'verified-trained-rollout',
        'replay_cut_index':begin-3,'future_first_real_frame':begin,'future_end_real_frame':end,'future_real_frames':end-begin,
        'source_start_frame':left,'source_end_frame':right,'actual_prefix_has_no_reply_or_request_tokens':True,
        'source_first_assistant_measured_onset_s':onset,'target_first_assistant_onset_s':onset+(delta+1)*.08,
        'retiming_offset_frames':delta,'native_response_boundary_version':3,'returned_text_delay_frames':1,
        'source_leading_epad_reused_from_actual_cache':True,'actual_boundary_cache_verified':True,
        'actual_replaced_prediction_token':replaced,'target_first_assistant_prediction_frame':begin,
        'target_first_assistant_readout_frame':begin+1,'all_teacher_reply_and_request_tokens_retained':not question,
        'user_pcm_and_word_times_retimed':False,'teacher_word_times_retimed':True,'measured_word_spacing_retained':True,
        'native_response_boundary':True,'trained_boundary_plan_code_sha256':sha256(Path(__file__)),
        'future_user_pcm_is_allowed_to_contain_real_speech':True,'future_recorded_receipts':[],
        'future_receipts_replayed_not_dispatched':False,'past_loss_targets':0,'audio_loss_targets':0,
        'user_loss_targets':0,'gold_targets_created':False,'training_approved':False,'free_running_delegation_qualified':False,
        'composition':'approved first TRAIN response replaces the trained model first word/request at its actual prediction step; cached EPAD reused; no changed past or backend',**details}
    return result,clean[:9,left:right].clone()
