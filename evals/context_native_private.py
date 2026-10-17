"""Retain the exact late private target, replacing only future audio conditioning."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def plan(late, examples):
    import torch
    if (late.get('target_kind') != 'private_result'
            or late.get('conditioning_policy') != 'verified-trained-rollout'
            or not late.get('actual_private_receipt')):
        raise RuntimeError('Own-audio private continuation needs its verified trained TRAIN receipt')
    history = Path(late['actual_history'])
    record = json.loads((history/'student-history.json').read_text())
    if record['source']['split'] != 'train' or record['source']['kind'] != 'private':
        raise RuntimeError('Own-audio private continuation excludes development and other task kinds')
    example = next(e for e in examples if e['record'] == late['teacher'])
    left, right = late['selection']['source_start_frame'], late['selection']['source_end_frame']
    gold = example['clean_codes'][:9, left:right].clone()
    begin = late['future_first_real_frame']; end = begin+late['future_frames']
    if (gold.shape[1] != end-begin or begin != late['replay_cut_index']+3
            or int(gold[0,0]) not in (0,3) or (gold[0]==4).any()):
        raise RuntimeError('Own-audio private sentence changed its approved cue, spacing or length')
    rows = [json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    if end > len(rows) or any(rows[j+1]['history_virtual_frames_before_step'] !=
            rows[j]['history_virtual_frames_after_step'] for j in range(begin-1,end-1)):
        raise RuntimeError('Late private continuation must not cross any new context span')
    saved = torch.load(history/'student-history.pt', map_location='cpu', weights_only=True)
    state = saved['states'][begin-1]; position = (state['offset']-1)%state['cache'].shape[2]
    cue = int(state['cache'][0,0,position])
    if (cue not in (0,3) or bool(state['provided'][0,0,position])
            or state['pending_exit_cb0'] is not None):
        raise RuntimeError('Late private prefix lacks its sampled quiet cue')
    return {**late, 'schema':'dl-context-early-future-plan-v1',
        'native_late_private_version':1, 'native_private_plan_code_sha256':sha256(Path(__file__)),
        'future_end_real_frame':end, 'future_real_frames':end-begin,
        'source_start_frame':left, 'source_end_frame':right,
        'actual_prefix_last_predicted_text_token':cue,
        'future_recorded_receipts':[], 'future_receipts_replayed_not_dispatched':False,
        'past_loss_targets':0, 'audio_loss_targets':0, 'user_loss_targets':0,
        'teacher_word_times_retimed':True, 'measured_word_spacing_retained':True,
        'user_pcm_and_word_times_retimed':False,
        'composition':'same approved late private sentence and actual trained prefix; future assistant audio sampled natively; actual receipt remains in past'}, gold
