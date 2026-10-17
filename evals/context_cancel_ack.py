"""Select only a measured completion acknowledgment after an actual cancellation.

The observed model's words remain conditioning, never labels. This late correction
does not approve its preceding denial or claim the first acknowledgment improved.
"""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def completed_receipt(events):
    lookups={r['task_id']:e['applied_at_s'] for e in events if e.get('event')=='backend_result_ready'
        and e.get('applied_at_s') is not None and e.get('result',{}).get('route')=='delegate'
        for r in e['result']['receipts'] if r.get('tool')=='lookup_record' and r.get('executed') is True}
    candidates=[]
    for e in events:
        result=e.get('result',{});applied=e.get('applied_at_s')
        if (e.get('event')!='backend_result_ready' or result.get('route')!='cancel'
                or applied is None or not math.isfinite(applied) or applied<0
                or e.get('disposition')!='actual backend text injected'):
            continue
        receipts=result.get('receipts',[])
        if len(receipts)!=1:continue
        r=receipts[0];nested=r.get('receipts',[]);ids=r.get('task_ids',[])
        if (r.get('tool')!='cancel_lookup' or r.get('cancelled') is not True or not ids
                or len(ids)!=len(set(ids)) or not set(ids)<=set(lookups)
                or any(not math.isfinite(lookups[t]) or lookups[t]>=applied for t in ids)
                or len(nested)!=len(ids) or {n.get('task_id') for n in nested}!=set(ids)
                or any(n.get('tool')!='cancel_lookup' or n.get('cancelled') is not True for n in nested)):
            continue
        candidates.append(e)
    if len(candidates)!=1:raise RuntimeError('Completion words require one actually delivered successful cancellation')
    return candidates[0]


def teacher_scope(actual,record,teacher,side):
    from evals.recovery_prepare_data import spoken_words
    if (actual['split']!='train' or actual['kind']!='cancel'
            or record['source']['split']!='train' or record['source']['kind']!='cancel'
            or teacher['kind']!='cancel' or teacher['source_split']!='train'
            or teacher['source_model']!='gpt-live-1' or teacher.get('request_policy')!='native-prelead-epad'):
        raise RuntimeError('Acknowledgment needs an independently approved same-user cancellation TRAIN teacher')
    arrivals=side['reference_arrivals']
    words=[a for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    if (len(arrivals)!=1 or not math.isfinite(arrivals[0]['ready_at_s'])
            or arrivals[0]['ready_at_s']<0
            or not any(w in ('canceled','cancelled') for w in spoken_words(arrivals[0]['text']))
            or len(words)<2 or spoken_words(' '.join(w[0] for w in words[-2:]))!=spoken_words("It's cancelled")
            or not math.isfinite(words[-2][1][0])
            or words[-2][1][0]<arrivals[0]['ready_at_s']+.24):
        raise RuntimeError('Only a complete measured terminal cancellation acknowledgment is eligible')


def plan(history,data,examples,spm,teacher_id=None):
    import numpy as np
    import soundfile as sf
    import torch
    from evals.recovery_data import validate_conversion
    from evals.context_actual_targets import matching_user_stimulus,sentence_target,captured_packets
    from evals.context_rollout_conditioning import verify
    record=json.loads((history/'student-history.json').read_text())
    report=json.loads((history/'report.json').read_text());events=json.loads((history/'worker-events.json').read_text())
    collector=verify(report['loading']);receipt=completed_receipt(events)
    actual=next(r for r in validate_conversion(data)['accepted'] if r['scenario_id']==record['source']['scenario'])
    side=json.loads(Path(actual['path']).with_suffix('.json').read_text())
    audio,rate=sf.read(actual['path'],dtype='float32',always_2d=True)
    world=record['source']['declared_counterfactual_world'];meta=side['timeline']['meta']
    if world!={meta['project']:{meta['field']:meta['phrase']}}:
        raise RuntimeError('Cancellation acknowledgment cannot change the source world')
    candidates=[]
    for example in examples:
        teacher=example['record']
        if teacher['kind']!='cancel' or teacher['source_model']!='gpt-live-1' or (teacher_id and teacher['id']!=teacher_id):continue
        cache=json.loads((Path(teacher['cache'])/'cache.json').read_text())
        source=next(r for r in validate_conversion(Path(cache['conversion']))['accepted'] if r['scenario_id']==teacher['id'])
        ts=json.loads(Path(source['path']).with_suffix('.json').read_text());tm=ts['timeline']['meta']
        teacher_audio,tr=sf.read(source['path'],dtype='float32',always_2d=True)
        if ((tm['project'],tm['field'],tm['phrase'])!=(meta['project'],meta['field'],meta['phrase'])
                or tr!=rate or not matching_user_stimulus(audio[:,1],teacher_audio[:,1],side['user_words'],ts['user_words'])):continue
        teacher_scope(actual,record,teacher,ts)
        selection=sentence_target(example['clean_codes'][0].tolist(),ts['alignments'],"It's cancelled",spm.encode,ts['reference_arrivals'][0]['ready_at_s'])
        if selection['source_end_word_index']!=len([a for a in ts['alignments'] if a[2]=='SPEAKER_MAIN']):
            raise RuntimeError('Completion acknowledgment must retain the complete final sentence')
        candidates.append((teacher['id'],example,selection))
    if len(candidates)!=1:raise RuntimeError('Need one unambiguous approved cancellation acknowledgment')
    _,example,selection=candidates[0];left,right=selection['source_start_frame'],selection['source_end_frame']
    gold=example['clean_codes'][:9,left:right].clone();length=gold.shape[1]
    if (gold<0).any() or (gold[0]==4).any():raise RuntimeError('Completion acknowledgment has an invalid target or new request')
    rows=[json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    wave,wr=sf.read(history/'model.wav',dtype='float32')
    if wr!=rate or sha256(history/'model.wav')!=report['audio_sha256']:raise RuntimeError('Actual speech recording changed')
    packets=captured_packets(wave,report['playout']['packets'],rows)
    quiet=[len(p)>0 and float(np.sqrt(np.mean(p*p)))<.005 and r['text_token'] in (0,3) for r,p in zip(rows,packets)]
    saved=torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)
    spans=[e for e in events if e.get('applied_at_s') is not None and 'virtual_span_frames' in e]
    delivered=max(e['applied_at_s'] for e in spans);last_user=max(w['end_s'] for w in side['user_words'])
    selected=None
    for i in range(len(rows)-length-3):
        begin=i+3;end=begin+length
        if (rows[i]['input_available_s']<max(delivered,last_user)+.24 or not all(quiet[i:begin])
                or saved['states'][i]['pending_exit_cb0'] is not None
                or any(rows[j+1]['history_virtual_frames_before_step']!=rows[j]['history_virtual_frames_after_step'] for j in range(i,end-1))):continue
        start=round(rows[begin]['input_start_s']*rate);stop=round(rows[end-1]['input_available_s']*rate)
        user=np.pad(audio[:,1],(0,20*rate))
        if len(user[start:stop])!=length*1920 or np.count_nonzero(user[start:stop]):continue
        selected=i;break
    if selected is None:raise RuntimeError('No quiet, fully arrived, receipt-complete native continuation boundary')
    result={'schema':'dl-context-actual-future-plan-v1','target_kind':'cancellation_acknowledgment',
        'actual_history':str(history),'actual_history_sha256':record['sha256'],
        'actual_history_manifest_sha256':sha256(history/'student-history.json'),'actual_report_sha256':sha256(history/'report.json'),
        'actual_generated_audio_sha256':report['audio_sha256'],
        'quiet_boundary_audio_source':'verified model.wav and completely checked packet ledger',
        'teacher':example['record'],'selection':selection,'matching_approved_teacher_count':1,'world':world,
        'actual_private_receipt':None,'actual_cancel_receipt':receipt,'trained_collector':collector,
        'cancel_ack_plan_code_sha256':sha256(Path(__file__)),'replay_cut_index':selected,
        'future_first_real_frame':selected+3,'future_frames':length,'all_past_labels_masked':True,
        'only_completed_cancel_acknowledgment_selected':True,'preceding_model_denial_is_conditioning_only':True,
        'first_acknowledgment_or_cancellation_timing_qualified':False,'user_loss_targets':0,'private_fact_targets':0,
        'composition':'complete measured terminal cancellation acknowledgment retimed to a quiet actual post-receipt boundary; spacing retained',
        'training_approved':False,'gold_targets_created':False}
    return result,gold
