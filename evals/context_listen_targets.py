"""Approve measured post-answer listening on actual TRAIN failure histories."""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def listen_window(actual, teacher, side, text):
    if (actual['split']!='train' or actual['kind']!='simple'
            or teacher['source_split']!='train' or teacher['kind']!='simple'
            or teacher['source_model']!='gpt-live-1' or side['reference_arrivals']):
        raise RuntimeError('Listening targets require approved simple TRAIN speech without receipts')
    users=[e for e in side['timeline']['events'] if e['ch']=='user' and e['kind']=='utterance']
    words=[a for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    if len(users)!=1 or not words:raise RuntimeError('Listening scope requires a complete single-question reply')
    last=max(w[1][1] for w in words)
    if not math.isfinite(last):raise RuntimeError('Invalid final measured word clock')
    begin=math.ceil(last*12.5)+5;end=begin+50
    if (end>len(text) or max(w['end_s'] for w in side['user_words'])+.24>begin*.08
            or any(t not in (0,3) for t in text[begin:end])):
        raise RuntimeError('Complete four-second listening window overlaps words, requests or future user speech')
    return begin,end


def plan(history,data,assets,examples,spm,voice,teacher_id,conditioning_policy):
    import numpy as np
    import soundfile as sf
    import torch
    from evals.recovery_data import validate_conversion
    from evals.context_history_validate import validate as validate_history
    from evals.context_actual_targets import matching_user_stimulus
    from evals.context_rollout_conditioning import verify
    if conditioning_policy!='verified-trained-rollout':raise RuntimeError('Listening corrections require an explicit trained-rollout policy')
    record=json.loads((history/'student-history.json').read_text())
    checked=validate_history(history,data,assets,record['source']['repo_commit'])
    loading=json.loads((history/'report.json').read_text())['loading'];collector=verify(loading)
    if record['source']['declared_counterfactual_world']!={} or record['execution']['math_sdp'] is not True:
        raise RuntimeError('Listening rollout must use the audited math protocol and no private world')
    actual=next(r for r in validate_conversion(data)['accepted'] if r['scenario_id']==checked['scenario'])
    side=json.loads(Path(actual['path']).with_suffix('.json').read_text())
    e=next(e for e in examples if e['record']['id']==teacher_id);teacher=e['record']
    cache=json.loads((Path(teacher['cache'])/'cache.json').read_text())
    approved=next(r for r in validate_conversion(Path(cache['conversion']))['accepted'] if r['scenario_id']==teacher_id)
    target_side=json.loads(Path(approved['path']).with_suffix('.json').read_text())
    source,rate=sf.read(actual['path'],dtype='float32',always_2d=True)
    gold,tr=sf.read(approved['path'],dtype='float32',always_2d=True)
    if (rate!=24000 or tr!=rate or not matching_user_stimulus(source[:,1],gold[:,1],side['user_words'],target_side['user_words'])):
        raise RuntimeError('Listening correction requires identical approved user PCM and words')
    begin,end=listen_window(actual,teacher,target_side,e['clean_codes'][0].tolist())
    window=gold[begin*1920:end*1920]
    if (len(window)!=(end-begin)*1920 or np.count_nonzero(window[:,1])
            or np.max(np.sqrt(np.mean(window[:,0].reshape(-1,1920)**2,axis=1)))>=.005):
        raise RuntimeError('Measured teacher listening interval is not quiet on both stems')
    saved=torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)
    if (not torch.equal(voice,torch.load(history/'student-voice.pt',map_location='cpu',weights_only=True)['codes'])
            or not torch.equal(voice,torch.load(Path(collector['proof'])/'voice.pt',map_location='cpu',weights_only=True)['codes'])):
        raise RuntimeError('Listening voice differs from the verified trained collector')
    rows=[json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    live=[(i,c) for i,c in enumerate(record['calls']) if c['mode']=='live'];chosen=live[begin:end]
    if end>len(rows) or len(chosen)!=50:raise RuntimeError('Listening interval exceeds the actual history')
    stop=chosen[0][1]['start']
    if (any(abs(row['input_start_s']-i*.08)>1e-7 for i,row in enumerate(rows))
            or any(c['frames']!=1 or c['start']!=stop+n for n,(_,c) in enumerate(chosen))
            or any(saved['states'][f]['pending_exit_cb0'] is not None for f in range(begin-1,end))):
        raise RuntimeError('Listening window contains a changed clock, span or pending span-exit readout')
    targets=e['clean_codes'][0,begin:end].clone()
    component={'delayed_inputs':saved['delayed_inputs'][:,:,stop:stop+50].clone(),
        'hidden':torch.cat([saved['temporal_outputs'][i]['hidden'] for i,_ in chosen],dim=1),
        'text_targets':targets,'text_mask':torch.ones(50,dtype=torch.bool)}
    manifest={'schema':'dl-context-action-plan-v1','target_kind':'return_to_listening',
        'conditioning_policy':conditioning_policy,'trained_collector':collector,'scenario':checked['scenario'],
        'actual_history':str(history),'source_history_sha256':checked['history_sha256'],
        'source_manifest_sha256':sha256(history/'student-history.json'),'source_report_sha256':checked['report_sha256'],
        'source_execution':record['execution'],'voice_codes_sha256':record['source']['voice_codes_sha256'],
        'teacher':teacher,'world':{},'future_first_real_frame':begin,'future_frames':50,'actual_prefix_frames':stop,
        'policy_code_sha256':sha256(Path(__file__)),'targets':targets.tolist(),
        'target_provenance':'approved measured TRAIN post-answer silence; observed assistant errors are conditioning only',
        'spoken_acknowledgement_targets':0,'private_fact_targets':0,
        'past_loss_targets':0,'span_loss_targets':0,'audio_loss_targets':0,'user_loss_targets':0,
        'training_approved':False,'free_running_delegation_qualified':False}
    return manifest,component
