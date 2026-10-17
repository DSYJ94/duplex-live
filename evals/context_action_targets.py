"""Plan approved cancellation controls on actual, unforced TRAIN conditioning."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def cancel_window(source, teacher, side, text):
    if (source['split']!='train' or source['kind']!='cancel'
            or teacher['source_split']!='train' or teacher['kind']!='cancel'
            or teacher['source_model']!='gpt-live-1'
            or teacher.get('request_policy')!='native-prelead-epad'):
        raise RuntimeError('Cancellation controls require approved TRAIN teacher actions')
    requests=teacher['derived_requests']
    if [r['operation'] for r in requests]!=['lookup','cancel']:
        raise RuntimeError('Cancellation teacher needs one lookup followed by one cancellation')
    frame=requests[1]['derived_request_frame']
    cancels=[e for e in side['timeline']['events'] if e['ch']=='user' and e['kind']=='utterance'
        and 'cancel' in e['text'].casefold()]
    if (len(cancels)!=1 or frame<2 or text[frame-2:frame+1]!=[3,0,4]
            or cancels[0]['actual_end_ms']/1000+.24>frame*.08
            or max(w['end_s'] for w in side['user_words'])+.24>frame*.08):
        raise RuntimeError('Cancellation must follow complete arrived user speech with native EPAD/RET cues')
    if any(a[2]=='SPEAKER_MAIN' and a[1][0]<(frame+1)*.08-1e-7
            and a[1][1]>(frame-2)*.08+1e-7 for a in side['alignments']):
        raise RuntimeError('Action-only labels cannot overwrite approved spoken words')
    return frame-2,frame+1


def plan(history,data,assets,examples,spm,voice,teacher_id):
    import torch
    import soundfile as sf
    from evals.recovery_data import validate_conversion
    from evals.context_history_validate import validate as validate_history
    from evals.context_actual_targets import matching_user_stimulus
    record=json.loads((history/'student-history.json').read_text())
    checked=validate_history(history,data,assets,record['source']['repo_commit'])
    loading=json.loads((history/'report.json').read_text())['loading']
    if (not loading['adapter'] or not loading['adapter']['zero'] or loading['adapter']['disabled']
            or not loading['layout'] or loading['input_warmup']):
        raise RuntimeError('Action data requires the verified zero-wrapper/layout protocol')
    original=next(r for r in validate_conversion(data)['accepted'] if r['scenario_id']==checked['scenario'])
    side=json.loads(Path(original['path']).with_suffix('.json').read_text())
    example=next(e for e in examples if e['record']['id']==teacher_id)
    teacher=example['record'];cache=Path(teacher['cache'])
    conversion=Path(json.loads((cache/'cache.json').read_text())['conversion'])
    approved=next(r for r in validate_conversion(conversion)['accepted'] if r['scenario_id']==teacher_id)
    target_side=json.loads(Path(approved['path']).with_suffix('.json').read_text())
    a,rate=sf.read(original['path'],dtype='float32',always_2d=True)
    b,teacher_rate=sf.read(approved['path'],dtype='float32',always_2d=True)
    if (rate!=24000 or teacher_rate!=rate or not matching_user_stimulus(a[:,1],b[:,1],
            side['user_words'],target_side['user_words'])):
        raise RuntimeError('Cancellation target requires exactly the same approved user PCM and words')
    for key in ('project','field','phrase'):
        if side['timeline']['meta'][key]!=target_side['timeline']['meta'][key]:
            raise RuntimeError('Cancellation teacher differs from the approved source/world')
    meta=side['timeline']['meta'];world={meta['project']:{meta['field']:meta['phrase']}}
    if record['source']['declared_counterfactual_world']!=world:
        raise RuntimeError('Action source differs from the declared TRAIN world')
    begin,end=cancel_window(original,teacher,target_side,example['clean_codes'][0].tolist())
    saved=torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)
    actual_voice=torch.load(history/'student-voice.pt',map_location='cpu',weights_only=True)['codes']
    if not torch.equal(actual_voice,voice):raise RuntimeError('Action data uses a different voice')
    rows=[json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    if end>len(rows) or any(abs(r['input_start_s']-i*.08)>1e-7 for i,r in enumerate(rows)):
        raise RuntimeError('Action target exceeds the unchanged actual input clock')
    live=[(i,c) for i,c in enumerate(record['calls']) if c['mode']=='live']
    chosen=live[begin:end];stop=chosen[0][1]['start']
    if (len(chosen)!=3 or any(c['frames']!=1 or c['start']!=stop+n for n,(_,c) in enumerate(chosen))):
        raise RuntimeError('Action window contains a span or noncontiguous actual input')
    if (begin<1 or record['calls'][chosen[0][0]-1]['mode']!='live'
            or any(saved['states'][f]['pending_exit_cb0'] is not None for f in range(begin-1,end))):
        raise RuntimeError('Action window overlaps a pending span-exit readout')
    component={'delayed_inputs':saved['delayed_inputs'][:,:,stop:stop+3].clone(),
        'hidden':torch.cat([saved['temporal_outputs'][i]['hidden'] for i,_ in chosen],dim=1),
        'text_targets':torch.tensor([3,0,4],dtype=torch.int64),'text_mask':torch.ones(3,dtype=torch.bool)}
    result={'schema':'dl-context-action-plan-v1','target_kind':'cancel_request',
        'scenario':checked['scenario'],
        'actual_history':str(history),'source_history_sha256':checked['history_sha256'],
        'source_manifest_sha256':sha256(history/'student-history.json'),
        'source_report_sha256':checked['report_sha256'],'source_execution':record['execution'],
        'voice_codes_sha256':record['source']['voice_codes_sha256'],'teacher':teacher,'world':world,
        'future_first_real_frame':begin,'future_frames':3,'actual_prefix_frames':stop,
        'request_real_frame':end-1,'request_input_clock_s':(end-1)*.08,
        'policy_code_sha256':sha256(Path(__file__)),'targets':[3,0,4],
        'target_provenance':'approved measured TRAIN teacher native EPAD/RET policy, not student predictions',
        'conditioning':'unchanged observed native trajectory; no future token intervention',
        'request_target_is_not_an_executed_cancellation':True,
        'spoken_acknowledgement_targets':0,'private_fact_targets':0,
        'past_loss_targets':0,'span_loss_targets':0,'audio_loss_targets':0,'user_loss_targets':0,
        'training_approved':False,'free_running_delegation_qualified':False}
    return result,component
