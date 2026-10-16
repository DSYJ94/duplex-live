"""Approved teacher sentence after verified actual student history; text-only component."""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.context_actual_targets import matching_user_stimulus, actual_private_receipt, sentence_target


def plan(history, data, examples, spm):
    import numpy as np
    import soundfile as sf
    from evals.recovery_data import validate_conversion
    from evals.recovery_prepare_data import spoken_words
    record=json.loads((history/'student-history.json').read_text())
    conversion=validate_conversion(data)
    actual=next(r for r in conversion['accepted'] if r['scenario_id']==record['source']['scenario'])
    if actual['split']!='train' or actual['kind']!='private':
        raise RuntimeError('Initial actual-future component is private TRAIN only')
    side=json.loads(Path(actual['path']).with_suffix('.json').read_text())
    project=side['timeline']['meta']['project'];field=side['timeline']['meta']['field']
    world=record['source']['declared_counterfactual_world']
    if set(world)!={project} or set(world[project])!={field}:
        raise RuntimeError('Actual declared world does not match the approved question')
    value=world[project][field]
    actual_pcm,rate=sf.read(actual['path'],dtype='float32',always_2d=True)
    candidates=[];sources_by_cache={}
    for example in examples:
        r=example['record']
        if r['kind']!='private' or r['source_model']!='gpt-live-1':continue
        cache=Path(r['cache']);cache_receipt=json.loads((cache/'cache.json').read_text())
        if cache not in sources_by_cache:
            approved=validate_conversion(Path(cache_receipt['conversion']))
            sources_by_cache[cache]={t['scenario_id']:t for t in approved['accepted']}
        teacher=sources_by_cache[cache][r['id']]
        ts=json.loads(Path(teacher['path']).with_suffix('.json').read_text());meta=ts['timeline']['meta']
        if (meta['project'],meta['field'],meta['phrase'])!=(project,field,value):continue
        teacher_pcm,tr=sf.read(teacher['path'],dtype='float32',always_2d=True)
        if tr!=rate or not matching_user_stimulus(actual_pcm[:,1],teacher_pcm[:,1],side['user_words'],ts['user_words']):continue
        if len(ts['reference_arrivals'])!=1:raise RuntimeError('Ambiguous approved teacher result clock')
        try:
            selection=sentence_target(example['clean_codes'][0].tolist(),ts['alignments'],value,
                spm.encode,ts['reference_arrivals'][0]['ready_at_s'])
        except RuntimeError:
            continue  # Reject this sentence, never repair measured labels.
        candidates.append((r['id'],example,teacher,selection))
    if not candidates:raise RuntimeError('No independently approved same-user/same-world teacher sentence')
    _,example,teacher,selection=sorted(candidates,key=lambda t:t[0])[0]
    left,right=selection['source_start_frame'],selection['source_end_frame']
    gold=example['clean_codes'][:9,left:right].clone()
    if (gold<0).any() or (gold[1:]>=2048).any() or (gold[0]>=32000).any():
        raise RuntimeError('Approved assistant sentence has invalid native targets')
    length=gold.shape[1]
    rows=[json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    events=json.loads((history/'worker-events.json').read_text())
    packet,pr=sf.read(history/'packet-stream.wav',dtype='float32')
    if pr!=rate or packet.ndim!=1:raise RuntimeError('Unexpected actual packet stream')
    quiet=[];cursor=0
    for row in rows:
        count=row['output_samples'];part=packet[cursor:cursor+count];cursor+=count
        quiet.append(count>0 and len(part)==count and float(np.sqrt(np.mean(part**2)))<.005
            and row['text_token'] in (0,3))
    if cursor!=len(packet):raise RuntimeError('Actual packet/frame accounting differs')
    last_user=max(w['end_s'] for w in side['user_words'])
    selected=None
    for i in range(len(rows)-length-3):
        if rows[i]['input_available_s']<last_user+.4 or not all(quiet[i:i+5]):continue
        # No actual context span can be silently dropped from the future interval.
        if any(rows[j+1]['history_virtual_frames_before_step']!=rows[j]['history_virtual_frames_after_step']
                for j in range(i,i+length+2)):continue
        cut_s=rows[i]['input_available_s']
        try:receipt=actual_private_receipt(events,project,field,value,cut_s-.24)
        except RuntimeError:continue
        past=spoken_words(spm.decode([r['text_token'] for r in rows[:i+3] if r['text_token']>=5]))
        phrase=spoken_words(value)
        if any(past[j:j+len(phrase)]==phrase for j in range(len(past)-len(phrase)+1)):continue
        begin=round(rows[i+3]['input_start_s']*rate)
        user=np.pad(actual_pcm[:,1],(0,20*rate))[begin:begin+length*1920]
        if len(user)!=length*1920 or np.count_nonzero(user):continue
        selected=(i,receipt);break
    if selected is None:raise RuntimeError('No verified quiet corrective boundary without an already-spoken value')
    i,receipt=selected
    result={'schema':'dl-context-actual-future-plan-v1','actual_history':str(history),
        'actual_history_sha256':record['sha256'],'actual_report_sha256':sha256(history/'report.json'),
        'teacher':example['record'],'selection':selection,'matching_approved_teacher_count':len(candidates),
        'world':world,'actual_private_receipt':receipt,'replay_cut_index':i,
        'future_first_real_frame':i+3,'future_frames':length,'all_past_labels_masked':True,
        'composition':'approved teacher sentence retimed after actual student history; source word spacing retained',
        'training_approved':False,'gold_targets_created':False}
    return result,gold


def generate(engine,saved,index,rows,pcm,gold,manifest,out):
    import numpy as np
    import torch
    state=engine.lm_gen._streaming_state;gen=engine.lm_gen
    if index+1!=manifest['future_first_real_frame'] or engine._pending_exit_cb0 is not None:
        raise RuntimeError('Future generation starts from the wrong verified actual state')
    original_main=state.graphed_main;original_prepare=gen.prepare_step_input
    inputs,features,targets,provided=[],[],[],[]
    def main(value):
        output=original_main(value)
        inputs.append(value.detach().cpu().clone());features.append(output[0].detach().cpu().clone())
        return output
    def prepare(*a,**kw):
        result=original_prepare(*a,**kw)
        if result is None:raise RuntimeError('Future component unexpectedly bootstrapped the generator')
        _,flags,target,_,_=result
        targets.append(target.detach().cpu().clone());provided.append(flags.detach().cpu().clone())
        return result
    state.graphed_main=main;gen.prepare_step_input=prepare
    try:
        with torch.no_grad():
            for n in range(gold.shape[1]):
                frame=index+1+n;row=rows[frame];begin=round(row['input_start_s']*24000)
                chunk=pcm[begin:begin+row['input_samples']]
                if len(chunk)<1920:chunk=np.pad(chunk,(0,1920-len(chunk)))
                user=engine.user_mimi.encode(torch.as_tensor(chunk,device=engine.device).reshape(1,1,-1))
                recorded=saved['states'][frame];cache=recorded['cache'];offset=recorded['offset'];ring=cache.shape[2]
                actual_user=torch.cat([cache[:,9:10,(offset-1)%ring:(offset-1)%ring+1],
                    cache[:,10:17,offset%ring:offset%ring+1]],dim=1)
                if not torch.equal(user.cpu(),actual_user):raise RuntimeError('Future user codec differs from actual streamed input')
                tok=gen.step(input_tokens=user,moshi_tokens=gold[None,1:9,n:n+1].to(engine.device),
                    text_token=gold[0,n:n+1].to(engine.device))
                if tok is None:raise RuntimeError('Future decoder unexpectedly lacks a native token')
                engine.mimi.decode(tok[:,1:9])
    finally:
        state.graphed_main=original_main;gen.prepare_step_input=original_prepare
    target=torch.cat(targets,dim=2);flags=torch.cat(provided,dim=2)
    if not torch.equal(target[0,0],gold[0]) or not flags[:,0].all() or not flags[:,1].all():
        raise RuntimeError('Current approved text/semantic targets are not provided at their native clock')
    if flags[:,2:9,0].any() or (gold.shape[1]>1 and not flags[:,2:9,1:].all()):
        raise RuntimeError('Pending assistant residuals were incorrectly relabelled as observed gold')
    text_mask=(gold[0]>=5)|(gold[0]==0)
    if int((gold[0]>=5).sum())<2:raise RuntimeError('Future text component lacks measured words')
    file=out/'actual-future.pt'
    torch.save({'delayed_inputs':torch.cat(inputs,dim=2),'hidden':torch.cat(features,dim=1),
        'text_targets':gold[0],'text_mask':text_mask,'provided':flags,
        'native_current_targets':target,'approved_assistant_codes':gold[1:9]},file)
    result={**manifest,'schema':'dl-context-actual-future-component-v1','file':file.name,'sha256':sha256(file),
        'actual_engine_restart_passed':True,'all_future_user_codes_match_actual_capture':True,
        'native_pending_assistant_residuals_sampled_not_relabelled':True,
        'past_label_count':0,'future_word_targets':int((gold[0]>=5).sum()),
        'future_text_targets':int(text_mask.sum()),'forced_gold_future':True,
        'future_text_targets_approved':True,'gold_targets_created':True,
        'audio_targets_forced_for_conditioning_only':True,'audio_loss_targets':0,
        'voice_quality_or_free_running_success_qualified':False,'training_approved':False}
    (out/'actual-future.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
