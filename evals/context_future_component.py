"""Approved teacher sentence after verified actual student history; text-only component."""
import json
import math
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.context_actual_targets import matching_user_stimulus, actual_private_receipt, sentence_target, captured_packets


def plan(history, data, examples, spm, early=False, teacher_id=None):
    import numpy as np
    import soundfile as sf
    from evals.recovery_data import validate_conversion
    from evals.recovery_prepare_data import spoken_words
    record=json.loads((history/'student-history.json').read_text())
    conversion=validate_conversion(data)
    actual=next(r for r in conversion['accepted'] if r['scenario_id']==record['source']['scenario'])
    if early and actual['kind']=='simple':
        from evals.context_ordinary_future import plan as ordinary_plan
        return ordinary_plan(history,data,examples,spm,teacher_id)
    if early and actual['kind']=='clarify' and json.loads((history/'report.json').read_text()).get('backend_execution_disabled'):
        from evals.context_clarification_future import plan as question_plan
        return question_plan(history,data,examples,spm,teacher_id)
    kinds = ('private','clarify') if early else ('private',)
    if actual['split']!='train' or actual['kind'] not in kinds:
        raise RuntimeError('Actual fact futures require eligible private/clarification TRAIN data')
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
        if teacher_id is not None and r['id'] != teacher_id:continue
        if r['kind']!=actual['kind'] or r['source_model']!='gpt-live-1':continue
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
    report=json.loads((history/'report.json').read_text())
    if sha256(history/'model.wav')!=report['audio_sha256']:
        raise RuntimeError('Actual generated recording changed')
    audio,pr=sf.read(history/'model.wav',dtype='float32')
    if pr!=rate:raise RuntimeError('Actual packet clock differs from source input')
    packets=captured_packets(audio,report['playout']['packets'],rows)
    quiet=[len(part)>0 and float(np.sqrt(np.mean(part**2)))<.005 and row['text_token'] in (0,3)
        for row,part in zip(rows,packets)]
    last_user=max(w['end_s'] for w in side['user_words'])
    selected = (0,actual_private_receipt(events,project,field,value,rows[-1]['input_available_s'])) if early else None
    # A late correction needs a quiet gap and must not repeat an already-spoken
    # fact. An early continuation instead independently checks a silent prefix
    # before every student reply, and its original teacher clock after receipt.
    for i in ([] if early else range(len(rows)-length-3)):
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
        'actual_history_sha256':record['sha256'],
        'actual_history_manifest_sha256':sha256(history/'student-history.json'),'actual_report_sha256':sha256(history/'report.json'),
        'actual_generated_audio_sha256':report['audio_sha256'],
        'quiet_boundary_audio_source':'not required; early prefix/receipt clocks checked separately' if early
            else 'verified model.wav and completely checked packet ledger',
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
    builder_names=['context_future_component.py','context_actual_targets.py','context_engine_replay.py']
    if manifest.get('target_kind')=='cancellation_acknowledgment':builder_names.append('context_cancel_ack.py')
    result={**manifest,'schema':'dl-context-actual-future-component-v1',
        'builder_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'builder_code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in builder_names},
        'actual_restart_gate_sha256':sha256(out/'engine-replay-before-future.json'),'file':file.name,'sha256':sha256(file),
        'actual_engine_restart_passed':True,'all_future_user_codes_match_actual_capture':True,
        'native_pending_assistant_residuals_sampled_not_relabelled':True,
        'past_label_count':0,'future_word_targets':int((gold[0]>=5).sum()),
        'future_text_targets':int(text_mask.sum()),'forced_gold_future':True,
        'future_text_targets_approved':True,'gold_targets_created':True,
        'audio_targets_forced_for_conditioning_only':True,'audio_loss_targets':0,
        'voice_quality_or_free_running_success_qualified':False,'training_approved':False}
    (out/'actual-future.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    """CPU-only eligibility plan; no model forward, labels or optimizer."""
    import argparse
    import sys
    import torch
    import sentencepiece
    from evals.context_history_validate import validate
    from evals.context_training_data import load_training
    from evals.personaplex_train import MODEL_HASHES
    from evals.lychee_context import BASE_INSTRUCTIONS
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('history','data','assets','tokenizer','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--early-window',action='store_true',help='Plan a complete future before any student reply; CPU only')
    ap.add_argument('--cancel-ack',action='store_true',help='Plan only a completed cancellation acknowledgment')
    ap.add_argument('--teacher-id',help='Bind planning to one independently approved exact-user teacher')
    ap.add_argument('--request-policy',choices=('source-dispatch','native-prelead-epad'),default='source-dispatch')
    ap.add_argument('--audit-early-capture',type=Path)
    ap.add_argument('--audit-future-capture',type=Path)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    args=ap.parse_args()
    if args.cancel_ack and (args.early_window or args.audit_early_capture or args.audit_future_capture or not args.teacher_id):
        ap.error('Cancellation plan needs explicit late teacher selection without another audit mode')
    if args.audit_future_capture and (args.early_window or args.audit_early_capture):
        ap.error('Choose one explicit future audit mode')
    args.out.mkdir(exist_ok=False)
    validate(args.history,args.data,args.assets,args.source_commit)
    if sha256(args.tokenizer)!=MODEL_HASHES['tokenizer_spm_32k_3.model']:
        raise RuntimeError('Changed native tokenizer')
    assets=json.loads(args.assets.read_text());sys.path.insert(0,assets['source'])
    from contextspan.model.sequence_convention import persona_prefix
    spm=sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    voice=torch.load(args.history/'student-voice.pt',weights_only=True)['codes']
    examples,_=load_training(args.cache,spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),request_policy=args.request_policy)
    try:
        if args.audit_future_capture:
            from evals.context_future_validate import validate as validate_future
            result,_=validate_future(args.audit_future_capture,args.data,args.assets,examples,spm,voice)
        elif args.audit_early_capture:
            from evals.context_early_validate import validate as validate_early
            result,_=validate_early(args.audit_early_capture,args.data,args.assets,examples,spm,voice)
        elif args.cancel_ack:
            from evals.context_cancel_ack import plan as cancel_plan
            result,_=cancel_plan(args.history,args.data,examples,spm,args.teacher_id)
        else:
            result,_=plan(args.history,args.data,examples,spm,early=args.early_window,teacher_id=args.teacher_id)
            if args.early_window:
                from evals.context_early_plan import plan as early_plan
                result,_=early_plan(result,examples)
        result['eligible_for_strict_replay']=True
    except RuntimeError as error:
        result={'eligible_for_strict_replay':False,'reason':str(error),
            'training_approved':False,'gold_targets_created':False}
    (args.out/('early-future-audit.json' if args.audit_early_capture else 'future-plan.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
