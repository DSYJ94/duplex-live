"""Roll approved PAD/EPAD/RET controls forward from an exact trained prefix.

Only three expert controls have loss. A fourth unforced frame verifies the
one-frame-delayed RET readout; its sampled text never becomes a target.
"""
import json
from pathlib import Path
import subprocess
from evals.moshi_model_download import sha256


def lookup_window(source,teacher,side,text):
    from evals.context_clarification_future import question_scope
    question_scope(source,teacher,side)
    requests=teacher.get('derived_requests',[])
    if teacher.get('request_policy')!='native-prelead-epad' or len(requests)!=1 or requests[0]['operation']!='lookup':
        raise RuntimeError('Clarified lookup needs one independently approved teacher request')
    frame=requests[0]['derived_request_frame']
    users=[e for e in side['timeline']['events'] if e['ch']=='user' and e['kind']=='utterance']
    if (frame<3 or text[frame-2:frame+1]!=[3,0,4]
            or users[1]['actual_end_ms']/1000+.24>frame*.08
            or (len(users)>2 and (frame+2)*.08+.24>users[2]['actual_start_ms']/1000)
            or any(a['ready_at_s']<=(frame+2)*.08 for a in side['reference_arrivals'])
            or any(a[2]=='SPEAKER_MAIN' and a[1][0]<(frame+1)*.08 and a[1][1]>(frame-2)*.08 for a in side['alignments'])):
        raise RuntimeError('Lookup controls must follow the full identified project and precede later speech/results')
    return frame-2,frame+1


def plan(history,data,assets,examples,spm,voice,teacher_id,kind):
    import torch
    import soundfile as sf
    from evals.recovery_data import validate_conversion
    from evals.context_history_validate import validate
    from evals.context_rollout_conditioning import verify
    from evals.context_actual_targets import matching_user_stimulus
    record=json.loads((history/'student-history.json').read_text())
    checked=validate(history,data,assets,record['source']['repo_commit'])
    collector=verify(json.loads((history/'report.json').read_text())['loading'])
    if record['execution']['math_sdp'] is not True:
        raise RuntimeError('Control roll-in needs the verified native MATH execution protocol')
    source=next(r for r in validate_conversion(data)['accepted'] if r['scenario_id']==checked['scenario'])
    side=json.loads(Path(source['path']).with_suffix('.json').read_text())
    example=next(e for e in examples if e['record']['id']==teacher_id);teacher=example['record']
    cache=json.loads((Path(teacher['cache'])/'cache.json').read_text())
    row=next(r for r in validate_conversion(Path(cache['conversion']))['accepted'] if r['scenario_id']==teacher_id)
    target=json.loads(Path(row['path']).with_suffix('.json').read_text())
    a,ar=sf.read(source['path'],dtype='float32',always_2d=True);b,br=sf.read(row['path'],dtype='float32',always_2d=True)
    if (ar!=24000 or br!=ar or not matching_user_stimulus(a[:,1],b[:,1],side['user_words'],target['user_words'])
            or any(side['timeline']['meta'][k]!=target['timeline']['meta'][k] for k in ('project','field','phrase'))):
        raise RuntimeError('Control roll-in needs the exact approved TRAIN user stimulus and world')
    if kind=='cancel_request':
        from evals.context_action_targets import cancel_window
        begin,end=cancel_window(source,teacher,target,example['clean_codes'][0].tolist())
    elif kind=='clarification_lookup':
        begin,end=lookup_window(source,teacher,target,example['clean_codes'][0].tolist())
    else:raise RuntimeError('Unsupported explicit control roll-in kind')
    meta=side['timeline']['meta'];world={meta['project']:{meta['field']:meta['phrase']}}
    if record['source']['declared_counterfactual_world']!=world:raise RuntimeError('Changed control world')
    if not torch.equal(voice,torch.load(history/'student-voice.pt',map_location='cpu',weights_only=True)['codes']):
        raise RuntimeError('Changed trained control voice')
    if not torch.equal(voice,torch.load(Path(collector['proof'])/'voice.pt',map_location='cpu',weights_only=True)['codes']):
        raise RuntimeError('Control voice differs from the verified trained collector')
    rows=[json.loads(l) for l in (history/'events.jsonl').read_text().splitlines()]
    live=[c for c in record['calls'] if c['mode']=='live'];saved=torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)
    # Include one unlabelled real frame to observe the RET predicted on frame2.
    stop=end+1
    if (begin<3 or stop>len(rows) or any(abs(r['input_start_s']-i*.08)>1e-7 for i,r in enumerate(rows))
            or any(live[i]['start']!=live[begin]['start']+i-begin for i in range(begin-2,stop))
            or any(saved['states'][i]['pending_exit_cb0'] is not None for i in range(begin-3,stop))):
        raise RuntimeError('Control/probe window overlaps a span or exceeds the actual input clock')
    state=saved['states'][begin-1];position=(state['offset']-1)%state['cache'].shape[2]
    return dict(schema='dl-context-control-rollin-plan-v1',target_kind=kind,actual_history=str(history),
        actual_history_sha256=record['sha256'],source_history=checked,teacher=teacher,world=world,
        source_execution=record['execution'],voice_codes_sha256=record['source']['voice_codes_sha256'],
        trained_collector=collector,conditioning_policy='verified-trained-rollout',
        replay_cut_index=begin-3,future_first_real_frame=begin,future_frames=4,
        request_prediction_real_frame=end-1,expected_request_readout_real_frame=end,
        actual_prefix_last_predicted_text_token=int(state['cache'][0,0,position]),
        controls=[3,0,4],probe_loss_targets=0,spoken_word_targets=0,private_fact_targets=0,
        past_loss_targets=0,span_loss_targets=0,audio_loss_targets=0,user_loss_targets=0,
        policy_code_sha256=sha256(Path(__file__)),training_approved=False,
        request_target_is_not_an_executed_action=True)


def generate(engine,saved,index,rows,pcm,manifest,out):
    import torch
    import numpy as np
    import soundfile as sf
    gen=engine.lm_gen;state=gen._streaming_state;begin=manifest['future_first_real_frame']
    if index+1!=begin or engine._pending_exit_cb0 is not None:raise RuntimeError('Wrong verified control restart')
    original_main=state.graphed_main;original_step=gen.step;original_prepare=gen.prepare_step_input
    inputs=[];hidden=[];current=[];provided=[];audio=[];codes=[];readout=[];n=-1
    def forward(value):
        if value.shape[-1]!=1:raise RuntimeError('Control roll-in unexpectedly prefills a span')
        result=original_main(value);inputs.append(value.detach().cpu().clone());hidden.append(result[0].detach().cpu().clone());return result
    def prepare(*a,**kw):
        result=original_prepare(*a,**kw)
        if result is None:raise RuntimeError('Control roll-in unexpectedly bootstraps')
        _,flags,target,_,_=result;provided.append(flags.detach().cpu().clone());current.append(target.detach().cpu().clone());return result
    def step(*a,**kw):
        if kw.get('text_token') is not None or kw.get('moshi_tokens') is not None:raise RuntimeError('Control roll-in acquired other forced inputs')
        if n<3:kw['text_token']=torch.tensor([manifest['controls'][n]],device=engine.device)
        return original_step(*a,**kw)
    state.graphed_main=forward;gen.prepare_step_input=prepare;gen.step=step
    try:
        for n in range(4):
            frame=begin+n;row=rows[frame];sample=round(row['input_start_s']*24000);chunk=pcm[sample:sample+row['input_samples']]
            if len(chunk)<1920:chunk=np.pad(chunk,(0,1920-len(chunk)))
            with torch.no_grad():result=engine.step(chunk)
            packet=np.asarray(result['agent_pcm'],dtype=np.float32)
            if packet.shape!=(1920,) or not np.isfinite(packet).all():raise RuntimeError('Invalid control probe audio')
            readout.append(result['text_token']);audio.append(packet)
            position=(state.offset-1)%state.cache.shape[2];codes.append(state.cache[:,1:9,position:position+1].detach().cpu().clone())
            old=saved['states'][frame];p=old['offset'];ring=old['cache'].shape[2];q=state.offset;r=state.cache.shape[2]
            actual=torch.cat([state.cache[:,9:10,(q-1)%r:(q-1)%r+1],state.cache[:,10:17,q%r:q%r+1]],dim=1).cpu()
            expected=torch.cat([old['cache'][:,9:10,(p-1)%ring:(p-1)%ring+1],old['cache'][:,10:17,p%ring:p%ring+1]],dim=1)
            if state.offset!=p or not torch.equal(actual,expected):raise RuntimeError('Control future changed actual user codes or clock')
    finally:state.graphed_main=original_main;gen.prepare_step_input=original_prepare;gen.step=original_step
    if readout!=[manifest['actual_prefix_last_predicted_text_token'],3,0,4] or not result['is_ret']:
        raise RuntimeError('Forced request did not produce its actual native delayed RET readout')
    values=dict(delayed_inputs=torch.cat(inputs,dim=2),hidden=torch.cat(hidden,dim=1),
        text_targets=torch.tensor([3,0,4,-1]),text_mask=torch.tensor([True,True,True,False]),
        live_native_targets=torch.cat(current,dim=2),live_provided=torch.cat(provided,dim=2),
        generated_delayed_audio=torch.cat(codes,dim=2))
    file=out/'control-rollin.pt';torch.save(values,file);wave=out/'control-rollin.wav';sf.write(wave,np.concatenate(audio),24000,subtype='FLOAT')
    names=['context_control_rollin.py','context_action_targets.py','context_clarification_future.py','context_engine_replay.py','context_rollout_conditioning.py']
    result=dict(manifest,schema='dl-context-control-rollin-capture-v1',file=file.name,sha256=sha256(file),
        native_future_text_readout=readout,native_request_readout_verified=True,
        actual_restart_gate_sha256=sha256(out/'engine-replay-before-future.json'),
        builder_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        builder_code_sha256={name:sha256(Path(__file__).with_name(name)) for name in names},
        audio_file=wave.name,audio_sha256=sha256(wave),own_assistant_audio_sampled=True,
        real_backend_call_executed=False,gold_targets_created=True,capture_complete=True)
    (out/'control-rollin.json').write_text(json.dumps(result,indent=2)+'\n');return result


def validate(folder,data,assets,examples,spm,voice):
    import hashlib
    import re
    import torch
    m=json.loads((folder/'control-rollin.json').read_text());history=Path(m['actual_history'])
    expected=plan(history,data,assets,examples,spm,voice,m['teacher']['id'],m['target_kind'])
    if (any(m.get(k)!=v for k,v in expected.items() if k!='schema')
            or m['schema']!='dl-context-control-rollin-capture-v1' or not m['capture_complete']
            or not m['native_request_readout_verified'] or not m['own_assistant_audio_sampled']
            or m['real_backend_call_executed'] or m['training_approved'] is not False or m['gold_targets_created'] is not True
            or m['file']!='control-rollin.pt' or sha256(folder/m['file'])!=m['sha256']
            or m['native_future_text_readout']!=[m['actual_prefix_last_predicted_text_token'],3,0,4]):
        raise RuntimeError('Control roll-in differs from its independently approved teacher policy')
    names={'context_control_rollin.py','context_action_targets.py','context_clarification_future.py','context_engine_replay.py','context_rollout_conditioning.py'}
    if set(m['builder_code_sha256'])!=names or not re.fullmatch('[a-f0-9]{40}',m['builder_commit']):
        raise RuntimeError('Incomplete control roll-in builder provenance')
    for name,digest in m['builder_code_sha256'].items():
        source=subprocess.check_output(['git','show',m['builder_commit']+':evals/'+name],cwd=Path(__file__).resolve().parents[1])
        if hashlib.sha256(source).hexdigest()!=digest:raise RuntimeError('Changed control roll-in builder')
    record=json.loads((history/'student-history.json').read_text());begin=m['future_first_real_frame']
    live=[c for c in record['calls'] if c['mode']=='live'];stop=live[begin]['start'];calls=[c for c in record['calls'] if c['start']<stop]
    gate_file=folder/'engine-replay-before-future.json';gate=json.loads(gate_file.read_text())
    if (sha256(gate_file)!=m['actual_restart_gate_sha256'] or gate['schema']!='dl-context-engine-restart-gate-v1'
            or not gate['passed'] or gate['source']!=m['source_history'] or gate['history']!=str(history)
            or gate['checked_real_frames']!=begin or gate['real_frame_boundary']!=m['replay_cut_index']
            or begin!=m['replay_cut_index']+3 or len(gate['steps'])!=begin or len(gate['inputs'])!=len(calls)
            or len(gate['temporal_numerics'])!=len(calls) or gate['new_backend_calls']
            or not gate['actual_user_encoder_and_agent_decoder_executed'] or not gate['actual_receipt_boundaries_replayed']
            or gate.get('conditioning_policy')!='verified-trained-rollout' or gate.get('trained_collector')!=m['trained_collector']
            or gate['execution']!=record['execution'] or gate['source_execution']!=record['execution']):
        raise RuntimeError('Control roll-in lost its exact native prefix restart')
    keys={'cache_exact','provided_exact','offset_exact','virtual_frames_exact','cpu_rng_exact','cuda_rng_exact','returned_text_exact'}
    for i,step in enumerate(gate['steps']):
        if set(step)!=keys|{'real_frame'} or step['real_frame']!=i or any(step[k] is not True for k in keys):
            raise RuntimeError('Incomplete control restart state checks')
    for i,(inp,num) in enumerate(zip(gate['inputs'],gate['temporal_numerics'])):
        if inp!={'call':i,'exact':True} or num!={'call':i,'hidden':{'exact':True,'max_abs':0.},'text_logits':{'exact':True,'max_abs':0.}}:
            raise RuntimeError('Control restart input or numerical mismatch')
    v=torch.load(folder/m['file'],map_location='cpu',weights_only=True)
    shapes=dict(delayed_inputs=(1,17,4),hidden=(1,4,4096),text_targets=(4,),text_mask=(4,),
        live_native_targets=(1,17,4),live_provided=(1,17,4),generated_delayed_audio=(1,8,4))
    if set(v)!=set(shapes) or any(tuple(v[k].shape)!=s for k,s in shapes.items()):raise RuntimeError('Invalid control tensor coverage')
    for k,value in v.items():
        dtype=torch.bfloat16 if k=='hidden' else torch.bool if k in ('text_mask','live_provided') else torch.int64
        if value.dtype!=dtype or not torch.isfinite(value).all():raise RuntimeError('Invalid control tensor dtype or values')
    if not torch.equal(v['text_targets'],torch.tensor([3,0,4,-1])) or not torch.equal(v['text_mask'],torch.tensor([True,True,True,False])):
        raise RuntimeError('Probe or other tokens acquired control supervision')
    flags=v['live_provided'];current=v['live_native_targets'];inputs=v['delayed_inputs'];generated=v['generated_delayed_audio']
    if (not flags[:,0,:3].all() or flags[:,0,3].any() or flags[:,1:9].any() or not flags[:,9:17].all()
            or not torch.equal(current[0,0,:3],torch.tensor([3,0,4]))
            or (generated<0).any() or (generated>=2048).any()):raise RuntimeError('Control target or acoustic ownership changed')
    raw=torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)['delayed_inputs']
    if not torch.equal(inputs[:,:,0],raw[:,:,stop]):raise RuntimeError('Control future altered its actual past')
    for n in range(4):
        if not torch.equal(inputs[:,9:17,n],raw[:,9:17,live[begin+n]['start']]):raise RuntimeError('Control future changed real user input')
        if n and (int(inputs[0,0,n])!=[3,0,4][n-1] or not torch.equal(inputs[:,1:9,n],generated[:,:,n-1])):
            raise RuntimeError('Control future did not condition on its own intervened text and sampled acoustics')
    if not torch.equal(voice,torch.load(folder/'voice.pt',map_location='cpu',weights_only=True)['codes']):raise RuntimeError('Changed control voice')
    import soundfile as sf
    audio,rate=sf.read(folder/m['audio_file'],dtype='float32')
    if (m['audio_file']!='control-rollin.wav' or sha256(folder/m['audio_file'])!=m['audio_sha256']
            or rate!=24000 or audio.shape!=(4*1920,) or not torch.isfinite(torch.from_numpy(audio)).all()):
        raise RuntimeError('Changed control diagnostic audio')
    return dict(schema='dl-context-control-rollin-audit-v1',component=str(folder),passed=True,
        manifest_sha256=sha256(folder/'control-rollin.json'),tensor_sha256=m['sha256'],
        source_history_sha256=m['actual_history_sha256'],source_execution=m['source_execution'],
        voice_codes_sha256=m['voice_codes_sha256'],builder_commit=m['builder_commit'],scenario=m['source_history']['scenario'],
        target_kind=m['target_kind'],diagnostic_kind='backend_request',diagnostic_frame=2,diagnostic_target=4,
        frames=4,text_targets=3,request_targets=1,past_loss_targets=0,span_loss_targets=0,audio_loss_targets=0,
        user_loss_targets=0,spoken_word_targets=0,private_fact_targets=0,probe_loss_targets=0,
        teacher_controls_rolled_in=True,native_request_readout_verified=True,trained_collector=m['trained_collector'],
        actual_action_execution_qualified=False),v


def main():
    import argparse
    import sys
    import torch
    import sentencepiece
    from evals.context_training_data import load_training
    from evals.personaplex_train import MODEL_HASHES
    from evals.lychee_context import BASE_INSTRUCTIONS
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('history','data','assets','tokenizer','out'):ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--teacher-id',required=True);ap.add_argument('--kind',choices=('cancel_request','clarification_lookup'),required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    ap.add_argument('--regression-capture',type=Path,action='append',default=[])
    ap.add_argument('--audit-capture',type=Path)
    args=ap.parse_args()
    if sha256(args.tokenizer)!=MODEL_HASHES['tokenizer_spm_32k_3.model']:raise RuntimeError('Changed control tokenizer')
    assets=json.loads(args.assets.read_text());sys.path.insert(0,assets['source'])
    from contextspan.model.sequence_convention import persona_prefix
    spm=sentencepiece.SentencePieceProcessor(str(args.tokenizer));voice=torch.load(args.history/'student-voice.pt',map_location='cpu',weights_only=True)['codes']
    examples,_=load_training(args.cache,spm,persona_prefix(BASE_INSTRUCTIONS,voice,spm),request_policy='native-prelead-epad')
    regressions=[]
    for folder in args.regression_capture:
        from evals.context_component_kind import manifest_name
        name=manifest_name(folder,mixed=True)
        if name=='early-future.json':from evals.context_early_validate import validate as check
        elif name=='actual-future.json':from evals.context_future_validate import validate as check
        elif name=='action-future.json':from evals.context_action_component import validate as check
        else:raise RuntimeError('Unsupported legacy control regression')
        audit,_=check(folder,args.data,args.assets,examples,spm,voice);regressions.append(audit)
    args.out.mkdir(exist_ok=False)
    if args.audit_capture:
        captured=json.loads((args.audit_capture/'control-rollin.json').read_text())
        if (Path(captured['actual_history']).resolve()!=args.history.resolve()
                or captured['target_kind']!=args.kind or captured['teacher']['id']!=args.teacher_id):
            raise RuntimeError('Control audit arguments differ from the selected history, teacher or kind')
        result,_=validate(args.audit_capture,args.data,args.assets,examples,spm,voice)
    else:
        result=plan(args.history,args.data,args.assets,examples,spm,voice,args.teacher_id,args.kind)
    result.update(cpu_eligible=True,legacy_regressions=regressions,all_legacy_regressions_passed=all(r['passed'] for r in regressions))
    (args.out/'control-plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
