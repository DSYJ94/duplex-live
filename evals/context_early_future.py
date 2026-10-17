"""Capture a quarantined complete teacher future through the native Engine.

Recorded receipts are supplied at their actual frame boundaries. Future requests
are supervised offline, not dispatched. This is never a live delegation pass.
"""
import json
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256


def generate(engine, saved, index, rows, pcm, gold, manifest, out, native_audio=False):
    import numpy as np
    import torch
    gate_file = out/'engine-replay-before-future.json'
    gate = json.loads(gate_file.read_text()); begin = manifest['future_first_real_frame']
    if (not gate['passed'] or gate['history'] != manifest['actual_history']
            or gate['source']['history_sha256'] != manifest['actual_history_sha256']
            or gate['checked_real_frames'] != begin or index+1 != begin
            or engine._pending_exit_cb0 is not None):
        raise RuntimeError('Early future lacks its exact actual-prefix restart')
    gen = engine.lm_gen; state = gen._streaming_state; lm = engine.lm
    original_main, original_direct = state.graphed_main, lm.forward_codes
    original_step, original_prepare = gen.step, gen.prepare_step_input
    phase = 'span'; n = -1; cursor = 0
    inputs, hidden, calls, targets, masks = [], [], [], [], []
    current_targets, provided, real_to_virtual = [], [], []
    generated_audio, audio_packets = [], []
    readout=[];check_readout=manifest.get('native_response_boundary_version')==2
    def capture(original, route):
        def wrapped(value):
            nonlocal cursor
            start = cursor; count = value.shape[-1]
            result = original(value)
            inputs.append(value.detach().cpu().clone()); hidden.append(result[0].detach().cpu().clone())
            label = torch.full((count,),-1,dtype=torch.int64); mask = torch.zeros(count,dtype=torch.bool)
            if phase == 'live':
                if route != 'graph' or count != 1: raise RuntimeError('Unexpected future live forward')
                label[0] = gold[0,n]; mask[0] = bool(label[0]>=4 or label[0]==0)
                real_to_virtual.append(start)
            elif route != 'direct': raise RuntimeError('Unexpected future span forward')
            calls.append({'phase':phase,'route':route,'start':start,'frames':count})
            targets.append(label); masks.append(mask); cursor += count
            return result
        return wrapped
    def force(*a,**kw):
        if phase == 'live':
            if kw.get('moshi_tokens') is not None or kw.get('text_token') is not None:
                raise RuntimeError('Native Engine acquired an unexpected forced input')
            kw.update(text_token=gold[0,n:n+1].to(engine.device))
            if not native_audio:kw['moshi_tokens']=gold[None,1:9,n:n+1].to(engine.device)
        return original_step(*a,**kw)
    def prepare(*a,**kw):
        result = original_prepare(*a,**kw)
        if phase == 'live':
            if result is None: raise RuntimeError('Early future unexpectedly bootstrapped')
            _,flags,target,_,_ = result
            provided.append(flags.detach().cpu().clone()); current_targets.append(target.detach().cpu().clone())
        return result
    state.graphed_main = capture(original_main,'graph'); lm.forward_codes = capture(original_direct,'direct')
    gen.step, gen.prepare_step_input = force, prepare
    spans = manifest['future_recorded_receipts']; applied = []
    try:
        with torch.no_grad():
            for n in range(gold.shape[1]):
                frame = begin+n
                for span in (s for s in spans if s['real_frame']==frame):
                    phase = 'span'; event = span['event']; start = cursor
                    from evals.context_engine_replay import recorded_span_text
                    count = engine.inject_context_span(recorded_span_text(event))
                    if count != event['virtual_span_frames'] or cursor-start != count:
                        raise RuntimeError('Recorded future context length changed')
                    applied.append({'real_frame':frame,'virtual_start':start,'virtual_frames':count,'event':event})
                phase = 'live'; row = rows[frame]; sample = round(row['input_start_s']*24000)
                chunk = pcm[sample:sample+row['input_samples']]
                if len(chunk)<1920:chunk=np.pad(chunk,(0,1920-len(chunk)))
                output=engine.step(chunk)  # Both native codecs and the span-exit readout run normally.
                if check_readout:
                    expected=0 if n==0 else int(gold[0,n-1])
                    if output['text_token']!=expected:
                        raise RuntimeError('Future returned text does not match the verified one-frame prediction delay')
                    readout.append(output['text_token'])
                if native_audio:
                    position=(state.offset-1)%state.cache.shape[2]
                    generated_audio.append(state.cache[:,1:9,position:position+1].detach().cpu().clone())
                    packet=np.asarray(output['agent_pcm'],dtype=np.float32)
                    if packet.shape!=(1920,) or not np.isfinite(packet).all():
                        raise RuntimeError('Invalid own-sampled future speech packet')
                    audio_packets.append(packet.copy())
                original_state = saved['states'][frame]
                if state.offset != original_state['offset']:
                    raise RuntimeError('Future native real/virtual clock changed')
                ring = state.cache.shape[2]; pos = state.offset
                old = original_state['cache']; old_ring = old.shape[2]; old_pos = original_state['offset']
                actual_user = torch.cat([state.cache[:,9:10,(pos-1)%ring:(pos-1)%ring+1],
                    state.cache[:,10:17,pos%ring:pos%ring+1]],dim=1).cpu()
                expected_user = torch.cat([old[:,9:10,(old_pos-1)%old_ring:(old_pos-1)%old_ring+1],
                    old[:,10:17,old_pos%old_ring:old_pos%old_ring+1]],dim=1)
                if not torch.equal(actual_user,expected_user):
                    raise RuntimeError('Actual future user encoder codes changed')
    finally:
        state.graphed_main, lm.forward_codes = original_main, original_direct
        gen.step, gen.prepare_step_input = original_step, original_prepare
    current = torch.cat(current_targets,dim=2); flags = torch.cat(provided,dim=2)
    if (not torch.equal(current[0,0],gold[0]) or not flags[:,0].all()
            or not flags[:,9:17].all() or len(real_to_virtual) != gold.shape[1]
            or len(applied) != len(spans)):
        raise RuntimeError('Incomplete early-future target ownership or receipt coverage')
    if native_audio:
        if flags[:,1:9].any():raise RuntimeError('Own-sampled future acquired provided assistant audio')
    elif (not flags[:,1].all() or not torch.equal(current[0,1],gold[1])
            or flags[:,2:9,0].any() or not flags[:,2:9,1:].all()
            or not torch.equal(current[0,2:9,1:],gold[2:9,:-1])):
        raise RuntimeError('Teacher future acoustic ownership changed')
    file = out/'early-future.pt'
    tensors={'delayed_inputs':torch.cat(inputs,dim=2),'hidden':torch.cat(hidden,dim=1),
        'text_targets':torch.cat(targets),'text_mask':torch.cat(masks),
        'live_native_targets':current,'live_provided':flags,'approved_assistant_codes':gold[1:9]}
    audio_record={}
    if native_audio:
        import soundfile as sf
        tensors['generated_delayed_audio']=torch.cat(generated_audio,dim=2)
        wave=out/'early-native-audio.wav';sf.write(wave,np.concatenate(audio_packets),24000,subtype='FLOAT')
        audio_record={'native_audio_file':wave.name,'native_audio_sha256':sha256(wave),
            'native_audio_samples':len(audio_packets)*1920}
    torch.save(tensors,file)
    history = Path(manifest['actual_history']); record = json.loads((history/'student-history.json').read_text())
    builder_names = ['context_early_future.py','context_early_plan.py','context_engine_replay.py']
    if manifest.get('target_kind')=='ordinary_answer':builder_names.append('context_ordinary_future.py')
    if manifest.get('native_response_boundary'):builder_names.append('context_response_boundary.py')
    if manifest.get('target_kind')=='clarification_question':builder_names.append('context_clarification_future.py')
    result = {**manifest,'schema':'dl-context-early-future-capture-v1','file':file.name,'sha256':sha256(file),
        'builder_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'builder_code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in builder_names},
        'actual_restart_gate_sha256':sha256(gate_file),'voice_codes_sha256':record['source']['voice_codes_sha256'],
        'source_execution':record['execution'],'calls':calls,'real_to_future_virtual':real_to_virtual,
        'applied_recorded_receipts':applied,'virtual_future_frames':cursor,
        'all_future_user_codes_match_actual_capture':True,'gold_targets_created':True,
        'capture_complete':True,'future_text_targets_approved':False,'training_approved':False,
        'future_audio_conditioning':'native_sampled' if native_audio else 'approved_teacher',
        'future_assistant_audio_tokens_forced':not native_audio,'future_text_tokens_forced':True,
        'teacher_audio_codes_are_reference_only':native_audio,**audio_record,
        'requires_independent_component_audit_before_any_optimizer':True}
    if check_readout:result.update(native_future_text_readout=readout,
        native_future_text_readout_delay_verified=True)
    (out/'early-future.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
