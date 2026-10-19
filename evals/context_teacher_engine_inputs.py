"""Compare a complete approved teacher with the actual native Engine input path.

Teacher text and assistant acoustics are forced for this engineering audit.
Both codecs execute; no backend, optimizer or native task claim is involved.
The separate native-audio probe supplies only teacher words and records sampled
agent audio under a different schema; it grants no training-data approval.
"""
import hashlib
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def forcing(clean,frame,device,native_audio):
    result=dict(text_token=clean[0,frame:frame+1].to(device))
    if not native_audio:result['moshi_tokens']=clean[None,1:9,frame:frame+1].to(device)
    return result


def provided_ownership(flags,current,clean,native_audio,prefix=None):
    import torch
    if native_audio:
        if prefix is None:return False
        expected=torch.ones_like(flags);expected[:,1:9]=False
        # The seven delay-one rows start with the last provided persona frame.
        # They are not teacher acoustic tokens or newly sampled predictions.
        expected[:,2:9,0]=True
        owned=torch.equal(flags,expected) and torch.equal(current[0,2:9,0],prefix[2:9,-1])
    else:owned=bool(flags.all())
    return owned and torch.equal(current[0,0],clean[0])


def compare(actual, expected):
    import torch
    if actual.shape != expected.shape:
        return dict(exact=False,actual_shape=list(actual.shape),expected_shape=list(expected.shape))
    different=actual!=expected
    first=different.nonzero().tolist()[:24]
    return dict(exact=not bool(different.any()),actual_shape=list(actual.shape),
        expected_shape=list(expected.shape),mismatches_per_channel=different.sum((0,2)).tolist(),
        first_mismatches=[dict(batch=b,channel=c,frame=f,actual=int(actual[b,c,f]),
            expected=int(expected[b,c,f])) for b,c,f in first],
        actual_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
        expected_sha256=hashlib.sha256(expected.contiguous().numpy().tobytes()).hexdigest())


def streamed_capture_control(actual, legacy, streamed, ownership):
    """Preserve the failed offline control; accept only exact streamed-user inputs."""
    old = compare(actual, legacy)
    new = compare(actual, streamed)
    counts = old.get('mismatches_per_channel', [])
    user_only = len(counts) == 17 and not any(counts[:9])
    return dict(passed=bool(ownership and user_only and new['exact']),
                legacy_offline_comparison=old, streamed_user_input_comparison=new,
                legacy_diagnostic_retained=True, only_user_conditioning_may_differ=user_only,
                independent_full_corpus_approval_required=True, training_approved=False)


def run(lm,mimi,spm,voice,examples,args):
    import numpy as np
    import soundfile as sf
    import torch
    import subprocess
    import contextspan.model.engine as native
    from contextspan.moshi.models.lm import _delay_sequence
    from contextspan.moshi.utils.compile import no_compile,no_cuda_graph
    from evals.context_native_layout import install
    from evals.context_execution import math_attention,snapshot
    from evals.context_retention_teacher import scope
    from evals.recovery_data import validate_conversion
    from evals.lychee_context import BASE_INSTRUCTIONS
    native_audio=bool(getattr(args,'teacher_engine_native_audio',False))
    streamed_capture=bool(getattr(args,'teacher_engine_input_streamed_user',False))
    if streamed_capture and native_audio:
        raise RuntimeError('Streamed-user input capture cannot change teacher acoustic ownership')
    audit=json.loads(args.teacher_engine_input_audit.read_text())
    if (audit['schema'] not in ('dl-context-all-teacher-scope-audit-v1','dl-context-all-teacher-scope-audit-v2') or not audit['passed']
            or audit['training_approved'] is not False or audit['optimizer_steps']!=0
            or audit['code_sha256']!=sha256(Path(__file__).with_name('context_retention_teacher.py'))
            or audit['voice_sha256']!=sha256(args.out/'voice.pt')
            or audit['assets_manifest_sha256']!=sha256(args.assets)):
        raise RuntimeError('Teacher Engine audit requires its matching independent all-teacher scope')
    example=next(e for e in examples if e['record']['id']==args.teacher_engine_input_id)
    if audit['schema'].endswith('-v2'):
        from evals.context_teacher_native_approval import checked_scope
        from evals.context_observed_teacher import canonical
        checked_scope(args.teacher_engine_input_audit,args.assets,args.out/'voice.pt')
        example=canonical(example,spm)
    labels,mask,checked=scope(example)
    if checked not in audit['teachers']:raise RuntimeError('Teacher is absent from the independent scope audit')
    record=example['record'];clean=example['clean_codes'];frames=clean.shape[1]
    if (clean[1:]<0).any() or (clean[1:]>=2048).any():
        raise RuntimeError('Full teacher codec inputs contain invalid or unavailable acoustic codes')
    cache=json.loads((Path(record['cache'])/'cache.json').read_text())
    source=next(r for r in validate_conversion(Path(cache['conversion']))['accepted']
        if r['scenario_id']==record['id'])
    audio,rate=sf.read(source['path'],dtype='float32',always_2d=True)
    if rate!=24000 or audio.shape[1]!=2 or not np.isfinite(audio).all():
        raise RuntimeError('Invalid independently approved teacher stereo recording')
    pcm=audio[:,1];length=frames*1920
    if len(pcm)>length and np.count_nonzero(pcm[length:]):
        raise RuntimeError('Teacher cache would omit observed user speech at the end')
    pcm=np.pad(pcm,(0,max(0,length-len(pcm))))[:length]
    # Match the recorded native runtime, rather than the trainer's CuDNN-off setting.
    torch.backends.cudnn.enabled=True
    math_attention();lm.eval().requires_grad_(False)
    old_loader=native.load_model
    native.load_model=lambda *a,**kw:(lm,mimi,spm)
    try:
        with no_compile(),no_cuda_graph():engine=native.Engine(device='cuda',temp=.8,temp_text=.7)
    finally:native.load_model=old_loader
    install(engine);native.seed_all(42424242);engine.reset()
    state=engine.lm_gen._streaming_state;gen=engine.lm_gen
    original_direct,original_graph=lm.forward_codes,state.graphed_main
    original_step,original_prepare=gen.step,gen.prepare_step_input
    chunks,calls,provided,targets,user_codes,packets,generated=[],[],[],[],[],[],[]
    phase='prefix';frame=-1;cursor=0;live_indices=[];readout=[];applied=[]
    def capture(original,route):
        def wrapped(value):
            nonlocal cursor
            chunks.append(value.detach().cpu().clone())
            calls.append(dict(phase=phase,route=route,start=cursor,frames=value.shape[-1]))
            if phase=='live':
                if route!='graph' or value.shape[-1]!=1:raise RuntimeError('Unexpected live teacher forward')
                live_indices.append(cursor)
            cursor+=value.shape[-1]
            return original(value)
        return wrapped
    def force(*a,**kw):
        if phase=='live':
            if a or kw.get('moshi_tokens') is not None or kw.get('text_token') is not None:
                raise RuntimeError('Teacher Engine acquired unexpected forced inputs')
            user_codes.append(kw['input_tokens'].detach().cpu().clone())
            kw.update(forcing(clean,frame,engine.device,native_audio))
        result=original_step(*a,**kw)
        if native_audio and phase=='live':
            if result is None:raise RuntimeError('Native teacher rendering lost a generated frame')
            generated.append(result.detach().cpu().clone())
        return result
    def prepare(*a,**kw):
        result=original_prepare(*a,**kw)
        if phase=='live':
            if result is None:raise RuntimeError('Teacher real frame unexpectedly bootstrapped')
            _,flags,target,_,_=result
            provided.append(flags.detach().cpu().clone());targets.append(target.detach().cpu().clone())
        return result
    lm.forward_codes,state.graphed_main=capture(original_direct,'direct'),capture(original_graph,'graph')
    gen.step,gen.prepare_step_input=force,prepare
    try:
        with torch.no_grad(),no_compile(),no_cuda_graph():
            engine.set_persona(BASE_INSTRUCTIONS,voice)
            for frame in range(frames):
                for span in (s for s in record['spans'] if s['inject_frame']==frame):
                    phase='span';begin=cursor;n=engine.inject_context_span(span['reference'])
                    applied.append(dict(real_frame=frame,virtual_start=begin,frames=n,reference=span['reference']))
                phase='live';output=engine.step(pcm[frame*1920:(frame+1)*1920])
                packet=np.asarray(output['agent_pcm'],dtype=np.float32)
                if packet.shape!=(1920,) or not np.isfinite(packet).all():
                    raise RuntimeError('Teacher native decoder returned invalid PCM')
                packets.append(packet.copy());readout.append(output['text_token'])
    finally:
        lm.forward_codes,state.graphed_main=original_direct,original_graph
        gen.step,gen.prepare_step_input=original_step,original_prepare
    actual=torch.cat(chunks,dim=2);initial=lm._get_initial_token().cpu()
    expected=torch.cat([initial,_delay_sequence(lm.delays,example['codes'][None],initial)],dim=2)[:,:,:-1]
    inputs=compare(actual,expected);encoded=torch.cat(user_codes,dim=2)
    codec=compare(encoded,clean[None,9:17])
    current=torch.cat(targets,dim=2);flags=torch.cat(provided,dim=2)
    from contextspan.model.sequence_convention import persona_prefix
    native_prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm) if native_audio else None
    ownership=provided_ownership(flags,current,clean,native_audio,native_prefix)
    native_comparison=None
    if native_audio:
        # Text and user inputs remain supplied; agent acoustic inputs must be sampled.
        from evals.context_teacher_native_approval import prepared
        oracle=prepared(example,native_prefix,encoded,spm)
        non_agent=[0,*range(9,17)]
        native_comparison=compare(actual[:,non_agent],oracle[:,non_agent])
    mapping=live_indices==record['real_to_virtual']
    if not mapping:raise RuntimeError('Teacher Engine real/virtual mapping changed')
    wave=args.out/('teacher-text-native-audio.wav' if native_audio else 'teacher-forced-native.wav')
    sf.write(wave,np.concatenate(packets),24000,subtype='FLOAT')
    file=args.out/('teacher-text-native-audio.pt' if native_audio else 'teacher-engine-inputs.pt')
    tensors=dict(delayed_inputs=actual,expected_prepared_inputs=expected,text_targets=labels,text_mask=mask,
        live_native_targets=current,live_provided=flags,encoded_user_codes=encoded)
    if native_audio:tensors['generated_codes']=torch.cat(generated,dim=2)
    torch.save(tensors,file)
    result=dict(schema='dl-context-teacher-engine-input-audit-v1',passed=inputs['exact'] and codec['exact'] and ownership,
        teacher=checked,teacher_scope_sha256=sha256(args.teacher_engine_input_audit),
        assets_manifest_sha256=sha256(args.assets),voice_sha256=sha256(args.out/'voice.pt'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256={n:sha256(Path(__file__).with_name(n)) for n in
            ('context_teacher_engine_inputs.py','context_native_layout.py','context_training_data.py')},
        input_comparison=inputs,streaming_user_codec_comparison=codec,all_real_targets_provided=ownership,
        real_to_virtual_exact=mapping,calls=calls,applied_spans=applied,returned_text_tokens=readout,
        file=file.name,sha256=sha256(file),audio_file=wave.name,audio_sha256=sha256(wave),
        native_user_encoder_and_agent_decoder_executed=True,teacher_text_and_audio_forced=True,
        execution_scope='native BF16 model and streaming codecs; eager/no graphs for integer-input audit, not latency qualification',
        virtual_context_is_recorded_teacher_context_not_live_backend=True,execution=snapshot(lm),
        optimizer_steps=0,training_approved=False,quality_qualified=False)
    if native_audio:
        result.update(schema='dl-context-teacher-text-native-audio-probe-v2',
            passed=ownership and native_comparison['exact'] and mapping,
            teacher_text_and_audio_forced=False,teacher_text_forced=True,agent_acoustic_tokens_forced=False,
            expected_text=spm.decode(clean[0,clean[0]>=5].tolist()),
            non_agent_native_input_comparison=native_comparison,
            all_real_targets_provided=False,provided_text_user_and_initial_prefix_only=ownership,
            initial_pending_agent_codes=native_prefix[2:9,-1].tolist(),
            initial_pending_agent_codes_verified=ownership,
            generated_acoustic_frames=int(tensors['generated_codes'].shape[-1]),
            native_reasoning_qualified=False,independent_speech_review='pending')
    if streamed_capture:
        from evals.context_teacher_native_approval import prepared
        prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
        result['streamed_user_capture_control']=streamed_capture_control(
            actual,expected,prepared(example,prefix,encoded,spm),ownership)
    report_name='teacher-text-render-audit.json' if native_audio else 'teacher-engine-input-audit.json'
    (args.out/report_name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    accepted = result['streamed_user_capture_control']['passed'] if streamed_capture else result['passed']
    if not accepted:raise RuntimeError('Prepared teacher differs from the actual native Engine input path')
    return result
