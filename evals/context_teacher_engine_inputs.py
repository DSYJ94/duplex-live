"""Compare a complete approved teacher with the actual native Engine input path.

Teacher text and assistant acoustics are forced for this engineering audit.
Both codecs execute; no backend, optimizer or native task claim is involved.
"""
import hashlib
import json
from pathlib import Path
from evals.moshi_model_download import sha256


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
    audit=json.loads(args.teacher_engine_input_audit.read_text())
    if (audit['schema']!='dl-context-all-teacher-scope-audit-v1' or not audit['passed']
            or audit['training_approved'] is not False or audit['optimizer_steps']!=0
            or audit['code_sha256']!=sha256(Path(__file__).with_name('context_retention_teacher.py'))
            or audit['voice_sha256']!=sha256(args.out/'voice.pt')
            or audit['assets_manifest_sha256']!=sha256(args.assets)):
        raise RuntimeError('Teacher Engine audit requires its matching independent all-teacher scope')
    example=next(e for e in examples if e['record']['id']==args.teacher_engine_input_id)
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
    chunks,calls,provided,targets,user_codes,packets=[],[],[],[],[],[]
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
            kw.update(moshi_tokens=clean[None,1:9,frame:frame+1].to(engine.device),
                text_token=clean[0,frame:frame+1].to(engine.device))
        return original_step(*a,**kw)
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
    ownership=bool(flags.all() and torch.equal(current[0,0],clean[0]))
    mapping=live_indices==record['real_to_virtual']
    if not mapping:raise RuntimeError('Teacher Engine real/virtual mapping changed')
    wave=args.out/'teacher-forced-native.wav';sf.write(wave,np.concatenate(packets),24000,subtype='FLOAT')
    file=args.out/'teacher-engine-inputs.pt'
    torch.save(dict(delayed_inputs=actual,expected_prepared_inputs=expected,text_targets=labels,text_mask=mask,
        live_native_targets=current,live_provided=flags,encoded_user_codes=encoded),file)
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
    (args.out/'teacher-engine-input-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('Prepared teacher differs from the actual native Engine input path')
    return result
