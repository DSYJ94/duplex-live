"""Replay actual TRAIN audio and recorded span boundaries through native Engine.

No API/ASR calls, new dispatch decisions, gold targets or optimizer updates.
Unlike token-only replay, this executes both codecs in their original order.
"""
import json


def run(lm, mimi, spm, voice, histories, data, assets, source_commit, cut_index=None, on_verified=None):
    import numpy as np
    import soundfile as sf
    import torch
    import contextspan.model.engine as native
    from evals.context_history_validate import validate
    from evals.context_execution import restore_attention, snapshot
    from evals.personaplex_adapters import install_adapters
    from evals.context_native_layout import install
    from evals.lychee_context import BASE_INSTRUCTIONS
    if len(histories) != 1:
        raise RuntimeError('Full engine restart requires one history per fresh process')
    path = histories[0]
    check = validate(path, data, assets, source_commit)
    record = json.loads((path/'student-history.json').read_text())
    report = json.loads((path/'report.json').read_text())
    adapter = report['loading']['adapter']
    if (not adapter or not adapter['zero'] or adapter['disabled']
            or not report['loading']['layout'] or report['loading']['input_warmup']):
        raise RuntimeError('Engine replay needs the captured zero-wrapper native-layout/no-audio-warmup baseline')
    if not torch.equal(voice, torch.load(path/'student-voice.pt', weights_only=True)['codes']):
        raise RuntimeError('Engine replay voice differs from actual capture')
    if not record.get('execution'):
        raise RuntimeError('Engine replay requires recorded execution settings')
    restore_attention(record['execution'])
    torch.backends.cudnn.enabled = record['execution']['cudnn']
    torch.backends.cudnn.deterministic = record['execution']['cudnn_deterministic']
    torch.backends.cudnn.benchmark = record['execution']['cudnn_benchmark']
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        install_adapters(lm,last_layers=4)
    lm.eval().requires_grad_(False)
    old_loader = native.load_model
    native.load_model = lambda *a, **kw: (lm, mimi, spm)
    try:
        engine = native.Engine(device='cuda',temp=.8,temp_text=.7)
    finally:
        native.load_model = old_loader
    install(engine)
    native.seed_all(42424242)
    engine.reset()
    saved = torch.load(path/'student-history.pt', map_location='cpu', weights_only=True)
    rows = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    events = json.loads((path/'worker-events.json').read_text())
    spans = [e for e in events if e.get('applied_at_s') is not None and 'virtual_span_frames' in e]
    index = next((i for i in range(len(rows)-2)
        if rows[i]['input_available_s'] >= max(e['applied_at_s'] for e in spans)+.24
        and rows[i+1]['history_virtual_frames_before_step'] == saved['states'][i]['virtual_frames']
        and rows[i+2]['history_virtual_frames_before_step'] == saved['states'][i+1]['virtual_frames']
        and saved['states'][i]['pending_exit_cb0'] is None),None) if spans else None
    if cut_index is not None:
        index = cut_index
        if (not 0 <= index < len(rows)-2
                or rows[index]['input_available_s'] < max(e['applied_at_s'] for e in spans)+.24
                or saved['states'][index]['pending_exit_cb0'] is not None
                or any(rows[i+1]['history_virtual_frames_before_step'] != saved['states'][i]['virtual_frames']
                    for i in (index,index+1))):
            raise RuntimeError('Requested actual cut has an unaccounted receipt or pending state')
    if index is None: raise RuntimeError('No actual bounded post-span restart boundary')
    # The probe consumes the approved FLOAT source directly, then pads20s.
    # Its user_in.wav is PCM16 serialization and is not the lossless input.
    from evals.recovery_data import validate_conversion
    receipt = validate_conversion(data)
    source = next(r for r in receipt['accepted'] if r['scenario_id'] == record['source']['scenario'])
    source_pcm,rate = sf.read(source['path'],dtype='float32',always_2d=True)
    if rate != 24000 or source_pcm.shape[1] != 2 or not np.isfinite(source_pcm).all():
        raise RuntimeError('Invalid independently approved original input')
    pcm = np.pad(source_pcm[:,1],(0,20*rate))
    if len(pcm) != report['real_user_samples_expected']:
        raise RuntimeError('Actual source PCM and recorded native input lengths differ')
    state = engine.lm_gen._streaming_state
    call_index, virtual = 0, 0
    mode = 'prefix'
    numerical, inputs, steps = [], [], []
    original_direct, original_graph = lm.forward_codes, state.graphed_main
    def wrap(original, route):
        def observed(value):
            nonlocal call_index, virtual
            call = record['calls'][call_index]
            expected = saved['delayed_inputs'][:,:,call['start']:call['start']+call['frames']]
            exact = (call['mode']==mode and call['path']==route and value.shape==expected.shape
                and torch.equal(value.cpu(),expected))
            inputs.append({'call':call_index,'exact':exact})
            output = original(value)
            actual = {'hidden':output[0][:,-1:].detach().cpu(),
                'text_logits':output[1][:,:,-1:].detach().cpu()}
            target = saved['temporal_outputs'][call_index]
            numerical.append({'call':call_index,**{k:{'exact':torch.equal(v,target[k]),
                'max_abs':float((v.float()-target[k].float()).abs().max())} for k,v in actual.items()}})
            call_index += 1; virtual += value.shape[-1]
            return output
        return observed
    lm.forward_codes, state.graphed_main = wrap(original_direct,'direct'), wrap(original_graph,'graph')
    try:
        engine.set_persona(BASE_INSTRUCTIONS,voice)
        span_index = 0
        with torch.no_grad():
            for i,row in enumerate(rows[:index+3]):
                # The recorded next live call identifies the actual real frame
                # at which each already delivered receipt was inserted.
                while record['calls'][call_index]['mode']=='span':
                    event = spans[span_index]; mode='span'
                    frames = engine.inject_context_span(event['result']['content'])
                    if frames != event['virtual_span_frames']:
                        raise RuntimeError('Actual replay span length changed')
                    span_index += 1
                mode='live'
                begin=round(row['input_start_s']*rate)
                chunk=pcm[begin:begin+row['input_samples']]
                if len(chunk)<engine.frame_size:chunk=np.pad(chunk,(0,engine.frame_size-len(chunk)))
                result=engine.step(chunk)
                target=saved['states'][i]
                same={'real_frame':i,'cache_exact':torch.equal(state.cache.cpu(),target['cache']),
                    'provided_exact':torch.equal(state.provided.cpu(),target['provided']),
                    'offset_exact':state.offset==target['offset'],
                    'virtual_frames_exact':virtual==target['virtual_frames'],
                    'cpu_rng_exact':torch.equal(torch.get_rng_state(),target['cpu_rng']),
                    'cuda_rng_exact':torch.equal(torch.cuda.get_rng_state(),target['cuda_rng']),
                    'returned_text_exact':result['text_token']==row['text_token']}
                steps.append(same)
    finally:
        lm.forward_codes, state.graphed_main = original_direct, original_graph
    passed=(all(c['exact'] for c in inputs)
        and all(all(s[k] for k in s if k!='real_frame') for s in steps)
        and all(all(n[k]['exact'] for k in ('hidden','text_logits')) for n in numerical))
    result = {'schema':'dl-context-engine-restart-gate-v1','history':str(path),'source':check,
        'source_execution':record['execution'],'execution':snapshot(lm),
        'real_frame_boundary':index,'checked_real_frames':len(steps),'steps':steps,
        'inputs':inputs,'temporal_numerics':numerical,'passed':passed,
        'actual_user_encoder_and_agent_decoder_executed':True,
        'actual_receipt_boundaries_replayed':True,'new_backend_calls':False,
        'training_approved':False,'gold_targets_created':False,
        'future_component':None,'microphone_tested':False}
    if passed and on_verified:
        result['future_component']=on_verified(engine,saved,index+2,rows,pcm,result)
        result['gold_targets_created']=True
    return result
