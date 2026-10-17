"""Independently approve only text labels in a quarantined early capture."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256


def validate(folder, data, assets, examples, spm, voice):
    import torch
    from evals.context_history_validate import validate as validate_history
    from evals.context_future_component import plan as late_plan
    from evals.context_early_plan import plan
    m = json.loads((folder/'early-future.json').read_text()); history = Path(m['actual_history'])
    record = json.loads((history/'student-history.json').read_text())
    source = validate_history(history,data,assets,record['source']['repo_commit'])
    late,_ = late_plan(history,data,examples,spm,early=True,teacher_id=m['teacher']['id'])
    expected,gold = plan(late,examples)
    expected['gold_targets_created'] = True
    if any(m.get(k) != v for k,v in expected.items() if k != 'schema'):
        raise RuntimeError('Early capture differs from independently revalidated TRAIN plan')
    if (m['schema'] != 'dl-context-early-future-capture-v1' or not m['capture_complete']
            or m['training_approved'] is not False or m['future_text_targets_approved'] is not False
            or not m['requires_independent_component_audit_before_any_optimizer']
            or m['file'] != 'early-future.pt' or sha256(folder/m['file']) != m['sha256']
            or m['past_loss_targets'] != 0 or m['audio_loss_targets'] != 0
            or m['voice_codes_sha256'] != record['source']['voice_codes_sha256']
            or m['source_execution'] != record['execution']
            or not m['all_future_user_codes_match_actual_capture']):
        raise RuntimeError('Changed or ineligible quarantined early capture')
    if not torch.equal(voice,torch.load(folder/'voice.pt',map_location='cpu',weights_only=True)['codes']):
        raise RuntimeError('Early capture voice differs from verified history voice')
    commit = m['builder_commit']; names = {'context_early_future.py','context_early_plan.py','context_engine_replay.py'}
    ordinary = m.get('target_kind') == 'ordinary_answer'
    if ordinary:names.add('context_ordinary_future.py')
    if not re.fullmatch('[a-f0-9]{40}',commit) or set(m['builder_code_sha256']) != names:
        raise RuntimeError('Incomplete early builder provenance')
    for name,digest in m['builder_code_sha256'].items():
        content = subprocess.check_output(['git','show',commit+':evals/'+name],cwd=Path(__file__).resolve().parents[1])
        if hashlib.sha256(content).hexdigest() != digest:raise RuntimeError('Changed early builder code')
    gate_file = folder/'engine-replay-before-future.json'; gate = json.loads(gate_file.read_text())
    begin = m['future_first_real_frame']; end = m['future_end_real_frame']; length = end-begin
    live = [c for c in record['calls'] if c['mode']=='live']; stop = live[begin]['start']
    prefix_calls = [c for c in record['calls'] if c['start']<stop]
    if (sha256(gate_file) != m['actual_restart_gate_sha256'] or gate['passed'] is not True
            or gate['schema'] != 'dl-context-engine-restart-gate-v1' or gate['source'] != source
            or gate['history'] != str(history) or gate['checked_real_frames'] != begin
            or gate['real_frame_boundary'] != m['replay_cut_index'] or begin != m['replay_cut_index']+3
            or len(gate['steps']) != begin or len(gate['inputs']) != len(prefix_calls)
            or len(gate['temporal_numerics']) != len(prefix_calls)
            or gate['source_execution'] != record['execution'] or gate['execution'] != record['execution']
            or gate['new_backend_calls'] or not gate['actual_receipt_boundaries_replayed']
            or not gate['actual_user_encoder_and_agent_decoder_executed']):
        raise RuntimeError('Early actual-prefix restart evidence changed')
    state_keys = {'cache_exact','provided_exact','offset_exact','virtual_frames_exact',
        'cpu_rng_exact','cuda_rng_exact','returned_text_exact'}
    for i,s in enumerate(gate['steps']):
        if set(s) != state_keys|{'real_frame'} or s['real_frame'] != i or not all(s[k] is True for k in state_keys):
            raise RuntimeError('Incomplete exact early restart state checks')
    for i,(inp,num) in enumerate(zip(gate['inputs'],gate['temporal_numerics'])):
        if inp != {'call':i,'exact':True} or num != {'call':i,
                'hidden':{'exact':True,'max_abs':0.},'text_logits':{'exact':True,'max_abs':0.}}:
            raise RuntimeError('Early restart numerical checks failed')
    raw = torch.load(history/'student-history.pt',map_location='cpu',weights_only=True)['delayed_inputs']
    saved = torch.load(folder/'early-future.pt',map_location='cpu',weights_only=True)
    total = m['virtual_future_frames']
    mode=m.get('future_audio_conditioning','approved_teacher')
    if mode not in ('approved_teacher','native_sampled'):
        raise RuntimeError('Unknown future audio-conditioning policy')
    own_audio=mode=='native_sampled'
    if (m.get('future_assistant_audio_tokens_forced',True) is not (not own_audio)
            or m.get('teacher_audio_codes_are_reference_only',False) is not own_audio
            or m.get('future_text_tokens_forced',True) is not True):
        raise RuntimeError('Future acoustic ownership declaration changed')
    shapes = {'delayed_inputs':(1,17,total),'hidden':(1,total,4096),
        'text_targets':(total,),'text_mask':(total,),'live_native_targets':(1,17,length),
        'live_provided':(1,17,length),'approved_assistant_codes':(8,length)}
    if own_audio:shapes['generated_delayed_audio']=(1,8,length)
    if set(saved) != set(shapes) or any(tuple(saved[k].shape) != shape for k,shape in shapes.items()):
        raise RuntimeError('Invalid early tensor coverage')
    for k,v in saved.items():
        dtype = torch.bfloat16 if k=='hidden' else (torch.bool if k in ('text_mask','live_provided') else torch.int64)
        if v.dtype != dtype or not torch.isfinite(v).all():raise RuntimeError('Invalid early tensor dtype/value')
    indices = m['real_to_future_virtual']; cursor = n = 0; span_calls = []
    for call in m['calls']:
        if call['start'] != cursor or call['frames']<1:raise RuntimeError('Noncontiguous future calls')
        if call['phase']=='live':
            if call['route']!='graph' or call['frames']!=1 or n>=length or indices[n]!=cursor:
                raise RuntimeError('Incorrect future real/virtual mapping')
            n += 1
        elif call['phase']=='span' and call['route']=='direct':
            span_calls.append((begin+n,cursor,call['frames']))
        else:raise RuntimeError('Unknown early forward route')
        cursor += call['frames']
    expected_spans = m['future_recorded_receipts']; applied = m['applied_recorded_receipts']
    if (cursor != total or n != length or len(indices) != length
            or span_calls != [(s['real_frame'],s['virtual_start'],s['virtual_frames']) for s in applied]
            or [(s['real_frame'],s['event']) for s in applied] != [(s['real_frame'],s['event']) for s in expected_spans]
            or any(s['virtual_frames'] != s['event']['virtual_span_frames'] for s in applied)):
        raise RuntimeError('Future spans differ from actual recorded receipt boundaries')
    labels = torch.full((total,),-1,dtype=torch.int64); labels[indices] = gold[0]
    mask = torch.zeros(total,dtype=torch.bool); mask[indices] = (gold[0]>=4)|(gold[0]==0)
    current = saved['live_native_targets']; flags = saved['live_provided']; inputs = saved['delayed_inputs']
    if (not torch.equal(saved['text_targets'],labels) or not torch.equal(saved['text_mask'],mask)
            or not torch.equal(saved['approved_assistant_codes'],gold[1:9])
            or not torch.equal(current[0,0],gold[0])
            or not flags[:,0].all() or not flags[:,9:17].all()):
        raise RuntimeError('Early labels acquired a span, wrong delay or unprovided acoustic target')
    if own_audio:
        import soundfile as sf
        generated=saved['generated_delayed_audio']
        if flags[:,1:9].any() or (generated<0).any() or (generated>=2048).any():
            raise RuntimeError('Native future acoustics were provided or out of range')
        if m['native_audio_file']!='early-native-audio.wav' or sha256(folder/m['native_audio_file'])!=m['native_audio_sha256']:
            raise RuntimeError('Changed native future audio artifact')
        audio,rate=sf.read(folder/m['native_audio_file'],dtype='float32')
        if (rate!=24000 or audio.ndim!=1 or len(audio)!=length*1920
                or m['native_audio_samples']!=len(audio) or not torch.isfinite(torch.from_numpy(audio)).all()):
            raise RuntimeError('Incomplete native future audio')
    elif (not torch.equal(current[0,1],gold[1])
            or not torch.equal(current[0,2:9,1:],gold[2:9,:-1])
            or flags[:,2:9,0].any() or not flags[:,2:9,1:].all() or not flags[:,1].all()):
        raise RuntimeError('Teacher future acoustic ownership changed')
    if not torch.equal(inputs[:,:,indices[0]],raw[:,:,stop]):
        raise RuntimeError('Early first input changed its actual past')
    for n,v in enumerate(indices):
        if not torch.equal(inputs[:,9:17,v],raw[:,9:17,live[begin+n]['start']]):
            raise RuntimeError('Early future omitted or changed observed user speech')
        if n>=1 and v-indices[n-1]==1:
            if int(inputs[0,0,v])!=int(gold[0,n-1]):raise RuntimeError('Future text input delay changed')
            if own_audio:
                if not torch.equal(inputs[0,1:9,v],generated[0,:,n-1]):
                    raise RuntimeError('Future does not condition on its own generated delayed acoustics')
            elif int(inputs[0,1,v])!=int(gold[1,n-1]):raise RuntimeError('Future semantic input delay changed')
        if not own_audio and n>=2 and v-indices[n-2]==2:
            if not torch.equal(inputs[0,2:9,v],gold[2:9,n-2]):raise RuntimeError('Future residual input delay changed')
    # Cache construction encodes each measured word separately, including quotes.
    # Encoding the entire unquoted value can produce different SentencePiece IDs.
    # Locate the fact in the independently selected measured sentence instead.
    from evals.recovery_data import validate_conversion
    from evals.recovery_prepare_data import spoken_words
    cache_record = json.loads((Path(m['teacher']['cache'])/'cache.json').read_text())
    teacher = next(r for r in validate_conversion(Path(cache_record['conversion']))['accepted']
        if r['scenario_id'] == m['teacher']['id'])
    side = json.loads(Path(teacher['path']).with_suffix('.json').read_text())
    words = [a[0] for a in side['alignments'] if a[2] == 'SPEAKER_MAIN']
    selection = late['selection']; left = selection['source_first_word_index']
    sentence = words[left:selection['source_end_word_index']]
    if ordinary:
        if m['world'] != {} or m['future_recorded_receipts'] or m['actual_private_receipt'] is not None:
            raise RuntimeError('Ordinary component acquired a private receipt or backend hint')
        ordinal = 0
    else:
        value = next(iter(next(iter(m['world'].values())).values())); phrase = spoken_words(value)
        starts = [i for i in range(len(sentence))
            if spoken_words(' '.join(sentence[i:i+len(phrase)])) == phrase]
        if not phrase or len(starts) != 1:raise RuntimeError('Ambiguous measured private fact')
        ordinal = sum(len(spm.encode(w.strip())) for w in words[:left+starts[0]])
    example = next(e for e in examples if e['record'] == m['teacher'])
    source_frame = int((example['clean_codes'][0]>=5).nonzero().flatten()[ordinal])
    if not (selection['source_start_frame'] <= source_frame < selection['source_end_frame']
            and begin <= source_frame < end):
        raise RuntimeError('Measured diagnostic content outside approved future sentence')
    fact_frame = indices[source_frame-begin]
    diagnostic = ({'diagnostic_frame':fact_frame,'diagnostic_target':int(saved['text_targets'][fact_frame]),
        'diagnostic_kind':'first_assistant_content','target_kind':'ordinary_answer',
        'ordinary_answer_without_backend_context':True} if ordinary else
        {'first_fact_frame':fact_frame,'first_fact_target':int(saved['text_targets'][fact_frame])})
    # The capture's original sparse mask is checked above, not rewritten on disk.
    # Full early continuations must also learn when to stay silent. Only real
    # frames have nonnegative labels; virtual context spans retain -1/no loss.
    full_mask = labels >= 0
    if int(full_mask.sum()) != length or full_mask.sum() <= mask.sum():
        raise RuntimeError('Incomplete real-frame silence supervision')
    approved = {**saved, 'text_mask':full_mask}
    result = {'schema':'dl-context-early-future-audit-v2','component':str(folder),'passed':True,
        'capture_manifest_sha256':sha256(folder/'early-future.json'),'tensor_sha256':m['sha256'],
        'restart_gate_sha256':m['actual_restart_gate_sha256'],'source_history_sha256':source['history_sha256'],
        'manifest_sha256':sha256(folder/'early-future.json'),'voice_codes_sha256':m['voice_codes_sha256'],
        'source_execution':m['source_execution'],'builder_commit':m['builder_commit'],
        'future_audio_conditioning':mode,'future_assistant_audio_tokens_forced':not own_audio,
        'scenario':source['scenario'],'world':m['world'],'frames':total,'text_targets':int(full_mask.sum()),
        **diagnostic,
        'teacher':m['teacher'],'actual_real_prefix_frames':begin,'real_future_frames':length,
        'virtual_future_frames':total,'text_loss_targets':int(full_mask.sum()),'request_targets':int((gold[0]==4).sum()),
        'source_capture_sparse_mask_targets':int(mask.sum()),
        'silence_pad3_targets':int((labels==3).sum()),'silence_pad3_loss_weight':.5,
        'loss_policy':'all independently verified real future text; PAD3 half weight; spans and past excluded',
        'past_loss_targets':0,'span_loss_targets':0,'audio_loss_targets':0,
        'native_assistant_text_component_approved':True,'replayed_future_receipts_not_live_dispatch':not ordinary,
        'whole_model_quality_or_delegation_qualified':False}
    return result,approved
