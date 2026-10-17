"""Revalidate saved corrective components before using their frozen features."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256


def validate(folder, data, assets, examples, spm, voice):
    import torch
    from evals.context_history_validate import validate as history_validate
    from evals.context_future_component import plan
    from evals.recovery_prepare_data import spoken_words
    manifest = json.loads((folder/'actual-future.json').read_text())
    history = Path(manifest['actual_history'])
    record = json.loads((history/'student-history.json').read_text())
    source = history_validate(history, data, assets, record['source']['repo_commit'])
    cancel_ack=manifest.get('target_kind')=='cancellation_acknowledgment'
    if cancel_ack:
        from evals.context_cancel_ack import plan as plan_ack
        expected,gold=plan_ack(history,data,examples,spm,manifest['teacher']['id'])
    else:
        expected, gold = plan(history, data, examples, spm)
    # Generation changes only this planner field; all provenance must recompute.
    expected['gold_targets_created'] = True
    if any(manifest.get(k) != v for k, v in expected.items() if k != 'schema'):
        raise RuntimeError('Corrective component differs from revalidated TRAIN selection')
    if (manifest['schema'] != 'dl-context-actual-future-component-v1'
            or manifest['training_approved'] is not False or manifest['past_label_count'] != 0
            or manifest['audio_loss_targets'] != 0 or manifest['file'] != 'actual-future.pt'
            or not all(manifest[k] is True for k in ('forced_gold_future', 'future_text_targets_approved',
                'actual_engine_restart_passed', 'all_past_labels_masked',
                'all_future_user_codes_match_actual_capture', 'audio_targets_forced_for_conditioning_only',
                'native_pending_assistant_residuals_sampled_not_relabelled'))
            or manifest['sha256'] != sha256(folder/'actual-future.pt')
            or manifest['actual_restart_gate_sha256'] != sha256(folder/'engine-replay-before-future.json')):
        raise RuntimeError('Unapproved or changed corrective component')
    commit = manifest['builder_commit']
    if not re.fullmatch('[a-f0-9]{40}', commit):
        raise RuntimeError('Invalid component builder commit')
    names = {'context_future_component.py', 'context_actual_targets.py', 'context_engine_replay.py'}
    if cancel_ack:names.add('context_cancel_ack.py')
    if set(manifest['builder_code_sha256']) != names:
        raise RuntimeError('Incomplete builder provenance')
    for name, digest in manifest['builder_code_sha256'].items():
        content = subprocess.check_output(['git', 'show', commit+':evals/'+name],
            cwd=Path(__file__).resolve().parents[1])
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError('Changed immutable component builder')
    gate = json.loads((folder/'engine-replay-before-future.json').read_text())
    if manifest.get('trained_collector') and (gate.get('conditioning_policy')!='verified-trained-rollout'
            or gate.get('trained_collector')!=manifest['trained_collector']):
        raise RuntimeError('Corrective future lost its exact trained collector identity')
    boundary = manifest['future_first_real_frame']
    live = [c for c in record['calls'] if c['mode'] == 'live']
    stop = live[boundary-1]['start'] + 1
    calls = [c for c in record['calls'] if c['start'] < stop]
    if (gate['schema'] != 'dl-context-engine-restart-gate-v1' or gate['passed'] is not True
            or gate['history'] != str(history) or gate['source'] != source
            or gate['source_execution'] != record['execution']
            or gate['real_frame_boundary'] != manifest['replay_cut_index']
            or gate['checked_real_frames'] != boundary or boundary != manifest['replay_cut_index']+3
            or len(gate['steps']) != boundary or len(gate['inputs']) != len(calls)
            or len(gate['temporal_numerics']) != len(calls)
            or not gate['actual_user_encoder_and_agent_decoder_executed']
            or not gate['actual_receipt_boundaries_replayed'] or gate['new_backend_calls']):
        raise RuntimeError('Missing exact actual-engine restart evidence')
    keys = {'cache_exact','provided_exact','offset_exact','virtual_frames_exact',
        'cpu_rng_exact','cuda_rng_exact','returned_text_exact'}
    for i, step in enumerate(gate['steps']):
        if set(step) != keys|{'real_frame'} or step['real_frame'] != i or not all(step[k] is True for k in keys):
            raise RuntimeError('Actual restart state check failed')
    for i, (inp, num) in enumerate(zip(gate['inputs'], gate['temporal_numerics'])):
        if (inp != {'call':i, 'exact':True} or num != {'call':i,
                'hidden':{'exact':True,'max_abs':0.}, 'text_logits':{'exact':True,'max_abs':0.}}):
            raise RuntimeError('Actual restart input/numerical check failed')
    execution = record['execution']
    if (not execution['math_sdp'] or any(execution[k] for k in ('flash_sdp','cudnn_sdp','mem_efficient_sdp'))
            or any(gate['execution'][k] != v for k,v in execution.items())):
        raise RuntimeError('Changed captured MATH execution')
    actual_voice = torch.load(history/'student-voice.pt', map_location='cpu', weights_only=True)['codes']
    if not torch.equal(voice, actual_voice):
        raise RuntimeError('Feature voice differs from native probe voice')
    saved = torch.load(folder/'actual-future.pt', map_location='cpu', weights_only=True)
    shapes = {'delayed_inputs':(1,17,gold.shape[1]), 'hidden':(1,gold.shape[1],4096),
        'text_targets':(gold.shape[1],), 'text_mask':(gold.shape[1],),
        'provided':(1,17,gold.shape[1]), 'native_current_targets':(1,17,gold.shape[1]),
        'approved_assistant_codes':(8,gold.shape[1])}
    if set(saved) != set(shapes) or any(tuple(saved[k].shape) != s for k,s in shapes.items()):
        raise RuntimeError('Incomplete corrective tensors')
    for key, value in saved.items():
        dtype = torch.bfloat16 if key == 'hidden' else (torch.bool if key in ('text_mask','provided') else torch.int64)
        if value.dtype != dtype or not torch.isfinite(value).all():
            raise RuntimeError('Invalid corrective feature dtype/value')
    mask = (gold[0]>=5)|(gold[0]==0)
    flags = saved['provided']; target = saved['native_current_targets']; inputs = saved['delayed_inputs']
    if (not torch.equal(saved['text_targets'],gold[0]) or not torch.equal(saved['text_mask'],mask)
            or not torch.equal(saved['approved_assistant_codes'],gold[1:9])
            or not torch.equal(target[0,0],gold[0]) or not torch.equal(target[0,1],gold[1])
            or not torch.equal(target[0,2:9,1:],gold[2:9,:-1])
            or not flags[:,0:2].all() or not flags[:,9:17].all()
            or flags[:,2:9,0].any() or not flags[:,2:9,1:].all()
            or not torch.equal(inputs[0,0,1:],gold[0,:-1])
            or not torch.equal(inputs[0,1,1:],gold[1,:-1])
            or not torch.equal(inputs[0,2:9,2:],gold[2:9,:-2])):
        raise RuntimeError('Incorrect native target ownership or extra target delay')
    raw = torch.load(history/'student-history.pt', map_location='cpu', weights_only=True)['delayed_inputs']
    for n in range(gold.shape[1]):
        col = live[boundary+n]['start']
        if not torch.equal(inputs[:,9:17,n],raw[:,9:17,col]):
            raise RuntimeError('Future user inputs differ from actual native capture')
    if not torch.equal(inputs[:,:,0],raw[:,:,live[boundary]['start']]):
        raise RuntimeError('First future input altered the actual past')
    # Measure the first fact token separately; later features contain earlier teacher audio.
    # Source sidecar, already hash checked by load_training/plan, holds measured alignments.
    cache_record = json.loads((Path(manifest['teacher']['cache'])/'cache.json').read_text())
    from evals.recovery_data import validate_conversion
    teacher = next(r for r in validate_conversion(Path(cache_record['conversion']))['accepted']
        if r['scenario_id'] == manifest['teacher']['id'])
    side = json.loads(Path(teacher['path']).with_suffix('.json').read_text())
    words = [a[0] for a in side['alignments'] if a[2] == 'SPEAKER_MAIN']
    selection = manifest['selection']; words = words[selection['source_first_word_index']:selection['source_end_word_index']]
    if cancel_ack:
        ordinal=0
    else:
        value = next(iter(next(iter(manifest['world'].values())).values()))
        phrase = spoken_words(value)
        starts = [i for i in range(len(words)) if spoken_words(' '.join(words[i:i+len(phrase)])) == phrase]
        if len(starts) != 1: raise RuntimeError('Ambiguous first fact target')
        ordinal = sum(len(spm.encode(w.strip())) for w in words[:starts[0]])
    fact_frame = int(torch.where(gold[0]>=5)[0][ordinal])
    audit = {'component':str(folder), 'manifest_sha256':sha256(folder/'actual-future.json'),
        'tensor_sha256':manifest['sha256'], 'restart_gate_sha256':manifest['actual_restart_gate_sha256'],
        'source_history_sha256':source['history_sha256'], 'voice_codes_sha256':record['source']['voice_codes_sha256'],
        'source_execution':execution, 'builder_commit':commit, 'scenario':source['scenario'],
        'world':manifest['world'], 'frames':gold.shape[1], 'text_targets':int(mask.sum()),
        **({'target_kind':'cancellation_acknowledgment','diagnostic_kind':'first_assistant_content',
            'diagnostic_frame':fact_frame,'diagnostic_target':int(gold[0,fact_frame]),
            'private_fact_targets':0,'user_loss_targets':0,'actual_cancellation_completed':True} if cancel_ack else
            {'first_fact_frame':fact_frame,'first_fact_target':int(gold[0,fact_frame])}),
        'past_loss_targets':0, 'audio_loss_targets':0, 'passed':True}
    return audit, saved
