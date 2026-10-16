"""Actual TRAIN token-state restart gate, before any forced-gold targets.

Replay the captured direct/graph inputs, restore small generator state and RNG,
then require the next two unforced steps to match actual recorded snapshots.
No PCM, user encoder or microphone restart is claimed.
"""
import json


def run(lm, voice, histories, data, assets, source_commit, warm_graphs=True, native_priming=False):
    import torch
    from contextspan.moshi.models import LMGen
    from evals.context_history_validate import validate
    from evals.personaplex_adapters import install_adapters
    checks = [validate(p, data, assets, source_commit) for p in histories]
    records = [json.loads((p / 'student-history.json').read_text()) for p in histories]
    if any(r['schema'] not in ('dl-context-student-history-v2','dl-context-student-history-v3') for r in records):
        raise RuntimeError('Actual restart gate needs recorded generator snapshots')
    for path in histories:
        report = json.loads((path / 'report.json').read_text())
        adapter = report['loading']['adapter']
        if not adapter or not adapter['zero'] or adapter['disabled']:
            raise RuntimeError('Initial restart gate supports the recorded zero-wrapper baseline only')
        if not torch.equal(voice, torch.load(path / 'student-voice.pt', weights_only=True)['codes']):
            raise RuntimeError('Actual history and replay voice codes differ')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        install_adapters(lm, last_layers=4)
    lm.eval().requires_grad_(False)
    # Match the pinned probe's ordinary inference flags, rather than its trainer.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_cudnn_sdp(True)

    def user_codes(snapshot):
        cache, offset = snapshot['cache'], snapshot['offset']
        ring = cache.shape[2]
        # The semantic code was supplied at offset-1; the seven actually
        # provided delayed residuals remain pending at offset.
        return torch.cat([cache[:, 9:10, (offset-1) % ring:(offset-1) % ring+1],
            cache[:, 10:17, offset % ring:offset % ring+1]], dim=1).to(lm.device)

    def temporal_state():
        positions, pointers = {}, {}
        for name, module in lm.transformer.named_modules():
            state = getattr(module, '_streaming_state', None)
            offset = getattr(state, 'offset', None)
            if isinstance(offset, torch.Tensor):
                positions[name + '.offset'] = int(offset.item())
                pointers[name + '.offset'] = offset.data_ptr()
            kv = getattr(state, 'kv_cache', None)
            if kv is not None:
                positions[name + '.kv'] = int(kv.end_offset.item())
                pointers[name + '.kv'] = kv.cache.data_ptr()
        if not positions:
            raise RuntimeError('No native temporal state was inspected')
        return positions, pointers

    from evals.context_execution import snapshot as execution_snapshot
    execution = execution_snapshot(lm)
    from evals.context_execution import restore_attention
    executions = [r.get('execution') for r in records]
    if any(e is not None for e in executions):
        if any(e is None or e != executions[0] for e in executions):
            raise RuntimeError('Actual restart batch mixes execution settings')
        restore_attention(executions[0])
        execution = execution_snapshot(lm)
    results = []
    for path, record, check in zip(histories, records, checks):
        saved = torch.load(path / 'student-history.pt', map_location='cpu', weights_only=True)
        rows = [json.loads(line) for line in (path / 'events.jsonl').read_text().splitlines()]
        # Fixed actual post-receipt boundary; this gate does not choose a new
        # speaking time or create supervision. Skip an intervening virtual span.
        events = json.loads((path / 'worker-events.json').read_text())
        applied = [e['applied_at_s'] for e in events if e.get('applied_at_s') is not None]
        if not applied: raise RuntimeError('Restart component requires an actual delivered span')
        index = next((i for i in range(len(rows)-2)
            if rows[i]['input_available_s'] >= max(applied) + .24
            and rows[i+1]['history_virtual_frames_before_step'] == saved['states'][i]['virtual_frames']
            and rows[i+2]['history_virtual_frames_before_step'] == saved['states'][i+1]['virtual_frames']
            and saved['states'][i]['pending_exit_cb0'] is None), None)
        if index is None: raise RuntimeError('No bounded actual continuation without an intervening span')
        snapshot = saved['states'][index]
        boundary = snapshot['virtual_frames']
        if boundary >= 3000: raise RuntimeError('Restart component exceeds pinned temporal context')
        with torch.no_grad():
            if native_priming:
                # Match Engine.__init__'s process-global lazy-kernel priming.
                # Resetting streams does not reset those compiled closures.
                from contextspan.model.sequence_convention import (SILENCE_TOKENS, SINE_TOKENS,
                    SPAN_OPEN_ID, SPAN_CLOSE_ID, TEXT_PAD)
                from contextspan.model.context_span_block import prefill
                primer = LMGen(lm, device=str(lm.device), temp=.8, temp_text=.7,
                    sample_rate=24000, frame_rate=12.5)
                with primer.streaming(1):
                    sine = torch.tensor(SINE_TOKENS, device=lm.device)[None, :, None]
                    silence = torch.tensor(SILENCE_TOKENS, device=lm.device)[None, :, None]
                    for _ in range(4): primer.step(input_tokens=sine)
                    prefill(primer, [SPAN_OPEN_ID] + [TEXT_PAD]*6 + [SPAN_CLOSE_ID], silence, sine)
                del primer
            gen = LMGen(lm, device=str(lm.device), temp=.8, temp_text=.7,
                sample_rate=24000, frame_rate=12.5)
            with gen.streaming(1):
                state = gen._streaming_state
                graphs = (id(state.graphed_main), id(state.graphed_depth))
                if warm_graphs:
                    for _ in range(16): gen.step(input_tokens=user_codes(snapshot))
                _, pointers = temporal_state()
                gen.reset_streaming()
                positions, after_reset = temporal_state()
                if pointers != after_reset or any(positions.values()):
                    raise RuntimeError('In-place reset changed KV storage or retained temporal positions')
                numerical = []
                prefix_trace = None
                if record.get('prefix_trace'):
                    from evals.context_prefix_trace import PrefixTrace
                    from evals.moshi_model_download import sha256
                    trace_file = path / record['prefix_trace']['file']
                    if sha256(trace_file) != record['prefix_trace']['sha256']:
                        raise RuntimeError('Actual prefix evidence hash changed')
                    expected_prefix = torch.load(trace_file, weights_only=True)
                    prefix_trace = PrefixTrace(lm, lambda: call['mode'] == 'prefix')
                def compare(call_index, output):
                    if 'temporal_outputs' not in saved: return
                    expected = saved['temporal_outputs'][call_index]
                    actual = {'hidden': output[0][:, -1:].detach().cpu(),
                        'text_logits': output[1][:, :, -1:].detach().cpu()}
                    numerical.append({'call': call_index, **{key: {
                        'exact': torch.equal(actual[key], expected[key]),
                        'max_abs': float((actual[key].float()-expected[key].float()).abs().max())}
                        for key in actual}})
                for call_index, call in enumerate(record['calls']):
                    begin, end = call['start'], call['start'] + call['frames']
                    if begin >= boundary: break
                    if end > boundary: raise RuntimeError('Restart boundary cuts a captured native forward')
                    inputs = saved['delayed_inputs'][:, :, begin:end].to(lm.device)
                    output = (state.graphed_main if call['path'] == 'graph' else lm.forward_codes)(inputs)
                    compare(call_index, output)
                prefix_numerics = None
                if prefix_trace:
                    prefix_trace.close()
                    if set(prefix_trace.values) != set(expected_prefix):
                        raise RuntimeError('Prefix layer evidence is incomplete')
                    prefix_numerics = {k: {'exact': torch.equal(v, expected_prefix[k]),
                        'max_abs': float((v.float()-expected_prefix[k].float()).abs().max())}
                        for k,v in prefix_trace.values.items()}
                positions, after_replay = temporal_state()
                if pointers != after_replay or any(v != boundary for v in positions.values()):
                    raise RuntimeError('Actual temporal replay changed KV storage or frame positions')
                if graphs != (id(state.graphed_main), id(state.graphed_depth)):
                    raise RuntimeError('Restart replaced the warmed native graph wrappers')
                state.cache.copy_(snapshot['cache'].to(lm.device))
                state.provided.copy_(snapshot['provided'].to(lm.device))
                state.offset = snapshot['offset']
                torch.set_rng_state(snapshot['cpu_rng'])
                torch.cuda.set_rng_state(snapshot['cuda_rng'])
                original_main = state.graphed_main
                next_calls = [i for i,c in enumerate(record['calls']) if c['start'] >= boundary]
                next_index = iter(next_calls)
                next_input_checks = []
                def checked_main(value):
                    i = next(next_index)
                    original = saved['delayed_inputs'][:, :, record['calls'][i]['start']:record['calls'][i]['start']+1]
                    next_input_checks.append({'call': i, 'exact': torch.equal(value.cpu(), original)})
                    output = original_main(value)
                    compare(i, output)
                    return output
                state.graphed_main = checked_main
                steps = []
                for following in (index+1, index+2):
                    expected = saved['states'][following]
                    output = gen.step(input_tokens=user_codes(expected))
                    torch.cuda.synchronize()
                    same = {'cache_exact': torch.equal(state.cache.cpu(), expected['cache']),
                        'provided_exact': torch.equal(state.provided.cpu(), expected['provided']),
                        'offset_exact': state.offset == expected['offset'],
                        'cpu_rng_exact': torch.equal(torch.get_rng_state(), expected['cpu_rng']),
                        'cuda_rng_exact': torch.equal(torch.cuda.get_rng_state(), expected['cuda_rng']),
                        'returned_text_exact': output is not None and int(output[0, 0, 0]) == rows[following]['text_token']}
                    same['cache_mismatches_per_channel'] = (state.cache.cpu() != expected['cache']).sum((0, 2)).tolist()
                    steps.append(same)
                passed = all(c['exact'] for c in next_input_checks) and all(all(s[k] for k in ('cache_exact','provided_exact','offset_exact',
                    'cpu_rng_exact','cuda_rng_exact','returned_text_exact')) for s in steps)
        results.append({'execution': execution, 'source_execution': record.get('execution'), 'history': str(path), 'source': check, 'real_frame_index': index,
            'virtual_boundary': boundary, 'steps': steps, 'temporal_numerics': numerical, 'next_input_checks': next_input_checks, 'prefix_numerics': prefix_numerics, 'passed': passed})
    return {'schema': 'dl-context-actual-restart-gate-v1', 'results': results,
        'graph_warmup_before_replay': warm_graphs, 'native_constructor_priming': native_priming,
        'passed': all(r['passed'] for r in results), 'training_approved': False,
        'gold_targets_created': False, 'PCM_or_microphone_restart_tested': False}
