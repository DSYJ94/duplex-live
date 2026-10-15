"""Trace actual native prefix/span/live inputs against the clean training oracle.

Teacher-forced TRAIN component only. No words are generated for evaluation.
"""


def audit(lm, spm, prefix, example, sample):
    import torch
    from contextspan.moshi.models import LMGen
    from contextspan.moshi.models.lm import _delay_sequence
    from contextspan.moshi.utils.compile import no_compile, no_cuda_graph
    from contextspan.model.prefix_prefill import prefill_forced_steps
    from contextspan.model.sequence_convention import SILENCE_TOKENS, SINE_TOKENS, context_span_ids
    from evals.context_native_layout import span_prefill
    span = example['record']['spans'][0]
    stop = span['inject_frame'] + 2
    codes = sample['codes'].to(lm.device)
    ag = torch.tensor(SILENCE_TOKENS, device=lm.device)[None, :, None]
    us = torch.tensor(SINE_TOKENS, device=lm.device)[None, :, None]
    text = lambda x: torch.tensor([int(x)], dtype=torch.long, device=lm.device)
    steps = [(prefix[9:17, f][None, :, None].to(lm.device),
        prefix[1:9, f][None, :, None].to(lm.device), text(prefix[0, f])) for f in range(prefix.shape[1])]
    results = {}
    original = lm.forward_codes
    for aligned in (False, True):
        observed = []
        def capture(value):
            observed.append(value.detach().cpu().clone())
            return original(value)
        lm.forward_codes = capture
        try:
            with torch.no_grad(), no_compile(), no_cuda_graph():
                gen = LMGen(lm, device=str(lm.device), use_sampling=False,
                    sample_rate=24000, frame_rate=12.5)
                with gen.streaming(1):
                    supplied = ([(us, ag, text(3))] if aligned else []) + steps
                    prefill_forced_steps(gen, supplied)
                    prefix_offset = gen._streaming_state.offset
                    for f in range(stop):
                        if f == span['inject_frame']:
                            span_prefill(gen, context_span_ids(span['reference'], spm), ag, us, aligned)
                        gen.step(input_tokens=codes[:, 9:17, f:f+1],
                            moshi_tokens=codes[:, 1:9, f:f+1], text_token=codes[:, 0, f])
        finally:
            lm.forward_codes = original
        actual = torch.cat(observed, dim=2)
        clean = example['codes'][None]
        initial = lm._get_initial_token().cpu()
        expected = torch.cat([initial, _delay_sequence(lm.delays, clean, initial)], dim=2)
        # Each actual forward predicts its target; omit the unconsumed future.
        expected = expected[:, :, :actual.shape[2]]
        mismatch = (actual != expected)
        positions = mismatch.nonzero().tolist()
        # Source-derived startup-only correction, retained as a diagnostic.
        dropped = torch.cat([initial, _delay_sequence(lm.delays, clean[:, :, 1:], initial)], dim=2)
        dropped = dropped[:, :, :actual.shape[2]]
        results['aligned_candidate' if aligned else 'released'] = {
            'actual_shape': list(actual.shape), 'prefix_offset_after_prefill': prefix_offset,
            'all_input_ids_exact': not bool(mismatch.any()),
            'mismatches_per_channel': mismatch.sum((0, 2)).tolist(),
            'first_mismatches_batch_channel_frame': positions[:24],
            'startup_dropped_oracle_mismatches_per_channel': (actual != dropped).sum((0, 2)).tolist(),
            'actual_input_sha256': __import__('hashlib').sha256(actual.numpy().tobytes()).hexdigest(),
            'expected_input_sha256': __import__('hashlib').sha256(expected.numpy().tobytes()).hexdigest()}
    return {'schema': 'dl-context-native-history-audit-v1', 'source_split': 'train',
        'scenario': example['record']['id'], 'clean_real_frames_used': stop,
        'span': span, 'native_default_unchanged': True,
        'production_audio_or_task_tested': False, 'results': results,
        'candidate_passed': results['aligned_candidate']['all_input_ids_exact']}
