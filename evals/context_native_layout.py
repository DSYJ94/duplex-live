"""Explicit candidate compatibility with clean native training coordinates.

Bootstrap the omitted prefix column. Preserve only actually provided pending
acoustics across virtual spans; never copy ungenerated assistant predictions.
Vendor code and released default remain unchanged. Require a native trace test.
"""


def span_prefill(gen, ids, agent_sil, user_sine, preserve_provided=False):
    from contextspan.model.context_span_block import prefill
    pending = []
    state = gen._streaming_state
    if preserve_provided:
        position = state.offset % state.cache.shape[2]
        for row in range(1, gen.lm_model.num_codebooks):
            delay = gen.lm_model.delays[row]
            if delay not in (0, 1):
                raise RuntimeError('Only the pinned one-frame acoustic delay is supported')
            if delay == 1:
                pending.append((row, state.cache[:, row, position].clone(),
                    state.provided[:, row, position].clone()))
    n = prefill(gen, ids, agent_sil, user_sine)
    if preserve_provided:
        import torch
        position = state.offset % state.cache.shape[2]
        for row, value, provided in pending:
            state.cache[:, row, position] = torch.where(provided, value, state.cache[:, row, position])
            state.provided[:, row, position] |= provided
    return n


def install(engine):
    """Apply only at a new conversation boundary, before persona prefill."""
    import torch
    from types import MethodType
    from contextspan.model.context_span_block import span_ids, pending_exit_cb0
    import time
    original = engine._prefix_steps
    def prefix(self, *args, **kwargs):
        bootstrap = (self._sine, self._sil,
            torch.tensor([self.lm_gen.zero_text_code], device=self.device, dtype=torch.long))
        return [bootstrap] + original(*args, **kwargs)
    @torch.no_grad()
    def inject(self, reference):
        self._pending_exit_cb0 = pending_exit_cb0(self.lm_gen)
        started = time.perf_counter()
        n = span_prefill(self.lm_gen, span_ids(reference, self.spm), self._sil, self._sine, True)
        torch.cuda.synchronize()
        self.last_prefill_ms = (time.perf_counter() - started) * 1000
        return n
    engine._prefix_steps = MethodType(prefix, engine)
    engine.inject_context_span = MethodType(inject, engine)
