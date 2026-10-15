"""Passive provided-token checks around actual free-running result spans."""


def attach(engine):
    import torch
    records = []
    original = engine.inject_context_span
    def checked(reference):
        state = engine.lm_gen._streaming_state
        position = state.offset % state.cache.shape[2]
        pending = [(row, state.cache[:, row, position].clone(), state.provided[:, row, position].clone())
            for row, delay in enumerate(engine.lm.delays) if row and delay == 1]
        assistant = sum(int(p.sum()) for row, _, p in pending if row <= 8)
        user = sum(int(p.sum()) for row, _, p in pending if row >= 9)
        if assistant != 0 or user != 7:
            raise RuntimeError('Free-running span did not have exactly the seven observed pending user acoustics')
        frames = original(reference)
        position = state.offset % state.cache.shape[2]
        for row, value, provided in pending:
            if not torch.equal(state.cache[:, row, position][provided], value[provided]) or not (
                    state.provided[:, row, position][provided]).all():
                raise RuntimeError('Actual observed pending audio changed across a context span')
        records.append({'assistant_pending_provided': assistant, 'user_pending_provided': user,
            'all_provided_codes_and_flags_preserved': True, 'ungenerated_assistant_codes_copied': False,
            'virtual_frames': frames})
        return frames
    engine.inject_context_span = checked
    return records
