"""Native depth parity and precise adapter checks, without changing vendor code."""
from contextlib import contextmanager


@contextmanager
def precise_native(model):
    import torch
    from contextspan.moshi.utils.compile import no_compile, no_cuda_graph
    from torch.nn.attention import sdpa_kernel, SDPBackend
    tensors = [(p, p.dtype) for p in model.parameters() if p.is_floating_point()]
    tensors += [(b, b.dtype) for b in model.buffers() if b.is_floating_point()]
    tf32 = torch.backends.cuda.matmul.allow_tf32
    for tensor, _ in tensors:
        tensor.data = tensor.data.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with no_compile(), no_cuda_graph(), sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        for tensor, dtype in tensors:
            tensor.data = tensor.data.to(dtype=dtype)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def depth_parity(lm, codes, indices):
    import torch
    from contextspan.moshi.models.lm import _delay_sequence
    from evals.personaplex_depth import native_depth_context
    evidence = []
    with torch.no_grad(), precise_native(lm):
        initial = lm._get_initial_token()
        delayed = torch.cat([initial, _delay_sequence(lm.delays, codes, initial)], dim=2)
        hidden, _ = lm.forward_codes(delayed[:, :, :-1])
        for index in indices:
            targets = delayed[:, :, index+1:index+2]
            state = hidden[:, index:index+1]
            original = lm.forward_depformer_training(targets, state)
            stream = []
            with lm.depformer.streaming(1):
                for q in range(16):
                    # Text for q0; observed/forced preceding audio for q>0.
                    stream.append(lm.forward_depformer(q, targets[:, q:q+1], state))
            streamed = torch.cat(stream, dim=1)
            with native_depth_context(lm):
                matched = lm.forward_depformer_training(targets, state)
            errors = (matched-streamed).abs().amax((0, 2, 3)).tolist()
            evidence.append({'delayed_frame': index,
                'published_batch_per_head_max_error': (original-streamed).abs().amax((0, 2, 3)).tolist(),
                'context15_batch_per_head_max_error': errors})
            if max(errors) > .001:
                raise RuntimeError('Native depth streaming/training parity failed')
    return {'passed': True, 'dtype': 'float32', 'threshold': .001, 'frames': evidence,
        'compatibility': 'batch depth context15; native inference unchanged'}
