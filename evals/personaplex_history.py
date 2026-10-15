"""Build teacher-forced native inputs from a captured, fixed voice/system prefix.

Each model input uses only earlier target codes, with the native channel delay.
The captured boundary supplies the first delayed frames; it contains no current
conversation codes. Marker targets remain separate from PersonaPlex's vocabulary.
"""


def delayed_history(template_inputs, prefix_frames, codes, delays):
    import torch
    if (codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] != len(delays)
            or template_inputs.shape[:2] != codes.shape[:2] or min(delays) < 0
            or template_inputs.shape[-1] < prefix_frames+max(delays)+1):
        raise ValueError("Invalid native prefix or code history")
    clean = codes.clone()
    clean[:, 0][clean[:, 0] == 4] = 3
    tail = torch.empty_like(clean)
    for channel, delay in enumerate(delays):
        boundary = min(delay+1, clean.shape[-1])
        tail[:, channel, :boundary] = template_inputs[:, channel, prefix_frames:prefix_frames+boundary]
        tail[:, channel, boundary:] = clean[:, channel, :clean.shape[-1]-boundary]
    return torch.cat([template_inputs[..., :prefix_frames], tail], dim=-1), clean


def delayed_targets(template_inputs, prefix_frames, clean_codes, delays):
    """Current native targets, including delayed acoustic streams at the boundary."""
    import torch
    result = torch.empty_like(clean_codes)
    for channel, delay in enumerate(delays):
        boundary = min(delay, clean_codes.shape[-1])
        result[:, channel, :boundary] = template_inputs[:, channel, prefix_frames+1:prefix_frames+1+boundary]
        result[:, channel, boundary:] = clean_codes[:, channel, :clean_codes.shape[-1]-boundary]
    return result
