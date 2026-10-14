"""Causal reference rows for whole-sequence MoshiRAG training.

The encoder stays frozen. Each row reaches exactly one temporal transformer
step, as in the inference queue. No new model parameters or checkpoint keys.
"""

def reference_schedule(frame_times, arrivals, dim):
    """Map completed encoder results onto model-step availability times.

arrivals contains ready_at_s, rows [R,D], and optionally valid_until_s.
Overlapping deliveries are rejected until training data specifies replacement.
"""
    import torch
    if frame_times.ndim != 1 or not len(frame_times) or not torch.isfinite(frame_times).all():
        raise ValueError("Invalid frame availability times")
    if (frame_times[1:] <= frame_times[:-1]).any():
        raise ValueError("Frame times must increase strictly")
    schedule = torch.zeros(len(frame_times), dim, dtype=torch.float32, device=frame_times.device)
    occupied = torch.zeros(len(frame_times), dtype=torch.bool, device=frame_times.device)
    for item in arrivals:
        rows = item["rows"]
        ready = item["ready_at_s"]
        if not torch.isfinite(torch.tensor(ready)) or ready < 0:
            raise ValueError("Invalid reference completion time")
        if rows.ndim != 2 or rows.shape[1] != dim or not len(rows) or rows.requires_grad or not torch.isfinite(rows).all():
            raise ValueError("Reference rows must be finite frozen [R,D] values")
        start = int(torch.searchsorted(frame_times, frame_times.new_tensor(ready)).item())
        end = min(start + len(rows), len(frame_times))
        valid_until = item.get("valid_until_s")
        if valid_until is not None:
            if not torch.isfinite(torch.tensor(valid_until)) or valid_until < ready:
                raise ValueError("Invalid cancellation/replacement time")
            end = min(end, int(torch.searchsorted(frame_times, frame_times.new_tensor(valid_until)).item()))
        if end <= start:
            continue
        if occupied[start:end].any():
            raise ValueError("Overlapping references require an explicit replacement policy")
        schedule[start:end] = rows[:end-start].detach().to(schedule)
        occupied[start:end] = True
    return schedule


def forward_with_reference(model, codes, reference, conditions=None):
    """Preserve native token delays, prefix positions and prediction masks.

Serving consumes the speaker prefix as embeddings before token frames. Build
that same causal sequence for parallel training, then remove prefix outputs
before the native depth decoder and target alignment. Nothing mutates the model.
"""
    import torch
    from moshi.models.lm import LMOutput, _delay_sequence, _undelay_sequence
    if codes.ndim != 3 or codes.shape[1] != model.num_codebooks:
        raise ValueError("Codes must match the native codebook layout")
    if reference.shape != (codes.shape[0], codes.shape[-1], model.dim):
        raise ValueError("Reference must match [batch, model steps, model dimension]")
    if reference.requires_grad or not torch.isfinite(reference).all():
        raise ValueError("Reference encoder must remain frozen and finite")
    initial = model._get_initial_token().expand(codes.shape[0], -1, -1)
    delayed = torch.cat([initial, _delay_sequence(model.delays, codes, initial)], dim=2)
    sequence = delayed[..., :-1]
    embedded = None
    for index in range(model.num_audio_codebooks):
        audio = model.emb[index](sequence[:, index + model.audio_offset])
        embedded = audio if embedded is None else embedded + audio
    text = model.text_emb(sequence[:, 0])
    embedded = text if embedded is None else embedded + text
    prefix, cross = None, None
    if conditions is not None:
        if model.fuser is None:
            raise ValueError("Conditions require the native fuser")
        existing_stream = model.fuser.get_streaming_sum(conditions)
        if existing_stream is not None and existing_stream.numel():
            raise ValueError("Do not combine whole-sequence and streaming references")
        static = model.fuser.get_sum(conditions)
        if static is not None:
            embedded = embedded + static.to(embedded)
        prefix = model.fuser.get_prepend(conditions)
        cross = model.fuser.get_cross(conditions)
    # Match serving arithmetic: sum static conditions, then the current row.
    embedded = embedded + reference.to(embedded)
    prefix_length = 0
    if prefix is not None and prefix.shape[1]:
        if prefix.ndim != 3 or prefix.shape[0] != codes.shape[0] or prefix.shape[2] != model.dim:
            raise ValueError("Invalid native speaker prefix")
        prefix_length = prefix.shape[1]
        embedded = torch.cat([prefix.to(embedded), embedded], dim=1)
    hidden, text_logits = model.forward_text(sequence_emb=embedded, cross_attention_src=cross)
    hidden = hidden[:, prefix_length:]
    text_logits = text_logits[:, :, prefix_length:]
    if hidden.shape[1] != codes.shape[-1]:
        raise RuntimeError("Prefix changed audio target alignment")
    audio_logits = model.forward_depformer_training(delayed[..., 1:], hidden)
    audio_logits, audio_mask = _undelay_sequence(
        model.delays[model.audio_offset:model.audio_offset + model.dep_q], audio_logits, fill_value=float("NaN"))
    audio_mask &= codes[:, model.audio_offset:model.audio_offset + model.dep_q] != model.zero_token_id
    text_logits, text_mask = _undelay_sequence(model.delays[:1], text_logits, fill_value=float("NaN"))
    text_mask &= codes[:, :1] != model.zero_token_id
    return LMOutput(audio_logits, audio_mask, text_logits, text_mask)
