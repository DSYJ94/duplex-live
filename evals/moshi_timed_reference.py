"""Causal reference rows for whole-sequence MoshiRAG training.

The encoder stays frozen. Each row reaches exactly one temporal transformer
step, as in the inference queue. No new model parameters or checkpoint keys.
"""
from types import MethodType


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
    """Use the native training forward with a per-step reference addition.

This training-only wrapper is not safe for concurrent use of the same model.
The native embedding sum handles dtype conversion before transformer execution.
The addition happens before checkpointed transformer layers, so recomputation
uses the same conditioned input even after this method is restored.
"""
    import torch
    if reference.shape != (codes.shape[0], codes.shape[-1], model.dim):
        raise ValueError("Reference must match [batch, model steps, model dimension]")
    if reference.requires_grad or not torch.isfinite(reference).all():
        raise ValueError("Reference encoder must remain frozen and finite")
    if getattr(model, "_timed_reference_active", False):
        raise RuntimeError("Nested/concurrent timed-reference forward")
    original = model.forward_text
    had_instance_method = "forward_text" in model.__dict__
    previous_instance_method = model.__dict__.get("forward_text")

    def conditioned(self, sequence=None, sequence_emb=None, sum_condition=None,
                    cross_attention_src=None, streaming_sum_condition=None):
        if sequence is None or sequence_emb is not None or sequence.shape[-1] != reference.shape[1]:
            raise ValueError("Unexpected training sequence/prepend layout")
        if streaming_sum_condition is not None and streaming_sum_condition.numel():
            raise ValueError("Do not combine whole-sequence and streaming references")
        combined = reference.to(device=sequence.device, dtype=self.dtype)
        if sum_condition is not None:
            combined = combined + sum_condition.to(combined)
        return original(sequence=sequence, sum_condition=combined,
                        cross_attention_src=cross_attention_src, streaming_sum_condition=None)

    model._timed_reference_active = True
    model.forward_text = MethodType(conditioned, model)
    try:
        return model(codes=codes, condition_tensors=conditions)
    finally:
        if had_instance_method:
            model.forward_text = previous_instance_method
        else:
            del model.forward_text
        del model._timed_reference_active
