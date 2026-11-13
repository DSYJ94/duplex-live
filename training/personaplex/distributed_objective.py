"""Preserve original retained-objective mass across collective model execution.

Every rank deliberately receives the SAME retained exposure sequence. New data
may be partitioned separately only with an explicit global weighting contract.
"""
import hashlib
import json
import torch.distributed as dist


def agree(exposures):
    digest = hashlib.sha256(json.dumps(exposures, sort_keys=True, allow_nan=False).encode()).hexdigest()
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, digest)
    if len(set(gathered)) != 1:
        raise ValueError('Retained ranks differ; unequal native call collectives would deadlock')
    return digest


def backward_retained(runtime, groups, references, exposures):
    from .matched_losses import Objective
    if not exposures:
        raise ValueError('Complete declared retained exposure sequence required')
    digest = agree(exposures)
    objective = Objective(runtime.view, groups, references)
    records = []
    for exposure in exposures:
        # No implicit accumulation/world-size division: DDP/FSDP averages equal
        # replicas, preserving the existing per-member/objective normalization.
        # Recorded calls and actual target boundaries stay byte-identical.
        records.append(objective.backward(exposure))
    return dict(exposure_sha256=digest, records=records,
                replicated_retention=True, implicit_loss_divisor=1,
                optimizer_updates=0)
