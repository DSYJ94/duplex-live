"""Collective checkpoint envelope; model/Adam plus rank-specific RNG/program state."""
import os
from pathlib import Path
import torch
import torch.distributed as dist


def save(path, runtime, optimizer, *, identity, step, rank_state):
    if type(step) is not int or step < 0 or not identity or 'rng' not in rank_state or 'opal' not in rank_state:
        raise ValueError('Exact lineage, step, RNG and OPAL state required')
    rank, world = dist.get_rank(), dist.get_world_size()
    identities = [None] * world
    dist.all_gather_object(identities, dict(identity=identity, step=step, world=world))
    if any(item != identities[0] for item in identities):
        raise ValueError('Ranks disagree on checkpoint lineage or update')
    state = runtime.full_state(optimizer)
    # OPAL carries Adam registries: never gather those large objects to every
    # GPU/rank. Publish each rank locally and gather only small identities.
    import hashlib
    path = Path(path)
    rank_path = path.with_name(path.name + '.rank-%04d.pt' % rank)
    descriptor = None
    try:
        with rank_path.open('xb') as stream:
            torch.save(rank_state, stream)
            stream.flush(); os.fsync(stream.fileno())
        digest = hashlib.sha256()
        with rank_path.open('rb') as stream:
            for block in iter(lambda: stream.read(4*1024*1024), b''): digest.update(block)
        descriptor = dict(file=rank_path.name, sha256=digest.hexdigest(), error=None)
    except Exception as exc:
        descriptor = dict(error=repr(exc))
    ranks = [None] * world
    dist.all_gather_object(ranks, descriptor)
    if any(item['error'] for item in ranks):
        raise RuntimeError('Rank checkpoint publication failed: ' + repr(ranks))
    error = None
    if rank == 0:
        try:
            path = Path(path)
            if path.exists(): raise ValueError('Immutable checkpoint already exists')
            temporary = path.with_suffix(path.suffix + '.partial')
            with temporary.open('xb') as stream:
                torch.save(dict(schema='dl-distributed-native-state-v2', identity=identity,
                    step=step, world_size=world, training=state, rank_states=ranks), stream)
                stream.flush(); os.fsync(stream.fileno())
            os.rename(temporary, path)
        except Exception as exc:
            error = repr(exc)
    errors = [error]
    dist.broadcast_object_list(errors, src=0)
    if errors[0]: raise RuntimeError('Checkpoint publication failed: ' + errors[0])


def restore(state, runtime, optimizer, *, identity, restore_rank_state, checkpoint_directory):
    world, rank = dist.get_world_size(), dist.get_rank()
    if (state.get('schema') != 'dl-distributed-native-state-v2' or state.get('identity') != identity
            or state.get('world_size') != world or len(state.get('rank_states', [])) != world
            or type(state.get('step')) is not int or state['step'] < 0):
        raise ValueError('Exact lineage/world-size/state restore required; no implicit reshard or replay')
    runtime.load_full_state(state['training'], optimizer)
    import hashlib
    descriptor = state['rank_states'][rank]
    name = descriptor['file']
    if Path(name).name != name or descriptor.get('error') is not None:
        raise ValueError('Exact local rank-state identity required')
    path = Path(checkpoint_directory) / name
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): digest.update(block)
    if digest.hexdigest() != descriptor['sha256']:
        raise ValueError('Rank state hash mismatch')
    restore_rank_state(torch.load(path, map_location='cpu', weights_only=False))
    return state['step']
