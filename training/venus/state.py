"""Exact optimizer/RNG checkpoint identity, with explicit CPU/CUDA state."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from .lora import adapter_state, load_adapter


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def exact_state(left, right):
    """Compare serialized Adam/RNG trees, including CPU versus CUDA tensors."""
    if torch.is_tensor(left):return torch.is_tensor(right) and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(exact_state(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(exact_state(x,y) for x,y in zip(left,right))
    return left==right


def restore_parent(state, model, optimizer):
    """Load actual saved adapter, Adam moments and all RNG streams without updates."""
    load_adapter(model,state['adapter']);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rng'])
    assert exact_state(state['adapter'],adapter_state(model))
    assert exact_state(state['optimizer'],optimizer.state_dict())
    assert exact_state(state['rng'],rng_state())
    return dict(adapter_exact=True,adam_exact=True,rng_exact=True,new_optimizer_updates=0)


def save(path, model, optimizer, step, metadata):
    path = Path(path)
    state = dict(adapter=adapter_state(model), optimizer=optimizer.state_dict(), rng=rng_state(),
                 step=step, metadata=metadata)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temp); os.replace(temp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, model, optimizer, expected):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['metadata'] != expected:
        raise ValueError('Checkpoint source/data/recipe identity mismatch')
    load_adapter(model, state['adapter']); optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'])
    return state['step']
