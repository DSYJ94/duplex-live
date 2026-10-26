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
