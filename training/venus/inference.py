"""Load our verified training checkpoint for explicit, non-adopted native tests."""
import hashlib
from pathlib import Path
import torch
from .lora import install, load_adapter

MODEL_REVISION = '4681004131ea1f1fb9aca9f0db76be322657cc79'


def load_candidate(llm, checkpoint):
    path = Path(checkpoint)
    if path.name.startswith('expected-'):
        raise ValueError('The restore reference branch is proof-only, never a candidate')
    # Only locally verified project checkpoints; torch states are not an upload API.
    state = torch.load(path, map_location='cpu', weights_only=False)
    metadata = state['metadata']; recipe = metadata['recipe']; step = state['step']
    if metadata['model_revision'] != MODEL_REVISION:
        raise ValueError('Candidate belongs to a different published model revision')
    if (recipe['forward_path'] != 'native-frozen-prefix-v1'
            or recipe['last_layers'] != 1
            or recipe['projections'] != ['q_proj', 'v_proj']
            or not 0 < step <= recipe['updates']):
        raise ValueError('Checkpoint is not a trained native suffix candidate')
    spec = install(llm, recipe['rank'], recipe['alpha'], recipe['last_layers'])
    load_adapter(llm, state['adapter'])  # Exact parameter names and shapes.
    llm.requires_grad_(False)
    return dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        optimizer_updates=step, planned_updates=recipe['updates'], adapter=spec,
        training_source=metadata['source'], model_revision=metadata['model_revision'],
        experimental_candidate=True, adopted=False, native_behavior_qualified=False)
