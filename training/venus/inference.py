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
    parent_steps=metadata.get('parent_training_steps',0)
    parents={190:'10354f93ec2b5e224ae7298b2f3b8687f9dc74d2fbf87c1db62bd79763155b63',244:'c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270',298:'e7340edd15537523db78120a97902399faa0af9caa46e875ec92e56889d6edef'}
    if parent_steps and (parent_steps not in parents or metadata.get('parent_checkpoint_sha256')!=parents[parent_steps]):
        raise ValueError('Unknown parent lineage')
    return dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        optimizer_updates=parent_steps+step, current_round_updates=step, parent_training_steps=parent_steps,planned_updates=parent_steps+recipe['updates'], adapter=spec,
        training_source=metadata['source'], model_revision=metadata['model_revision'],
        experimental_candidate=True, adopted=False, native_behavior_qualified=False)
