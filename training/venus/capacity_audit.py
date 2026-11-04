"""Configuration-derived capacity counts. Never allocates native weight storage."""
import hashlib
import json
from pathlib import Path


def audit(config_path):
    path = Path(config_path); config = json.loads(path.read_text())
    h, mid, layers = (config[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers'))
    q = config['num_attention_heads'] * config['head_dim']
    kv = config['num_key_value_heads'] * config['head_dim']
    if config['attention_bias'] or q != h:
        raise ValueError('This audit requires the pinned bias-free Qwen3 geometry')
    sizes = {'q_proj': (h,q), 'k_proj': (h,kv), 'v_proj': (h,kv), 'o_proj': (q,h),
             'gate_proj': (h,mid), 'up_proj': (h,mid), 'down_proj': (mid,h)}
    rank_unit = sum(a+b for a,b in sizes.values())
    # Two block RMSNorms plus Q/K head RMSNorms.
    block = sum(a*b for a,b in sizes.values()) + 2*h + 2*config['head_dim']
    return dict(config_path=str(path), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                hidden_size=h, intermediate_size=mid, layers=layers, projection_shapes=sizes,
                dense_parameters_per_block=block,
                variants=[dict(name='current_qv_last1_rank64', trainable_parameters=64*((h+q)+(h+kv)), new_foundation_parameters=0),
                          dict(name='all7_last16_rank64', trainable_parameters=16*64*rank_unit, new_foundation_parameters=0),
                          dict(name='all7_all36_rank128', trainable_parameters=layers*128*rank_unit, new_foundation_parameters=0),
                          dict(name='all7_all36_rank256', trainable_parameters=layers*256*rank_unit, new_foundation_parameters=0),
                          dict(name='all7_all36_rank512', trainable_parameters=layers*512*rank_unit, new_foundation_parameters=0),
                          dict(name='dense_existing_last6', trainable_parameters=6*block, new_foundation_parameters=0),
                          dict(name='six_new_identity_blocks', trainable_parameters=6*block, new_foundation_parameters=6*block)],
                excludes='Embedding/head/speech parameters and activations; optimizer memory not counted.',
                native_qualified=False, optimizer_updates=0)
