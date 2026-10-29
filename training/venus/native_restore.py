"""Independent native CUDA process: exact saved-state restore, ZERO updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .action_objective import AUTH
from .checkpoint_restore import restore
from .inference import MODEL_REVISION
from .lora import install, scene_loss
from .native_cache import load_native
from .state import exact_state, rng_state
from .train import read_example, assert_deadline


def main(a):
    from portable.venus.native_app import ensure_guard
    ensure_guard()
    assert_deadline(a.lease, 300)
    authority = json.loads(AUTH.read_text())
    if authority['status'] != 'APPROVED' or authority['maximum_candidate_new_updates'] != 108:
        raise ValueError('Exact approved action-fidelity phase required')
    path = Path(a.checkpoint)
    if hashlib.sha256(path.read_bytes()).hexdigest() != a.sha256:
        raise ValueError('Exact saved checkpoint hash required')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if a.kind == 'parent':
        if a.sha256 != authority['parent_checkpoint_sha256']:
            raise ValueError('Original298 parent required')
    elif (state['metadata']['parent_training_steps'] != 298 or state['step'] not in (54,108)
          or state['metadata']['parent_checkpoint_sha256'] != authority['parent_checkpoint_sha256']):
        raise ValueError('Complete admitted action-phase final state required')
    if state['metadata']['model_revision'] != MODEL_REVISION:
        raise ValueError('Exact native foundation revision required')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(a.cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    if hashlib.sha256((cache/'manifest.json').read_bytes()).hexdigest() != state['metadata']['cache_sha256']:
        raise ValueError('Exact original native cache required')
    row = next(r for r in sorted(manifest['scenes'], key=lambda r:r['id'])
               if r['fold'] == 'TRAIN' and r['family'] == 'ordinary')
    model, _ = load_native(a.model, language_only=True)
    recipe = state['metadata']['recipe']
    spec = install(model.llm, recipe['rank'], recipe['alpha'], recipe['last_layers'])
    if spec['parameters'] != 851968 or len(model.llm.model.layers) != 36:
        raise ValueError('Exact native suffix adapter required')
    base = torch.optim.AdamW([p for p in model.llm.parameters() if p.requires_grad],
                            lr=recipe['learning_rate'], betas=tuple(recipe['adam_betas']),
                            eps=recipe['adam_eps'], weight_decay=recipe['weight_decay'])
    runtime, proof = restore(path, a.sha256, model.llm, base,
                             source=a.opal_source, run_dir=out/'opal')
    try:
        before = rng_state()
        with torch.no_grad():
            loss, parts = scene_loss(model.llm, read_example(cache,row), 'cuda')
        if not exact_state(before, rng_state()):
            raise ValueError('Zero-update native comparison changed RNG')
        proof.update(native_process_pid=os.getpid(), model_revision=MODEL_REVISION,
                     original_loss=float(loss), original_loss_parts=parts,
                     fixed_train_scene=row['id'], kind=a.kind,
                     actual_cuda_process=True, optimizer_updates_performed=0,
                     source=a.source, native_behavior_qualified=False)
        (out/'restore.json').write_text(json.dumps(proof, indent=2)+'\n')
        print(json.dumps(proof), flush=True)
    finally:
        runtime.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=('parent','final'), required=True)
    for key in ('checkpoint','sha256','cache','model','output','opal-source','lease','source'):
        p.add_argument('--'+key, required=True)
    main(p.parse_args())
