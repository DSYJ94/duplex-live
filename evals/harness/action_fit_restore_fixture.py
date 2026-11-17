"""Real pinned OPAL CPU fixture; synthetic features, never native training.

Invoke fit and restore in separate processes. This fixture does not load a
native model, establish CUDA compatibility or qualify delegation behavior.
"""
import argparse
import json
import os
import uuid
from pathlib import Path
import torch
from training.personaplex.action_features import selected
from training.personaplex.action_fit import fit, model, new_runtime, evaluate
from training.personaplex.action_inference import identity
from training.personaplex.action_restore import restore_fit
from training.venus.state import exact_state


def run(a):
    bound = identity(dict(step=196, checkpoint_sha256='a'*64, trainer_source='b'*40))
    if a.stage == 'fit':
        a.root.mkdir(exist_ok=False)
        torch.manual_seed(42)
        x = torch.randn(1, 6, 5, dtype=torch.bfloat16)
        z = torch.randn(1, 1, 6, 12, dtype=torch.bfloat16)
        row = dict(group='actual', id='synthetic-CPU-only', fold='TRAIN',
                   **selected(x, z, torch.ones(6, dtype=torch.bool), torch.tensor([4, 0, 4, 1, 2, 3])))
        cache = dict(schema='dl-native-ret-TRAIN-features-v1', optimizer_updates=0,
                     native_backbone_updated=False, population=dict(positive=2, negative=4), rows=[row])
        torch.save(cache, a.root/'synthetic-cache.pt')
        for mode in ('readout', 'bias'):
            fit(cache, a.root/mode, mode=mode, device='cpu', opal_source=a.opal_source,
                identity=bound, provenance=dict(fixture_only=True, execution=uuid.uuid4().hex))
        return dict(stage='fit', passed=True, synthetic_updates_per_arm=128,
                    native_training_updates=0, writer_pid=os.getpid())
    cache = torch.load(a.root/'synthetic-cache.pt', weights_only=False)
    proofs = {}
    for mode in ('readout', 'bias'):
        state = torch.load(a.root/mode/'step-0128.pt', weights_only=False)
        if state['writer_pid'] == os.getpid() or not state['metadata']['provenance']['fixture_only']:
            raise ValueError('A different fixture process is required')
        delta = model(5, mode, 'cpu')
        runtime, recipe = new_runtime(delta, mode, opal_source=a.opal_source,
                                     out=a.root/(mode+'-fresh-restore-opal'))
        try:
            if recipe != state['metadata']['recipe']:
                raise ValueError('Saved paired recipe changed')
            proofs[mode] = restore_fit(state, delta, runtime, bound)
            if not exact_state(evaluate(delta, cache['rows']), state['metrics']):
                raise ValueError('Full synthetic fit metrics differ after fresh restore')
        finally:
            runtime.close()
    return dict(stage='restore', passed=True, actual_pinned_OPAL=True, arms=proofs,
                native_model_loaded=False, CUDA_tested=False, new_updates=0,
                reader_pid=os.getpid())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('fit', 'restore'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--opal-source', type=Path, required=True)
    a = p.parse_args()
    result = run(a)
    (a.root/(a.stage+'-proof.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
