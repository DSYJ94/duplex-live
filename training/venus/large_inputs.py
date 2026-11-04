"""Read the immutable complete corpus for all-layer native differentiation.

The suffix cache is only a storage container here. Original full embeddings,
segments, targets and action masks are retained; frozen-prefix values are never
passed to the larger model. This module grants no optimizer or compute scope.
"""
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '1bf09867b5d1ce78ed4c3504057773d20e0025bb0662e7442311fd79b08a4807'
APPROVAL_SHA = 'fe6b9c061ef7083658e7c6254598d916f96689578c9b44ef0d6e981be8c1add1'
PARENT_SHA = 'e2c16cb825dc654519e922379a220b8c962e9d8a78eec652a3b195922a69720f'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def full_native(example):
    """Select saved original native inputs, never reconstruct teacher targets."""
    import torch
    required = {'embeddings', 'segments', 'positions', 'targets', 'decision_mask',
                'id', 'fold', 'family'}
    if not required <= example.keys() or example['fold'] not in ('TRAIN', 'DEV'):
        raise ValueError('Original complete approved TRAIN/DEV native example required')
    x = example['embeddings']
    if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
        raise ValueError('Finite original native embeddings required')
    cursor = 0
    ends = set()
    for segment in example['segments']:
        if (type(segment['start']) is not int or segment['start'] != cursor
                or type(segment['length']) is not int or segment['length'] < 1):
            raise ValueError('Contiguous original native feed segmentation required')
        cursor += segment['length']
        ends.add(cursor - 1)
    positions = example['positions'].tolist()
    if (cursor != len(x) or not positions or len(set(positions)) != len(positions)
            or positions != sorted(positions) or not set(positions) <= ends
            or min(positions) < 0 or max(positions) >= len(x) - 1
            or len(example['targets']) != len(positions)
            or example['decision_mask'].shape != example['targets'].shape):
        raise ValueError('Exact native next-token positions and masks required')
    result = {k: v for k, v in example.items() if k not in ('prefix_inputs', 'prefix_layers')}
    result['forward_path'] = 'native-segmented-bptt-v1'
    return result


class Corpus:
    def __init__(self, root, *, action_index, admission):
        from .context_objective import ContextObjective
        self.root = Path(root)
        if sha(self.root/'manifest.json') != MANIFEST_SHA or sha(self.root/'approval.json') != APPROVAL_SHA:
            raise ValueError('Exact complete preserved304 cache and approval required')
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        approval = json.loads((self.root/'approval.json').read_text())
        self.rows = self.manifest['scenes']
        if (not approval['approved'] or approval['train'] != 240 or approval['dev'] != 64
                or approval['manifest_sha256'] != self.manifest['source_manifest_sha256']
                or len({r['id'] for r in self.rows}) != 304
                or {f: sum(r['fold'] == f for r in self.rows) for f in ('TRAIN', 'DEV')}
                != {'TRAIN': 240, 'DEV': 64}):
            raise ValueError('Whole original corpus, folds and approval required')
        self.objective = ContextObjective(action_index, self.root/'manifest.json', admission)

    def read(self, row, *, full=True):
        import torch
        path = self.root/row['file']
        if Path(row['file']).is_absolute() or '..' in Path(row['file']).parts or sha(path) != row['sha256']:
            raise ValueError('Exact approved native artifact hash required')
        example = torch.load(path, map_location='cpu', weights_only=False)
        self.objective.validate(example)
        return full_native(example) if full else example

    def loss(self, model, example, device):
        from .lora import scene_loss
        self.objective.validate(example)
        return scene_loss(model, example, device, kl_weight=.1, hidden_weight=.01,
                          action_token_ids=self.objective.ids, action_aux_weight=.25)
