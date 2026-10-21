"""Independent destination restore of the pinned native7590052 adapter/Adam/RNG.

This does not claim native28/acoustic49/stream49 numerical or behavioral approval.
Expose exactly one GPU before launching, matching the saved RNG device inventory.
"""
import argparse
from collections import OrderedDict
import datetime
import hashlib
import json
from pathlib import Path
import random
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--proof', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from evals.context_training_state import restore
    from training.control.train_context import assert_equal, cpu_tree
    assert torch.cuda.device_count() == 1
    for name, wanted in [
        ('adapters.safetensors', '7bbb2e9f1822cb43db50f6ead7c14d510e7c280d520cc7e1436dced6edbcfeb5'),
        ('training-state.pt', '6a9fc6cb0949d1562f654af74cb6e86a827a0214a486f869e7bd92b1ae0b63b4'),
        ('report.json', '397a259d124948dec3aeed63564ad24dbf2f13bb2b3699565f9f1974e60fa3eb')]:
        assert hashlib.sha256((args.proof/name).read_bytes()).hexdigest() == wanted
    report = json.loads((args.proof/'report.json').read_text())
    checkpoint = load_file(args.proof/'adapters.safetensors')
    # Installation order and nn.Module traversal order differ: text_linear is
    # registered before the temporal stack. Reconstruct traversal from the pinned
    # published state_dict, independently of the saved optimizer name list.
    published = args.proof.parents[1] / ('models/hf-cache/hub/'
        'models--mindlogicinc--context-spanning-7b/snapshots/'
        'e62a4494a295ff8cee345451cf40aae85719a9f5/context_spanning_7b.pt')
    model = torch.load(published, map_location='meta', weights_only=True, mmap=True)['model']
    modules = [n[:-7] for n in model if n.endswith('.weight') and
               n[:-7] in report['adapter_modules']]
    assert set(modules) == set(report['adapter_modules'])
    names = [module + '.lora_' + suffix + '.weight'
             for module in modules for suffix in ('A', 'B')]
    assert set(names) == set(checkpoint) and len(names) == 26
    learned = OrderedDict((n, torch.nn.Parameter(torch.zeros_like(checkpoint[n], device='cuda')))
                          for n in names)
    optimizer = torch.optim.AdamW(list(learned.values()), lr=2e-7, weight_decay=0.)
    keys = ['source_commit', 'repo_commit', 'assets_manifest_sha256', 'data_sha256',
            'native_protocol', 'voice_sha256', 'frozen_hashes', 'retention_examples',
            'warm_start', 'warm_start_optimizer_reset']
    metadata = {key: report[key] for key in keys}
    metadata['recipe'] = report['combined_recipe']
    saved = restore(args.proof/'training-state.pt', learned, optimizer, metadata)
    assert saved['completed_steps'] == 8
    for name in names:
        assert_equal(learned[name].detach().cpu(), checkpoint[name])
    assert_equal(saved['optimizer'], cpu_tree(optimizer.state_dict()))
    assert torch.equal(saved['torch_cpu_rng'], torch.get_rng_state())
    assert_equal(saved['torch_cuda_rng'], [x.cpu() for x in torch.cuda.get_rng_state_all()])
    assert random.getstate() == saved['python_rng']
    rng = np.random.get_state()
    assert (rng[0] == saved['numpy_rng']['name'] and
            rng[1].tolist() == saved['numpy_rng']['keys'] and
            rng[2] == saved['numpy_rng']['position'] and
            rng[3] == saved['numpy_rng']['has_gauss'] and
            rng[4] == saved['numpy_rng']['cached_gauss'])
    clocks = sorted({int(x['step']) for x in optimizer.state.values()})
    assert clocks == [528]
    result = {'schema': 'dl-runpod-independent-adapter-state-restore-v1',
              'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'passed': True, 'optimizer_updates': 0, 'adapter_tensors': 26,
              'trainable_parameters': sum(p.numel() for p in learned.values()),
              'phase_updates_restored': 8, 'adam_clocks': clocks,
              'optimizer_and_python_numpy_torch_cpu_cuda_rng_exact': True,
              'metadata_reconstructed_from_pinned_report': True,
              'parameter_order_from_published_model_state_dict': True,
              'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
              'cuda': torch.version.cuda, 'native_full_numerical_or_behavior_approved': False}
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
