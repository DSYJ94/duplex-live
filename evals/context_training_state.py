"""Portable adapter/optimizer/RNG state; published base weights remain separately pinned."""
import json
import random
from pathlib import Path


def _metadata(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def _cpu(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    return value


def _optimizer_names(learned, optimizer):
    flat = [p for group in optimizer.param_groups for p in group['params']]
    if len(flat) != len(learned) or any(a is not b for a,b in zip(flat,learned.values())):
        raise RuntimeError('Optimizer order must match explicit adapter parameter names')
    return list(learned)


def save(path, learned, optimizer, completed_steps, metadata, scheduler=None):
    import numpy as np
    import torch
    if type(completed_steps) is not int or completed_steps<1:raise RuntimeError('Invalid completed update count')
    names = _optimizer_names(learned,optimizer); state = np.random.get_state()
    snapshot = {'schema':'dl-context-training-state-v1','metadata':_metadata(metadata),
        'completed_steps':completed_steps,'parameter_names':names,
        'adapters':{n:_cpu(p) for n,p in learned.items()},'optimizer':_cpu(optimizer.state_dict()),
        'optimizer_class':type(optimizer).__module__+'.'+type(optimizer).__qualname__,
        'scheduler':None if scheduler is None else _cpu(scheduler.state_dict()),
        'torch_cpu_rng':torch.get_rng_state().cpu().clone(),
        'torch_cuda_rng':_cpu(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else [],
        'python_rng':random.getstate(),'numpy_rng':{'name':state[0],'keys':state[1].tolist(),
            'position':int(state[2]),'has_gauss':int(state[3]),'cached_gauss':float(state[4])},
        'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,
        'cross_hardware_bitwise_identity_guaranteed':False}
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    torch.save(snapshot,temporary); temporary.replace(path)
    return snapshot


def restore(path, learned, optimizer, metadata, scheduler=None):
    import numpy as np
    import torch
    state = torch.load(path,map_location='cpu',weights_only=True)
    names = _optimizer_names(learned,optimizer)
    if (state['schema'] != 'dl-context-training-state-v1' or state['metadata'] != _metadata(metadata)
            or state['parameter_names'] != names or set(state['adapters']) != set(learned)
            or state['optimizer_class'] != type(optimizer).__module__+'.'+type(optimizer).__qualname__
            or type(state['completed_steps']) is not int or state['completed_steps']<1
            or (state['scheduler'] is None) != (scheduler is None)
            or len(state['torch_cuda_rng']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0)):
        raise RuntimeError('Training state identity, parameter order or device count changed')
    for name,p in learned.items():
        value = state['adapters'][name]
        if value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
            raise RuntimeError('Invalid saved adapter tensor')
    for entry in state['optimizer']['state'].values():
        for value in entry.values():
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite saved optimizer state')
    with torch.no_grad():
        for name,p in learned.items():p.copy_(state['adapters'][name].to(p.device))
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:scheduler.load_state_dict(state['scheduler'])
    # Restore after construction/loading, which may itself consume random state.
    random.setstate(state['python_rng']); n = state['numpy_rng']
    np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gauss']))
    torch.set_rng_state(state['torch_cpu_rng'])
    if state['torch_cuda_rng']:torch.cuda.set_rng_state_all(state['torch_cuda_rng'])
    return state
