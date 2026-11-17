"""Exact device-independent fingerprints for complete rank-owned Adam states."""
import hashlib
import json
import math
import torch


def tree(value):
    if torch.is_tensor(value):
        if value.layout!=torch.strided:raise ValueError('Declared dense Adam tensors required')
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError('Finite Adam state required')
        data=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest=hashlib.sha256()
        view=memoryview(data)
        for start in range(0,len(view),4*1024*1024):digest.update(view[start:start+4*1024*1024])
        return ['tensor',str(value.dtype),list(value.shape),digest.hexdigest()]
    if isinstance(value,dict):
        if any(type(key) not in (int,str) for key in value):raise ValueError('Declared optimizer field keys required')
        return ['dict',[[tree(key),tree(value[key])]for key in sorted(value,key=lambda key:(type(key).__name__,repr(key)))]]
    if isinstance(value,(list,tuple)):
        return [type(value).__name__,[tree(item)for item in value]]
    if value is None or type(value) in (bool,int,str):return [type(value).__name__,value]
    if type(value) is float and math.isfinite(value):return ['float',value.hex()]
    raise ValueError('Unsupported/nonfinite optimizer state field')


def fingerprint(state):
    return hashlib.sha256(json.dumps(tree(state),separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_exact(optimizer,state):
    # Capture before loading: CPU loading can alias supplied tensors, so a
    # comparison against the same mutable tree afterwards could be tautological.
    expected=fingerprint(state)
    optimizer.load_state_dict(state)
    actual=fingerprint(optimizer.state_dict())
    if actual!=expected:
        raise ValueError('Every local Adam field/dtype/shape/value must restore exactly')
    return dict(local_Adam_exact=True,expected_sha256=expected,actual_sha256=actual,
                optimizer_updates_performed=0)
