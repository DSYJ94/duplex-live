"""Low-rank output correction with an immutable published projection.

No attention/runtime changes. Only Delta's two explicit tensors are trainable.
Checkpoint loading validates the exact base asset manifest and strict inventory.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA='dl-frozen-controller-head-adapter-v1'


class Delta(nn.Module):
    def __init__(self,hidden,vocabulary,rank=8,scale=1.):
        super().__init__()
        if not 1<=rank<=64 or not 0<scale<=8:raise ValueError('Invalid bounded output adapter')
        self.hidden=hidden;self.vocabulary=vocabulary;self.rank=rank;self.scale=float(scale)
        self.down=nn.Parameter(torch.empty(rank,hidden,dtype=torch.float32))
        self.up=nn.Parameter(torch.zeros(vocabulary,rank,dtype=torch.float32))
        nn.init.normal_(self.down,std=.02)
    def forward(self,hidden):return F.linear(F.linear(hidden.float(),self.down),self.up)*self.scale


class ResidualHead(nn.Module):
    def __init__(self,base,delta):
        super().__init__()
        if not isinstance(base,nn.Linear) or base.bias is not None or (base.in_features,base.out_features)!=(delta.hidden,delta.vocabulary):raise ValueError('Output projection shape/type differs')
        if any(p.requires_grad for p in base.parameters()):raise ValueError('Published output projection must be frozen')
        self.base=base;self.delta=delta
    @property
    def weight(self):return self.base.weight
    def forward(self,hidden):
        original=self.base(hidden)
        return original+self.delta(hidden).to(original.dtype)


def save_delta(folder,delta,assets_sha256,*,purpose,provenance):
    from safetensors.torch import save_file
    if purpose not in ('mechanism-test-only','trained-development-candidate'):raise ValueError('Explicit checkpoint purpose required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    state={k:v.detach().cpu().contiguous() for k,v in delta.state_dict().items()}
    save_file(state,str(folder/'adapter.safetensors'))
    metadata=dict(schema=SCHEMA,base_assets_sha256=assets_sha256,rank=delta.rank,scale=delta.scale,hidden=delta.hidden,vocabulary=delta.vocabulary,
        tensor_shapes={k:list(v.shape) for k,v in state.items()},weights_sha256=hashlib.sha256((folder/'adapter.safetensors').read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),purpose=purpose,provenance=provenance,production_adopted=False)
    (folder/'adapter.json').write_text(json.dumps(metadata,indent=2)+'\n');return metadata


def load_delta(folder,assets_sha256,device,*,allow_mechanism_fixture=False):
    from safetensors.torch import load_file
    folder=Path(folder);m=json.loads((folder/'adapter.json').read_text())
    if (m['schema']!=SCHEMA or m['base_assets_sha256']!=assets_sha256
            or m['implementation_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or m['weights_sha256']!=hashlib.sha256((folder/'adapter.safetensors').read_bytes()).hexdigest()):raise ValueError('Changed or incompatible output adapter')
    if m['purpose']!='trained-development-candidate' and not (allow_mechanism_fixture and m['purpose']=='mechanism-test-only'):raise ValueError('Mechanism fixture is not a model candidate')
    delta=Delta(m['hidden'],m['vocabulary'],m['rank'],m['scale']);state=load_file(str(folder/'adapter.safetensors'))
    if set(state)!=set(delta.state_dict()) or {k:list(v.shape) for k,v in state.items()}!=m['tensor_shapes']:raise ValueError('Output adapter inventory differs')
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in state.values()):raise ValueError('Invalid output adapter tensors')
    delta.load_state_dict(state,strict=True);return delta.to(device),m


def parameter_hashes(parameters):
    result={}
    for name,value in parameters.items():
        digest=hashlib.sha256();flat=value.detach().reshape(-1)
        for first in range(0,flat.numel(),8*1024*1024):digest.update(flat[first:first+8*1024*1024].contiguous().view(torch.uint8).cpu().numpy().tobytes())
        result[name]=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=digest.hexdigest())
    return result
