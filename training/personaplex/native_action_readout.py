"""Reversible learned RET-logit residual on the actual native text-head input.

This is supervised action capacity, NOT a forced tool call or a claimed WSP
replication. The existing native model/adapter stays frozen. Only native token4
can change; non-RET relative logits are invariant. Native efficacy is untested.
"""
import torch
import re
from torch import nn


class RetReadout(nn.Module):
    def __init__(self,original,*,base_identity):
        super().__init__()
        if not isinstance(base_identity,str)or re.fullmatch('[0-9a-f]{64}',base_identity)is None:raise ValueError('Pinned native backbone/adapter identity digest required')
        self.base_identity=base_identity
        if not hasattr(original,'in_features')or original.out_features<=4:raise ValueError('Actual native text projection required')
        self.original=original
        for p in self.original.parameters():p.requires_grad_(False)
        self.delta=nn.Linear(original.in_features,1,bias=True,device=original.weight.device,dtype=torch.float32)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)

    @property
    def in_features(self):return self.original.in_features

    @property
    def out_features(self):return self.original.out_features

    @property
    def weight(self):return self.original.weight

    @property
    def bias(self):return self.original.bias

    def forward(self,x):
        base=self.original(x);change=self.delta(x.to(torch.float32)).squeeze(-1)
        result=base.clone();result[...,4]=base[...,4]+change.to(base.dtype)
        return result

    def learned_state(self):
        return dict(schema='dl-native-ret-readout-v1',token=4,input_features=self.in_features,base_identity=self.base_identity,
                    state={k:v.detach().cpu().clone()for k,v in self.delta.state_dict().items()})

    def restore(self,state):
        if state.get('schema')!='dl-native-ret-readout-v1'or state.get('token')!=4 or state.get('input_features')!=self.in_features or state.get('base_identity')!=self.base_identity:raise ValueError('Exact native readout identity required')
        expected=self.delta.state_dict()
        if set(state['state'])!=set(expected)or any(v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not torch.isfinite(v).all()for k,v in state['state'].items()):raise ValueError('Complete finite readout state required')
        self.delta.load_state_dict(state['state'],strict=True)


def install(lm,*,base_identity):
    if isinstance(lm.text_linear,RetReadout):raise ValueError('No duplicate native action readout')
    readout=RetReadout(lm.text_linear,base_identity=base_identity)
    for p in lm.parameters():p.requires_grad_(False)
    lm.text_linear=readout
    return lm.text_linear
