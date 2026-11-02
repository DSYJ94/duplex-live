from types import SimpleNamespace
import pytest
import torch
from torch import nn
from training.venus.lora import install,suffix_forward


def model(layers=3,device='cpu',q_in=8,q_out=8,v_out=4):
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.self_attn=nn.Module()
            self.self_attn.q_proj=nn.Linear(q_in,q_out,device=device)
            self.self_attn.v_proj=nn.Linear(q_in,v_out,device=device)
    llm=nn.Module();llm.model=nn.Module();llm.model.layers=nn.ModuleList([Layer() for _ in range(layers)]);llm.eval();return llm


def test_frozen_prefix_rejects_adapter_that_requires_earlier_layers():
    llm=model();install(llm,rank=2,alpha=2,last_layers=2)
    # Actual fast-forward entry point must reject before reading cached values.
    with pytest.raises(ValueError,match='trainable or adapted layers'):
        suffix_forward(llm,dict(prefix_layers=2),'cpu')


def test_meta_device_counts_actual_installer_without_base_weight_allocation():
    for layers,expected in ((1,851968),(4,3407872),(8,6815744)):
        llm=model(36,'meta',4096,4096,1024)
        report=install(llm,rank=64,alpha=64,last_layers=layers)
        assert report['parameters']==expected and len(report['modules'])==layers*2
        assert all(p.device.type=='meta' for p in llm.parameters())
