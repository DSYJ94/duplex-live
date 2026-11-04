"""Saved-activation offload, preserving full native temporal gradients.

Frozen and adapter parameter storage stays on GPU. Only large saved backward
tensors move losslessly to pinned CPU memory; forward arithmetic is unchanged.
"""
from contextlib import contextmanager
import torch


class ActivationOffload:
    def __init__(self,model,minimum_bytes=256*1024,maximum_host_bytes=48*1024**3,force_cpu_test=False):
        self.parameter_storage={p.untyped_storage().data_ptr() for p in model.parameters()}
        self.minimum_bytes=minimum_bytes;self.maximum_host_bytes=maximum_host_bytes;self.force_cpu_test=force_cpu_test
        self.saved_bytes=0;self.tensors=0

    def pack(self,tensor):
        size=tensor.numel()*tensor.element_size()
        if ((not tensor.is_cuda and not self.force_cpu_test) or size<self.minimum_bytes or
                tensor.untyped_storage().data_ptr() in self.parameter_storage):
            return ('direct',tensor.detach())
        if self.saved_bytes+size>self.maximum_host_bytes:
            raise MemoryError(f'Native activation CPU reservation exceeded{self.maximum_host_bytes/1024**3:g}GiB; no objective/gradient truncation')
        host=torch.empty_like(tensor,device='cpu',pin_memory=tensor.is_cuda)
        host.copy_(tensor.detach(),non_blocking=tensor.is_cuda)
        self.saved_bytes+=size;self.tensors+=1
        return ('offloaded',host,tensor.device)

    def unpack(self,packed):
        if packed[0]=='direct':return packed[1]
        return packed[1].to(packed[2],non_blocking=packed[2].type=='cuda')

    @contextmanager
    def context(self):
        with torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack):yield self
