"""Saved-activation offload, preserving full native temporal gradients.

Frozen and adapter parameter storage stays on GPU. Only large saved backward
tensors move losslessly to CPU memory; forward arithmetic is unchanged.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import torch


def available_host_bytes():
    path=Path('/proc/meminfo')
    if not path.exists():return None
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('Host memory availability cannot be verified')


class ActivationOffload:
    def __init__(self,model,minimum_bytes=256*1024,maximum_host_bytes=48*1024**3,force_cpu_test=False,
                 policy='legacy-pinned',available_host_fn=available_host_bytes):
        if policy not in ('legacy-pinned','repeat4-pageable-v1','repeat4-prefix-pageable-v2'):raise ValueError('Explicit lossless storage policy required')
        self.parameter_storage={p.untyped_storage().data_ptr() for p in model.parameters()}
        self.minimum_bytes=minimum_bytes;self.maximum_host_bytes=maximum_host_bytes;self.force_cpu_test=force_cpu_test
        self.saved_bytes=0;self.tensors=0;self.logical_bytes=0;self.compressed_tensors=0
        self.policy=policy;self.available_host_fn=available_host_fn
        self.prefix_histories={};self.prefix_reuses=0

    def reserve(self,stored):
        if self.saved_bytes+stored>self.maximum_host_bytes:
            raise MemoryError(f'Native activation CPU reservation exceeded{self.maximum_host_bytes/1024**3:g}GiB; no objective/gradient truncation')

    def host_headroom(self,stored,is_cuda):
        free=self.available_host_fn()
        if free is None and is_cuda:raise RuntimeError('Linux host memory guard required')
        if free is not None and free<stored+16*1024**3:
            raise MemoryError('Preserve16GiB actual host headroom before saved-tensor allocation')

    @staticmethod
    def byte_equal(a,b):
        return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

    def prefix_pack(self,host,device,repeat):
        """Keep immutable verified KV prefixes in a growable owned CPU buffer.

        Saved handles refer to this record and their OWN original length, never
        to a subsequently changed cache length. Existing prefix bytes are never
        overwritten. Hashes only choose candidates; full byte equality admits
        sharing, including signed zeros and NaN payloads.
        """
        b,h,n,d=host.shape
        first=host[:,:,:1].contiguous().view(torch.uint8).numpy().tobytes()
        key=(host.dtype,b,h,d,hashlib.sha256(first).digest())
        records=self.prefix_histories.get(key,[])
        for record in records:
            overlap=min(n,record['length'])
            if self.byte_equal(record['buffer'][:,:,:overlap],host[:,:,:overlap]):break
        else:
            record=None
        if record is None:
            capacity=1 << (n-1).bit_length()
            size=b*h*capacity*d*host.element_size()
            self.reserve(size);self.host_headroom(size,device.type=='cuda')
            buffer=torch.empty((b,h,capacity,d),dtype=host.dtype,device='cpu')
            buffer[:,:,:n].copy_(host)
            record=dict(buffer=buffer,length=n,bytes=size)
            self.prefix_histories.setdefault(key,[]).append(record);self.saved_bytes+=size
        else:
            self.prefix_reuses+=1
            if n>record['length']:
                if n>record['buffer'].shape[2]:
                    capacity=1 << (n-1).bit_length();size=b*h*capacity*d*host.element_size()
                    self.reserve(size-record['bytes']);self.host_headroom(size,device.type=='cuda')
                    buffer=torch.empty((b,h,capacity,d),dtype=host.dtype,device='cpu')
                    buffer[:,:,:record['length']].copy_(record['buffer'][:,:,:record['length']])
                    self.saved_bytes+=size-record['bytes'];record.update(buffer=buffer,bytes=size)
                record['buffer'][:,:,record['length']:n].copy_(host[:,:,record['length']:n])
                record['length']=n
        return ('prefix',record,n,device,repeat)

    def compact_repeated_heads(self,tensor):
        # GQA repeat_kv materializes four identical heads. Check BYTES, including
        # signed zeros, rather than assuming a shape proves repeated values.
        if tensor.ndim!=4 or tensor.shape[1]%4:return tensor.detach(),1
        b,h,n,d=tensor.shape
        groups=tensor.detach().reshape(b,h//4,4,n,d)
        first=groups[:,:,0]
        original=first.contiguous().view(torch.uint8)
        for head in range(1,4):
            if not torch.equal(original,groups[:,:,head].contiguous().view(torch.uint8)):
                return tensor.detach(),1
        return first.contiguous(),4

    def pack(self,tensor):
        size=tensor.numel()*tensor.element_size()
        if ((not tensor.is_cuda and not self.force_cpu_test) or size<self.minimum_bytes or
                tensor.untyped_storage().data_ptr() in self.parameter_storage):
            return ('direct',tensor.detach())
        payload,repeat=(self.compact_repeated_heads(tensor) if self.policy!='legacy-pinned' else (tensor.detach(),1))
        stored=payload.numel()*payload.element_size()
        prefix=self.policy=='repeat4-prefix-pageable-v2' and repeat==4
        if not prefix:self.reserve(stored)
        pinned=tensor.is_cuda and self.policy=='legacy-pinned'
        if self.policy!='legacy-pinned':self.host_headroom(stored,tensor.is_cuda)
        host=torch.empty_like(payload,device='cpu',pin_memory=pinned)
        # Pageable copies must finish before host data can be staged back to
        # CUDA. This also avoids the unbounded CUDA pinned-host cache.
        host.copy_(payload,non_blocking=pinned)
        self.logical_bytes+=size;self.tensors+=1
        self.compressed_tensors+=int(repeat!=1)
        if prefix:return self.prefix_pack(host,tensor.device,repeat)
        self.saved_bytes+=stored
        return ('offloaded',host,tensor.device,repeat)

    def unpack(self,packed):
        if packed[0]=='direct':return packed[1]
        if packed[0]=='prefix':
            result=packed[1]['buffer'][:,:,:packed[2]].to(packed[3])
            return result.repeat_interleave(packed[4],dim=1)
        result=packed[1].to(packed[2],non_blocking=packed[2].type=='cuda' and self.policy=='legacy-pinned')
        repeat=packed[3] if len(packed)>3 else 1
        return result.repeat_interleave(repeat,dim=1) if repeat!=1 else result

    @contextmanager
    def context(self):
        with torch.autograd.graph.saved_tensors_hooks(self.pack,self.unpack):yield self
