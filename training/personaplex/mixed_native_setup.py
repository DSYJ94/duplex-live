"""Fresh common-foundation setup for broad/selective/full mixed native learning.

No jobs, optimizer steps or old196 resumes. The future phase must separately
admit qualified language/speech data, storage, full state and a useful schedule.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

METHODS=('broad-lora64','selective-trunk-text','full-lm')


def build(method,groups,folder,*,device,guard,source):
    if method not in METHODS:
        raise ValueError('Declared matched broad/selective/full method required')
    lease=guard(600)
    if lease.get('source')!=source or not lease.get('mixed_training_authorization_sha256'):
        raise ValueError('Separate admitted mixed native phase required before loading weights')
    import torch
    import torch.distributed as dist
    if (not dist.is_initialized() or dist.get_world_size()!=4 or not torch.cuda.is_available()
            or torch.cuda.device_count()!=4 or torch.device(device).type!='cuda'
            or torch.device(device).index!=int(os.environ['LOCAL_RANK'])):
        raise ValueError('Exactly four actual collective native GPUs and local device required')
    torch.cuda.set_device(torch.device(device))
    from .matched_native import model,fingerprint
    from .distributed_native import DistributedNative
    from evals.context_native_audio49_data import digest
    from training.venus.state import rng_state,restore_rng
    folder=Path(folder)
    lm,loads=model(SimpleNamespace(base='personaplex'),folder)
    before=rng_state();baseline=fingerprint(lm,groups);references={};repeated={}
    with torch.no_grad():
        for row in groups['reference']:
            mask=row['mask'].cuda();hidden,logits=lm.forward_codes(row['inputs'].cuda())
            repeated[row['id']]=dict(hidden=digest(hidden[:,mask]),text_logits=digest(logits[0,0,mask]))
            references[row['id']]=(hidden[:,mask].detach().cpu().clone(),
                                   logits[0,0,mask].float().softmax(-1).detach().cpu())
    restore_rng(before)
    if repeated!=baseline:raise ValueError('Fresh common foundation must reproduce all four native references')
    def ownership_only(current,ownership):
        # FP32 master conversion is exact on BF16 parameter values. Do not run
        # an unwrapped FP32 forward and mislabel its compute-precision change
        # as the BF16 native inference function. The real check is post-wrap.
        if method=='broad-lora64' and ownership['trainable_parameters']!=105070592:
            raise ValueError('Exact broad adapter ownership required')
        if method!='broad-lora64' and ownership['details'].get('new_model_weights')!=0:
            raise ValueError('Selective/full methods must not silently add model weights')
    runtime=DistributedNative(lm,method,device,after_configure=ownership_only)
    guard(600);before=rng_state();wrapped=fingerprint(runtime.view,groups);restore_rng(before)
    if wrapped!=baseline:
        raise ValueError('Wrapped native BF16 function changed before any learning; stop this method')
    report=dict(source=source,method=method,base='personaplex',ownership=runtime.ownership,
                fresh_released_foundation=True,old_saved_adapter_used=False,old_updates_replayed=0,
                raw_repeated=baseline==repeated,wrapped_native_exact=True,baseline=baseline,
                master_precision='FP32',sharded_compute_precision='BF16' if runtime.sharded else 'native BF16 plus FP32 adapters',
                full_Adam_RNG_OPAL_restore_passed=False,training_admitted=False,optimizer_updates=0)
    with (folder/'mixed-native-setup.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    return dict(runtime=runtime,references=references,baseline=baseline,report=report,load_receipts=loads)
