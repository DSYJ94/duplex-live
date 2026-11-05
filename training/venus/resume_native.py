"""Independent exact restore of the actually saved post-update native state."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .large_inputs import PARENT_SHA,MANIFEST_SHA,sha


def validate_state(state,evidence):
    meta=state['metadata'];r=meta['recipe']
    if (state['step']!=evidence['actual_new_updates'] or state['optimizer']['update']!=state['step']
            or state['step']!=8 or meta['source']!=evidence['source'] or meta['arm']!='main'
            or meta['capacity_schema']!='native-capacity-trained-v1' or meta['parent_training_steps']!=466
            or meta['parent_checkpoint_sha256']!=PARENT_SHA or meta['cache_sha256']!=MANIFEST_SHA
            or meta['expansion']['parameters']!=698351616 or meta['adopted']):
        raise ValueError('Exact actual saved main8 state and lineage required')
    fields=dict(epochs=1,updates=60,train_count=240,dev_count=64,rank=256,alpha=256,last_layers=36,
                microbatch=1,accumulation=4,learning_rate=1e-5,warmup_updates=16,
                ordinary_kl_weight=.1,ordinary_hidden_weight=.01,action_aux_weight=.25,
                forward_path='native-segmented-bptt-v1',activation_offload_policy='repeat4-prefix-pageable-v3-layout')
    if any(r.get(k)!=v for k,v in fields.items()) or r['opal']['backend']!='keep' or r['opal']['agent_budget_usd']!=0:
        raise ValueError('Original exact one-epoch recipe and OPAL clock required')
    return meta


def verify_proof(proof,evidence,lease,current_pid,boot_id):
    if (not proof['passed'] or not proof['actual_CUDA'] or not proof['all_Adam_OPAL_adapter_RNG_exact']
            or proof['checkpoint_sha256']!=evidence['checkpoint_sha256'] or proof['saved_step']!=8
            or proof['source']!=evidence['source'] or proof['new_optimizer_updates']!=0
            or proof['execution_source']!=lease['execution_source'] or proof['vm_id']!=lease['vm_id']
            or proof['disk_id']!=lease['disk_id'] or proof['boot_id']!=boot_id
            or proof['restore_pid']==current_pid):
        raise ValueError('Actual independent same-boot native resume proof required')


def main(a):
    import torch
    from .large_native import guarded,restore_expanded,tensor_sha,write
    from .large_inputs import Corpus
    from .native_cache import load_native
    from .lora import selected_forward
    from .state import restore_rng,rng_state,exact_state
    from .sustained import validate
    guarded(a.lease,900)
    root=Path(a.root);lease=json.loads(Path(a.lease).read_text());e=json.loads(Path(a.evidence).read_text())
    execution=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if (execution!=lease['execution_source'] or sha(a.evidence)!=lease['trained_state_receipt_sha256']
            or e['source']!=a.source or e['vm_id']!=lease['vm_id'] or e['disk_id']!=lease['disk_id']
            or sha(a.checkpoint)!=e['checkpoint_sha256']):
        raise ValueError('Actual frozen checkpoint hash and execution binding required')
    validate(root,a.source)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);validate_state(state,e)
    folder=root/'resume-validation-0008';folder.mkdir(exist_ok=False)
    corpus=Corpus(a.cache,action_index=a.action_index,admission=a.data_admission)
    model,_=load_native(a.model,language_only=True);llm=model.llm;llm.eval();llm.gradient_checkpointing_disable()
    runtime=restore_expanded(state,llm,a.opal_source,folder)
    try:
        row=next(r for r in sorted(corpus.rows,key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
        with torch.no_grad():logits,_=selected_forward(llm,corpus.read(row),'cuda')
        restore_rng(state['rng'])
        if not exact_state(state['rng'],rng_state()):raise ValueError('Saved CPU/CUDA RNG differs')
        write(root,'fresh-resume-0008.json',dict(passed=True,actual_CUDA=True,source=a.source,
            execution_source=execution,vm_id=lease['vm_id'],disk_id=lease['disk_id'],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),restore_pid=os.getpid(),
            checkpoint_sha256=e['checkpoint_sha256'],saved_step=8,new_optimizer_updates=0,
            all_Adam_OPAL_adapter_RNG_exact=True,native_reference=dict(id=row['id'],logits_sha256=tensor_sha(logits)),adopted=False))
    finally:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('model','cache','action-index','data-admission','opal-source','lease','source','root','checkpoint','evidence'):p.add_argument('--'+n,required=True)
    main(p.parse_args())
