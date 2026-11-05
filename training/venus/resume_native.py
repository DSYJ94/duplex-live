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
            or type(state['step']) is not int or not 0<state['step']<60
            or meta['source']!=evidence['source'] or meta['arm']!='main'
            or meta['capacity_schema']!='native-capacity-trained-v1' or meta['parent_training_steps']!=466
            or meta['parent_checkpoint_sha256']!=PARENT_SHA or meta['cache_sha256']!=MANIFEST_SHA
            or meta['expansion']['parameters']!=698351616 or meta['adopted']):
        raise ValueError('Exact actual saved main state and lineage required')
    fields=dict(epochs=1,updates=60,train_count=240,dev_count=64,rank=256,alpha=256,last_layers=36,
                microbatch=1,accumulation=4,learning_rate=1e-5,warmup_updates=16,seed=20261104,
                projections=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                ordinary_kl_weight=.1,ordinary_hidden_weight=.01,action_aux_weight=.25,
                forward_path='native-segmented-bptt-v1',activation_offload_policy='repeat4-prefix-pageable-v3-layout')
    if any(r.get(k)!=v for k,v in fields.items()) or r['opal']['backend']!='keep' or r['opal']['agent_budget_usd']!=0:
        raise ValueError('Original exact one-epoch recipe and OPAL clock required')
    return meta


def verify_proof(proof,evidence,lease,current_pid,boot_id):
    if (not proof['passed'] or not proof['actual_CUDA'] or not proof['all_Adam_OPAL_adapter_RNG_exact']
            or proof['checkpoint_sha256']!=evidence['checkpoint_sha256'] or proof['saved_step']!=evidence['actual_new_updates']
            or proof['source']!=evidence['source'] or proof['new_optimizer_updates']!=0
            or proof['execution_source']!=lease['execution_source'] or proof['vm_id']!=lease['vm_id']
            or proof['disk_id']!=lease['disk_id'] or proof['boot_id']!=boot_id
            or proof['restore_pid']==current_pid):
        raise ValueError('Actual independent same-boot native resume proof required')
    if (lease.get('resume_latest_saved_state') and
            proof.get('seed_receipt_sha256')!=lease['trained_state_receipt_sha256']):
        raise ValueError('Actual selected state must bind the immutable prestart seed')


def verify_selection(selection,seed,lease):
    selected=selection['selected']
    digest=selected['checkpoint_sha256'];step=selected['actual_new_updates']
    if (type(step) is not int or not isinstance(digest,str) or len(digest)!=64
            or any(x not in '0123456789abcdef' for x in digest)):
        raise ValueError('Actual selected checkpoint clock and SHA256 required')
    expected=Path(seed['remote_checkpoint'])
    if digest!=seed['checkpoint_sha256']:
        expected=expected.parent/('main-step-%04d-latest-%s.pt'%(step,digest[:16]))
    if (selection['seed_receipt_sha256']!=lease['trained_state_receipt_sha256']
            or type(selected['actual_new_updates']) is not int
            or not seed['actual_new_updates']<=selected['actual_new_updates']<60
            or any(selected[k]!=seed[k] for k in ('source','vm_id','disk_id','cache_manifest_sha256'))
            or Path(selected['remote_checkpoint'])!=expected
            or digest==seed['checkpoint_sha256'] and step!=seed['actual_new_updates']):
        raise ValueError('Exact same-arm latest saved state and seed binding required')
    return selected


def select_latest(root,checkpoint,seed,lease,seed_sha):
    """Freeze actual serialized state; never infer its clock from a log tail."""
    import torch
    root=Path(root);checkpoint=Path(checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    validate_state(state,seed)
    latest=root/'main/live-resume.pt'
    digest=sha(latest)
    current=torch.load(latest,map_location='cpu',weights_only=False)
    selected={k:seed[k] for k in ('source','vm_id','disk_id','cache_manifest_sha256')}
    selected.update(actual_new_updates=current['step'],checkpoint_sha256=digest)
    validate_state(current,selected)
    if current['step']<state['step']:raise ValueError('Latest saved clock rolled backward')
    if digest!=seed['checkpoint_sha256']:
        checkpoint=checkpoint.parent/('main-step-%04d-latest-%s.pt'%(current['step'],digest[:16]))
        if not checkpoint.exists():os.link(latest,checkpoint)
        if sha(checkpoint)!=digest:raise ValueError('Frozen latest-state snapshot differs')
    selected['remote_checkpoint']=str(checkpoint)
    selection=dict(seed_receipt_sha256=seed_sha,selected=selected,
                   evidence='Actual serialized main/live-resume.pt optimizer clock, metadata and full SHA256',
                   independent_native_restore_still_required=True,no_adoption=True)
    verify_selection(selection,seed,lease)
    return current,selection


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
    selection=None
    if lease.get('resume_latest_saved_state'):
        state,selection=select_latest(root,a.checkpoint,e,lease,sha(a.evidence))
        write(root,'resume-selection-0012.json',selection)
        e=selection['selected']
    else:
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);validate_state(state,e)
    suffix='latest-0012' if selection else '%04d'%e['actual_new_updates']
    folder=root/('resume-validation-'+suffix);folder.mkdir(exist_ok=False)
    corpus=Corpus(a.cache,action_index=a.action_index,admission=a.data_admission)
    model,_=load_native(a.model,language_only=True);llm=model.llm;llm.eval();llm.gradient_checkpointing_disable()
    runtime=restore_expanded(state,llm,a.opal_source,folder)
    try:
        row=next(r for r in sorted(corpus.rows,key=lambda r:r['id']) if r['fold']=='TRAIN' and r['family']=='ordinary')
        with torch.no_grad():logits,_=selected_forward(llm,corpus.read(row),'cuda')
        restore_rng(state['rng'])
        if not exact_state(state['rng'],rng_state()):raise ValueError('Saved CPU/CUDA RNG differs')
        result=dict(passed=True,actual_CUDA=True,source=a.source,
            execution_source=execution,vm_id=lease['vm_id'],disk_id=lease['disk_id'],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),restore_pid=os.getpid(),
            checkpoint_sha256=e['checkpoint_sha256'],saved_step=e['actual_new_updates'],new_optimizer_updates=0,
            all_Adam_OPAL_adapter_RNG_exact=True,native_reference=dict(id=row['id'],logits_sha256=tensor_sha(logits)),adopted=False)
        if selection:result.update(seed_receipt_sha256=lease['trained_state_receipt_sha256'],selection_sha256=sha(root/'resume-selection-0012.json'))
        write(root,'fresh-resume-'+suffix+'.json',result)
    finally:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('model','cache','action-index','data-admission','opal-source','lease','source','root','checkpoint','evidence'):p.add_argument('--'+n,required=True)
    main(p.parse_args())
