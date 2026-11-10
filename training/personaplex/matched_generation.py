"""Native inference preparation after ALL four complete, restored training arms.

The active trainer is untouched. Generation has its own actual source/lease.
"""
import json
from pathlib import Path
from .matched_reuse import ORIGIN,ORIGINAL_ROOT,DATA_SHA,preflight
from .matched_continue_policy import PACKET_SHA,restored_entry
from . import matched_guard as g


def completed_states(root,trainer_source):
    root=Path(root);phase=json.loads((root/'phase-result.json').read_text())
    terminal_paths=list(root.glob('execution-*/terminal.json'))
    if len(terminal_paths)!=1:raise ValueError('One actual training host terminal required')
    host=json.loads(terminal_paths[0].read_text())
    if (phase.get('source')!=trainer_source or phase.get('all_four_complete') is not True
            or phase.get('partial_lineages') or host.get('source')!=trainer_source or host.get('actual_exit_code')!=0):
        raise ValueError('All four completed actual training arms required before evaluation')
    packet=json.loads((root/'joint-admission.json').read_text())
    if packet.get('source')!=trainer_source or packet.get('data_report_sha256')!=DATA_SHA:
        raise ValueError('Exact training source/data packet required')
    originals=preflight(ORIGINAL_ROOT,PACKET_SHA)['lineages'];originals={(r['base'],r['capacity']):r for r in originals}
    entries=packet['entries']
    if len(entries)!=4 or {(r['base'],r['capacity']) for r in entries}!=set(originals):
        raise ValueError('Every one of the four trained lineages is required')
    selected=[]
    for entry in entries:
        row=dict(entry);folder=root/row['base']/row['capacity'];done=folder/'train/training-terminal.json'
        if done.exists():
            value=json.loads(done.read_text())
            row.update(checkpoint=value['final_checkpoint'],checkpoint_sha256=value['checkpoint_sha256'],
                restore=entry['final_restore'],restore_sha256=g.sha(entry['final_restore']),
                restore_terminal=entry['final_restore_terminal'],previous_training_terminal=str(done),
                previous_training_child_terminal=entry['training_child_terminal'])
            for marker in ('normal88_recovery','postupdate_failure_recovery','quota_interrupted_state','storage_recovery'):row.pop(marker,None)
        step=restored_entry(row,trainer_source,DATA_SHA,originals[(row['base'],row['capacity'])])
        if step!=196 or g.sha(row['checkpoint'])!=row['checkpoint_sha256']:
            raise ValueError('Full196 and actual checkpoint bytes required')
        selected.append(dict(base=row['base'],capacity=row['capacity'],checkpoint=row['checkpoint'],
            checkpoint_sha256=row['checkpoint_sha256'],native_restore=row['restore'],native_restore_sha256=row['restore_sha256'],
            trainer_source=trainer_source,lineage_source=ORIGIN,training_steps=196))
    return selected


def construct_engine(module,loader,**kwargs):
    """Adapters must be present before Engine creates LMGen and warmup graphs."""
    original=module.load_model;module.load_model=loader
    try:return module.Engine(**kwargs)
    finally:module.load_model=original


def load_inference_adapter(lm,capacity,state):
    from .matched_native import configure
    from .matched_state import load_adapter
    # Public loaders return trainable Parameters. Narrow configure deliberately
    # requires ONLY newly installed adapters to own gradients, as in training.
    lm.requires_grad_(False)
    configure(lm,capacity);load_adapter(lm,state)
    return lm.requires_grad_(False).eval()


def make_engine(a,selection):
    import torch
    from .foundation_context import raw_model,WEIGHTS
    from .matched_native import CAPACITIES
    from evals.personaplex_adapters import adapter_parameters
    from contextspan.model import engine as module,weights
    from evals.context_spanning_probe import audited_loads
    original=module.load_model;state=None
    if a.capacity!='baseline':
        if selection['base']!=a.base or selection['capacity']!=a.capacity:
            raise ValueError('Candidate cannot change base or adapter capacity')
        path=Path(selection['checkpoint'])
        if g.sha(path)!=selection['checkpoint_sha256']:raise ValueError('Exact restored candidate checkpoint required')
        state=torch.load(path,map_location='cpu',weights_only=False)
        meta=state['metadata']
        if (state['step']!=196 or state['optimizer']['update']!=196 or meta['source']!=ORIGIN
                or meta['base']!=a.base or meta['capacity']!=a.capacity
                or meta['data_report_sha256']!=DATA_SHA or meta['trainable_parameters']!=CAPACITIES[a.capacity]):
            raise ValueError('Full original named candidate identity required')
    def loader(checkpoint=None,device='cuda',cpu_offload=False):
        if cpu_offload:raise ValueError('No changed offload in this matched comparison')
        if a.base=='personaplex':lm,mimi,spm=raw_model(weights,device),weights.load_mimi(device),weights.load_tokenizer()
        else:lm,mimi,spm=original(checkpoint,device,cpu_offload)
        if state is not None:
            load_inference_adapter(lm,a.capacity,state['adapter'])
        lm.requires_grad_(False).eval();return lm,mimi,spm
    loads=[]
    with audited_loads(loads):
        engine=construct_engine(module,loader,checkpoint=str(WEIGHTS/'context_spanning_7b.pt'),device='cuda',temp=.8,temp_text=.7)
    own=[r for r in loads if r['module_identity']==id(engine.lm)]
    if not own or not own[-1]['strict'] or own[-1]['missing'] or own[-1]['unexpected']:
        raise ValueError('Strict complete public base loading required')
    if engine.lm.dep_q!=16 or engine.frame_size!=1920 or engine.frame_rate!=12.5:
        raise ValueError('Original complete native geometry/clock required')
    actual=adapter_parameters(engine.lm)
    if a.capacity=='baseline':
        if actual:raise ValueError('Untouched baseline unexpectedly has adapters')
    elif (sum(p.numel() for p in actual.values())!=CAPACITIES[a.capacity]
            or any(p.dtype!=torch.float32 for p in actual.values())
            or any(not torch.equal(value.detach().cpu(),state['adapter'][name]) for name,value in actual.items())):
        raise ValueError('Engine warmup changed or bypassed the exact full adapter state')
    engine.evaluation_model_loads=own
    return engine
