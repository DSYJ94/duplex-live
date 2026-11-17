"""One admitted native update: retained losses, expanded speech, OPAL, checkpoint.

The phase owner supplies qualified tensors and a complete coverage schedule.
This executor does not submit jobs, generate approvals or repeat saved updates.
"""
import json,math,os
from pathlib import Path
import torch.distributed as dist
from .distributed_objective import backward_retained
from .distributed_expanded_objective import backward as backward_expanded
from .distributed_state import save
from training.venus.opal_adapter import checkpoint_manifest
from training.venus.state import rng_state


def update(runtime,base,opal,*,step,retained_groups,references,exposures,
           expanded_row,valid,expanded_coefficient,identity,checkpoint_path,log_path,expanded_microbatches=None,admission=None,
           language_window=None):
    if type(step) is not int or step<0 or checkpoint_manifest(opal.active.optimizer)['update']!=step:
        raise ValueError('Exact current OPAL step required; no replay of saved updates')
    proof=identity if admission is None else admission
    if not proof.get('native_data_qualified') or not proof.get('native_fresh_restore_passed'):
        raise ValueError('Actual complete native data and fresh restore admission required')
    if ('language_training' in identity)!=(language_window is not None):
        raise ValueError('Language schedule cannot be silently added or omitted from an existing lineage')
    if language_window is not None:
        from .text_rehearsal import validate_window_identity
        validate_window_identity(language_window,identity,proof,step)
        from .distributed_objective import agree
        agree(dict(identity=language_window['identity'],step=step))
    if checkpoint_path is not None and Path(checkpoint_path).exists():raise ValueError('Immutable next checkpoint already exists')
    microbatches=[dict(row=expanded_row,valid=valid)] if expanded_microbatches is None else expanded_microbatches
    if not microbatches or any(type(x['valid']) is not bool for x in microbatches):raise ValueError('Declared nonempty native microbatch slots required')
    local_geometry=(len(microbatches),sum(x['valid'] for x in microbatches))
    geometries=[None]*dist.get_world_size();dist.all_gather_object(geometries,local_geometry)
    if any(x[0]!=len(microbatches) for x in geometries):raise ValueError('Ranks must execute identical native collective counts')
    global_count=sum(x[1] for x in geometries)
    if global_count<1:raise ValueError('Nonempty complete global new-data batch required')
    base.zero_grad(set_to_none=True)
    retained=backward_retained(runtime,retained_groups,references,exposures)
    expanded=[backward_expanded(runtime,x['row'],valid=x['valid'],coefficient=expanded_coefficient,global_batch_count=global_count) for x in microbatches]
    language=None
    if language_window is not None:
        from .text_rehearsal import backward_window
        language=backward_window(runtime.view,language_window['batches'],language_window['spec'],
                                coefficient=language_window['identity']['coefficient'])
    local=(sum(x['weighted_loss'] for x in retained['records'])+sum(x['weighted_loss'] for x in expanded)
           +(sum(x['weighted_loss'] for x in language['batches']) if language is not None else 0.))
    values=[None]*dist.get_world_size();dist.all_gather_object(values,local)
    if not all(math.isfinite(x) for x in values):raise ValueError('Finite complete global native objective required')
    loss=sum(values)/len(values)
    result=opal.step(loss,step)
    # The admitted phase chooses durable epoch/rolling boundaries. Do not
    # create a full native-LM checkpoint at every update by default.
    if checkpoint_path is not None:
        rank_state=opal.rank_state(rng_state())
        if language_window is not None:
            rank_state['language_cursor']=dict(identity=language_window['identity'],next_update=step+1)
        save(checkpoint_path,runtime,base,identity=identity,step=step+1,
             rank_state=rank_state)
    record=dict(step=step+1,previous_step=step,actual_optimizer_updates=1,
        global_loss=loss,retained=retained,expanded=expanded,opal=result,
        checkpoint=None if checkpoint_path is None else str(checkpoint_path),rank=dist.get_rank(),world=dist.get_world_size())
    if language_window is not None:
        record.update(language=language,language_cursor=dict(identity=language_window['identity'],next_update=step+1))
    with Path(log_path).open('a') as stream:
        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return record
