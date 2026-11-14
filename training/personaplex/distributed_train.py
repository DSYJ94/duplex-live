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
           expanded_row,valid,expanded_coefficient,identity,checkpoint_path,log_path,expanded_microbatches=None):
    if type(step) is not int or step<0 or checkpoint_manifest(opal.active.optimizer)['update']!=step:
        raise ValueError('Exact current OPAL step required; no replay of saved updates')
    if not identity.get('native_data_qualified') or not identity.get('native_fresh_restore_passed'):
        raise ValueError('Actual complete native data and fresh restore admission required')
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
    local=sum(x['weighted_loss'] for x in retained['records'])+sum(x['weighted_loss'] for x in expanded)
    values=[None]*dist.get_world_size();dist.all_gather_object(values,local)
    if not all(math.isfinite(x) for x in values):raise ValueError('Finite complete global native objective required')
    loss=sum(values)/len(values)
    result=opal.step(loss,step)
    # The admitted phase chooses durable epoch/rolling boundaries. Do not
    # create a full native-LM checkpoint at every update by default.
    if checkpoint_path is not None:
        save(checkpoint_path,runtime,base,identity=identity,step=step+1,
             rank_state=opal.rank_state(rng_state()))
    record=dict(step=step+1,previous_step=step,actual_optimizer_updates=1,
        global_loss=loss,retained=retained,expanded=expanded,opal=result,
        checkpoint=None if checkpoint_path is None else str(checkpoint_path),rank=dist.get_rank(),world=dist.get_world_size())
    with Path(log_path).open('a') as stream:
        stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    return record
