"""Complete new-data/retention cycles, preserving original retained loss mass."""
from collections import Counter
import math,random
from .matched_schedule import STEPS,EPOCHS,epoch as retained_epoch,validate


def build(groups,rows,*,world_size,epochs=4,seed=42424242):
    validate(groups)
    if type(world_size) is not int or world_size<1 or type(epochs) is not int or not 1<=epochs<=EPOCHS:raise ValueError('Declared distributed complete epochs required')
    if len(rows)!=1072 or len({r['id'] for r in rows})!=1072 or Counter(r['fold'] for r in rows)!={'TRAIN':1008,'DEV':64}:raise ValueError('All1072 preserved source identities/folds required')
    training=[r['id'] for r in rows if r['fold']=='TRAIN'];result=[]
    for e in range(epochs):
        order=list(training);random.Random(seed+e).shuffle(order);old=retained_epoch(groups,e);steps=[]
        for step in range(STEPS):
            ids=order[step*len(order)//STEPS:(step+1)*len(order)//STEPS]
            if not ids:raise ValueError('No empty global new-data update')
            slots=[]
            for i in range(math.ceil(len(ids)/world_size)):
                ranks=[]
                for rank in range(world_size):
                    index=i*world_size+rank;valid=index<len(ids)
                    ranks.append(dict(id=ids[index] if valid else ids[0],valid=valid,padding_only=not valid))
                slots.append(ranks)
            # DDP/FSDP's rank average and whole-global-batch normalization
            # produce49/N per example; the cycle MEAN is exactly1/N.
            steps.append(dict(epoch=e,index=step,retained=old[step],new_global_ids=ids,new_global_valid_count=len(ids),new_coefficient=len(ids)*STEPS/len(training),microbatches=slots,per_example_cycle_mean_weight=1/len(training)))
        seen=[x for s in steps for x in s['new_global_ids']]
        if Counter(seen)!=Counter(training):raise ValueError('Complete new TRAIN coverage exactly once per epoch required')
        result.append(steps)
    return dict(world_size=world_size,epochs=epochs,steps_per_epoch=STEPS,total_new_updates=epochs*STEPS,TRAIN=1008,DEV=64,seed=seed,schedule=result,retained_objectives_and_cycle_mean_weights_unchanged=True,new_padding_targets=0,native_training_approved=False)
