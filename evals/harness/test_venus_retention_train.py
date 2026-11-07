import copy
import pytest
from training.venus.retention_train import validate_parent,PARENT_SHA,PARENT_SOURCE


def test_saved646_is_only_parent_and_cannot_be_mislabelled_as_a_new_update():
    state=dict(step=120,optimizer=dict(update=120),metadata=dict(source=PARENT_SOURCE,
        parent_training_steps=526,capacity_schema='native-capacity-continuation-v1',
        recipe=dict(updates=120,learning_rate=5e-6)))
    validate_parent(state,PARENT_SHA)
    for key in ('step','optimizer','metadata'):
        bad=copy.deepcopy(state)
        if key=='step':bad[key]=8
        elif key=='optimizer':bad[key]['update']=119
        else:bad[key]['source']='different-producer'
        with pytest.raises(ValueError):validate_parent(bad,PARENT_SHA)
    with pytest.raises(ValueError):validate_parent(state,'another-checkpoint')
