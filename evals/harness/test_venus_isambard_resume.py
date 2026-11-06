import copy
import pytest
from training.venus.isambard_resume import validate,SAVED_SHA,SAVED_SOURCE

def state():
    return dict(step=8,optimizer=dict(update=8),metadata=dict(source=SAVED_SOURCE,capacity_schema='native-capacity-continuation-v1',parent_training_steps=526,recipe=dict(updates=120,epochs=2,seed=20261106,learning_rate=5e-6)))

def test_exact_saved_clock_and_phase_recipe_admitted():validate(state(),SAVED_SHA)

@pytest.mark.parametrize('change',[lambda x:x.update(step=14),lambda x:x['optimizer'].update(update=0),lambda x:x['metadata']['recipe'].update(updates=60),lambda x:x['metadata'].update(source='relabelled')])
def test_unsaved_clock_or_changed_recipe_never_relabelled_as_recovered(change):
    s=state();change(s)
    with pytest.raises(ValueError):validate(s,SAVED_SHA)

def test_other_artifact_rejected():
    with pytest.raises(ValueError):validate(state(),'bad')
