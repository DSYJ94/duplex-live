import copy
import json
from pathlib import Path

import pytest

from training.venus.resume_schedule import remaining_schedule


def original_evidence():
    root=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v73')
    work=json.loads((root/'workload-geometry.json').read_text());work.pop('source')
    measured=json.loads((root/'admission.json').read_text())['measured']
    control=json.loads((root/'control-runtime.json').read_text())['measured']
    dev=json.loads((root/'main/dev-baseline.json').read_text())['seconds']
    return work,measured['seconds'],control,dev


def test_remaining_work_is_exact_suffix_of_original_schedule_with_all_margins():
    from training.venus.large_train import batches
    work,seconds,control,dev=original_evidence()
    result=remaining_schedule(work,seconds,control,dev,24)
    original=[r['id'] for batch in batches(work['rows'],1,20261104) for r in batch]
    assert result['completed_train_ids']==original[:96]
    assert result['remaining_train_ids']==original[96:]
    assert len(set(result['completed_train_ids']+result['remaining_train_ids']))==240
    assert result['remaining_main_updates']==36 and result['remaining_control_updates']==60
    assert result['train_margin']==1.35 and result['dev_margin']==1.25
    assert result['final_reserve_seconds']==4200 and result['full_dev_per_review']==64
    assert result['joint_required_seconds']==pytest.approx(39906.6193459921)
    assert result['joint_required_seconds']+1800<51600
    later=remaining_schedule(work,seconds,control,dev,26)
    assert later['completed_train_ids']==original[:104] and later['remaining_train_ids']==original[104:]
    assert later['joint_required_seconds']<result['joint_required_seconds']


def test_modified_data_geometry_and_invalid_saved_clocks_are_rejected():
    work,seconds,control,dev=original_evidence()
    for step in (0,60,True,24.5):
        with pytest.raises(ValueError):remaining_schedule(work,seconds,control,dev,step)
    for field in ('weights','train_ids'):
        broken=copy.deepcopy(work)
        if field=='weights':broken[field][broken['train_ids'][0]]*=.1
        else:broken[field].pop()
        with pytest.raises(ValueError):remaining_schedule(broken,seconds,control,dev,24)
    with pytest.raises(ValueError):remaining_schedule(work,seconds,dict(control,optimizer_updates=1),dev,24)
