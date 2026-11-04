import copy
import pytest
from training.venus.workload import whole_corpus,epoch_seconds
from training.venus.large_train import admission


def example_work():
    rows=[dict(id=str(i),fold='TRAIN' if i<240 else 'DEV',family='delegate',
               tokens=100 if i<4 else 20,segments=100 if i<4 else 20,
               prefix_tokens=100 if i<4 else 20,attention_cells=100 if i<4 else 20,
               head_tokens=100 if i<4 else 20) for i in range(304)]
    return whole_corpus(rows,['0','1','2','3'])


def test_complete_workload_and_measured_control_replace_two_worst_large_arms():
    work=example_work();control=dict(seconds=2,full_dev_seconds=10,optimizer_updates=0,effective_batch=4)
    with pytest.raises(TimeoutError):admission(200,20,10_000,100_000_000)
    result=admission(200,20,10_000,100_000_000,workload=work,control=control)
    assert result['epochs']>=1 and result['joint_required_seconds']<10_000
    assert result['complete_train_per_epoch']==240 and result['full_dev_per_review']==64
    assert result['epoch_runtime_basis']['control']<result['epoch_runtime_basis']['main']


def test_runtime_accounting_rejects_omitted_scene_or_tampered_weights():
    work=example_work()
    with pytest.raises(ValueError):whole_corpus(work['rows'][:-1],work['calibration_ids'])
    bad=copy.deepcopy(work);bad['weights']['4']=0
    with pytest.raises(ValueError):epoch_seconds(20,1,bad)
    bad=copy.deepcopy(work);bad['rows'][4]['fold']='DEV'
    with pytest.raises(ValueError):epoch_seconds(20,1,bad)


def test_retention_additional_forward_is_included_in_all_scene_work():
    work=example_work();rows=copy.deepcopy(work['rows']);rows[4]['family']='ordinary'
    ordinary=whole_corpus(rows,work['calibration_ids'])
    assert ordinary['weights']['4']==2*work['weights']['4']
    assert ordinary['train_weight']>work['train_weight']
