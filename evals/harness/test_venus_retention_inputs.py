from collections import Counter
import copy
import pytest
from training.venus.retention_inputs import epoch_batches,validate_proof,LIMITS


def rows():
    old=[];new=[]
    for family,count in dict(ordinary=32,delegate=48,pending=48,cancel=48,pause=32,quiet=32).items():
        old += [dict(id=f'old-{family}-{i}',fold='TRAIN',family=family) for i in range(count)]
    old += [dict(id=f'dev-{i}',fold='DEV',family='ordinary') for i in range(64)]
    for family,count in dict(ordinary=48,delegate=12,pending=12,cancel=24).items():
        new += [dict(id=f'new-{family}-{i}',fold='TRAIN',family=family) for i in range(count)]
    return old,new


def test_pair_preserves_all_old_train_and_matches_family_exposure_without_dev_leakage():
    old,new=rows();before=copy.deepcopy((old,new));arms={}
    for arm in ('expanded-data','rehearsal'):
        batches=epoch_batches(old,new,arm,7);assert len(batches)==84 and all(len(b)==4 for b in batches)
        flat=[r for b in batches for r in b];arms[arm]=flat
        assert all(r['fold']=='TRAIN' for r in flat)
        assert {r['id'] for r in old if r['fold']=='TRAIN'}<={r['id'] for r in flat}
        assert batches==epoch_batches(old,new,arm,7)
    assert Counter(r['family'] for r in arms['expanded-data'])==Counter(r['family'] for r in arms['rehearsal'])
    assert [r['family'] for r in arms['expanded-data']]==[r['family'] for r in arms['rehearsal']]
    assert len({r['id'] for r in arms['expanded-data']})==336
    assert len({r['id'] for r in arms['rehearsal']})==240
    assert (old,new)==before


def test_changed_folds_or_partial_corpus_never_form_an_epoch():
    old,new=rows()
    with pytest.raises(ValueError):epoch_batches(old[:-1],new,'expanded-data',7)
    new[0]['fold']='DEV'
    with pytest.raises(ValueError):epoch_batches(old,new,'expanded-data',7)


def test_pass_flags_cannot_override_measured_replay_error_or_weakened_limits():
    proof=dict(native_segmented=dict(passed=True,maximum_logit_error=0,compared=32),
        differentiable_input=dict(passed=True,compared=32,limits=LIMITS.copy(),minimum_cosine=1.,relative_l2=0.,maximum_target_ce_delta=0.,argmax_agreement=1.))
    validate_proof(proof)
    bad=copy.deepcopy(proof);bad['native_segmented']['maximum_logit_error']=.001
    with pytest.raises(ValueError):validate_proof(bad)
    bad=copy.deepcopy(proof);bad['differentiable_input']['relative_l2']=.02
    with pytest.raises(ValueError):validate_proof(bad)
    bad=copy.deepcopy(proof);bad['differentiable_input']['limits']['maximum_relative_l2']=.03
    with pytest.raises(ValueError):validate_proof(bad)
