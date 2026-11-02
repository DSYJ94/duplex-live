import pytest
from training.venus.context_controller import require_complete_context_cache


def complete():
    old=[dict(id=f'old-{i}',fold='TRAIN' if i<216 else 'DEV',file=f'old/{i}.pt') for i in range(268)]
    new=[dict(id=f'new-{i}',fold='TRAIN' if i<24 else 'DEV',file=f'new/{i}.pt',native_prefix_proof=dict(passed=True,logit_error=0,hidden_error=0)) for i in range(36)]
    return dict(scenes=old+new),dict(approved=True,complete_new36_native_replay=True),dict(complete=True,rows=[{}]*304)


def test_all36_exact_new_histories_and_all_old_folds_required():
    m,a,i=complete();require_complete_context_cache(m,a,i)
    m['scenes'].pop()
    with pytest.raises(ValueError):require_complete_context_cache(m,a,i)


def test_successful_unit_status_cannot_replace_missing_or_failed_data():
    m,a,i=complete();a['complete_new36_native_replay']=False
    with pytest.raises(ValueError):require_complete_context_cache(m,a,i)
    m,a,i=complete();m['scenes'][-1]['native_prefix_proof']['hidden_error']=.001
    with pytest.raises(ValueError):require_complete_context_cache(m,a,i)
