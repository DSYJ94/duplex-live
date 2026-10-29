import pytest
from training.venus.diagnostics import aggregate,coverage,review


def test_group_losses_weight_actual_labelled_scenes_not_missing_text_groups():
    first=aggregate([('quiet',dict(ce=2.,decision_ce=2.,text_ce=None),1),('quiet',dict(ce=4.,decision_ce=3.,text_ce=5.),1)])
    merged=aggregate([('quiet',first['quiet'],2),('quiet',dict(ce=1.,decision_ce=1.,text_ce=1.),1)])
    assert merged['quiet']['count']==3 and merged['quiet']['text_ce_count']==2
    assert merged['quiet']['text_ce']==3. and merged['quiet']['ce']==pytest.approx(7/3)


def test_coverage_reports_counterfactual_reuse_and_excludes_heldout():
    rows=[dict(fold='TRAIN',family='pending',user_wav_sha256='a'*64) for _ in range(4)]
    rows+=[dict(fold='DEV',family='pending',user_wav_sha256='b'*64),dict(fold='HOLDOUT',family='pending',user_wav_sha256='secret')]
    result=coverage({'scenes':rows})
    assert result['TRAIN']['pending']==dict(scenes=4,unique_user_timelines=1)
    assert result['DEV']['pending']==dict(scenes=1,unique_user_timelines=1)
    assert 'HOLDOUT' not in result


def test_review_flags_divergence_without_claiming_dataset_causality():
    train=aggregate([('ordinary',dict(ce=1.,decision_ce=.5,text_ce=1.5),4)])
    dev=aggregate([('ordinary',dict(ce=2.),12)])
    rows=[dict(step=8,family_train_metrics=train,gradient_norm=2.,gradient_clip=1.)]
    packet,state=review(rows,8,dict(train_ce=1.5,dev_ce=1.8),dev=dev)
    assert packet['signals'] and not packet['causal_diagnosis_established']
    assert not packet['heldout_available'] and packet['agent_metrics']['observed_reference_clip_fraction']==1
    assert state==dict(train_ce=1.,dev_ce=2.,last_review_step=8)


def test_unknown_agent_clipping_is_not_reported_as_unclipped():
    packet,_=review([dict(step=8,gradient_clip=None,gradient_norm=2.)],8,{})
    assert 'observed_reference_clip_fraction' not in packet['agent_metrics']


def test_nonfinite_metric_and_unknown_family_rejected():
    with pytest.raises(ValueError):aggregate([('ordinary',dict(ce=float('nan')),1)])
    with pytest.raises(ValueError):aggregate([('not-a-family',dict(ce=1.),1)])


def test_opal_phase_stops_after_one_complete_epoch_and_reserves_full_dev(monkeypatch):
    from training.venus.train import RECIPE,admit_epochs
    for key,value in dict(train_count=216,maximum_complete_epochs=1,opal={'enabled':True},post_training_reserve_seconds=3600).items():
        monkeypatch.setitem(RECIPE,key,value)
    result=admit_epochs(10,30,14400)
    assert result['epochs']==1 and result['updates']==54
    assert result['maximum_complete_epochs']==1
    assert result['conservative_required_seconds']==54*10*1.35+30*1.25+3600
