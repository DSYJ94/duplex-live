import json,copy,pytest
from pathlib import Path
from training.venus.retention_evaluate import completed_training,TRAINING_JOB,TRAINING_SOURCE
from training.venus.large_inputs import sha


def fixture(root):
    arm=root/'expanded-data';arm.mkdir();execution=root/('execution-'+TRAINING_JOB);execution.mkdir()
    checkpoint=arm/'step-0084.pt';checkpoint.write_bytes(b'fixture-immutable-state');digest=sha(checkpoint)
    records={execution/'terminal.json':dict(actual_exit_code=0,job_id=TRAINING_JOB,source=TRAINING_SOURCE,stage='retention-train'),execution/'expanded-data-restore-terminal.json':dict(actual_exit_code=0,source=TRAINING_SOURCE,execution_id='restore'),arm/'training-terminal.json':dict(completed=True,actual_new_updates=84,total_optimizer_updates=730,TRAIN_exposures=336,DEV=64,arm='expanded-data'),arm/'final-native-reference.json':dict(source=TRAINING_SOURCE,training_execution_id='training',checkpoint_sha256=digest),arm/'final-native-restore.json':dict(passed=True,source=TRAINING_SOURCE,new_optimizer_updates=0,actual_CUDA=True,Adam_RNG_OPAL_exact=True,native_logits_exact=True,training_execution_id='training',restore_execution_id='restore',checkpoint_sha256=digest),arm/'parent-restore.json':dict(passed=True,source=TRAINING_SOURCE,Adam_RNG_OPAL_exact=True,native_logits_exact=True,checkpoint_sha256='46e84c6d7b1d02ec16563c1d0e666230f8644963d3406b0f095bab97938950aa')}
    for p,r in records.items():p.write_text(json.dumps(r))
    return records


def test_whole_training_and_distinct_actual_final_restore_are_required(tmp_path):
    records=fixture(tmp_path);assert completed_training(tmp_path,'expanded-data')['optimizer_updates']==730
    for path,key,value in [(tmp_path/'execution-8179044/terminal.json','actual_exit_code',1),(tmp_path/'expanded-data/training-terminal.json','actual_new_updates',83),(tmp_path/'expanded-data/final-native-restore.json','restore_execution_id','training')]:
        changed=dict(records[path]);changed[key]=value;path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):completed_training(tmp_path,'expanded-data')
        path.write_text(json.dumps(records[path]))
    (tmp_path/'expanded-data/step-0084.pt').write_bytes(b'changed-state')
    with pytest.raises(ValueError):completed_training(tmp_path,'expanded-data')


def test_factual_screen_is_not_word_agreement():
    from training.venus.retention_comparison import factual_screen
    assert factual_screen('case-0222','It guides liquid into a narrow opening.')
    assert not factual_screen('case-0222','It filters liquids.')
    assert factual_screen('case-0217','Twelve.')


def test_new_candidate_loader_does_not_reinterpret_old_lineage_or_changed_objectives():
    torch=pytest.importorskip('torch')
    if not hasattr(getattr(torch,'nn',None),'Module'):pytest.skip('Actual Torch required; run on ARM container')
    from training.venus.large_inference import validate_data_phase
    from training.venus.lora import ALL_PROJECTIONS
    from training.venus.large_inputs import MANIFEST_SHA
    meta=dict(capacity_schema='native-retention96-data-v1',arm='expanded-data',source=TRAINING_SOURCE,
        parent_training_steps=646,parent_checkpoint_sha256='46e84c6d7b1d02ec16563c1d0e666230f8644963d3406b0f095bab97938950aa',
        original_capacity_metadata=dict(source='d8161fd40c80b814242f93c8d205cf33a83c4c54'),adopted=False,
        phase_transition=dict(adam_moments_and_counters_exact=True,rng_exact=True),
        composite_manifest=dict(schema='dl-preserved304-plus-native96-v1',old_manifest_sha256=MANIFEST_SHA,new_manifest_sha256='c33f520d1fddc8778e8e8fb9e0f9f2cb7c978df4cdb194a78f48e49dafef6cd8',TRAIN=336,DEV=64),
        recipe=dict(updates=84,epochs=1,rank=256,alpha=256,last_layers=36,ordinary_kl_weight=.1,ordinary_hidden_weight=.01,action_aux_weight=.25,microbatch=1,accumulation=4,forward_path='native-segmented-bptt-v1',projections=list(ALL_PROJECTIONS)))
    state=dict(step=84,metadata=meta);assert validate_data_phase(state)==meta
    for field,value in [('parent_training_steps',526),('source','other-producer'),('arm','main'),('adopted',True)]:
        bad=copy.deepcopy(state);bad['metadata'][field]=value
        with pytest.raises(ValueError):validate_data_phase(bad)
    bad=copy.deepcopy(state);bad['metadata']['recipe']['ordinary_kl_weight']=0
    with pytest.raises(ValueError):validate_data_phase(bad)
