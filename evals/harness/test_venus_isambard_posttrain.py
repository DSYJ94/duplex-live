import hashlib
import json

import pytest

from training.venus import isambard_posttrain as post


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completed(tmp_path):
    root = tmp_path / 'training'
    checkpoint = root / 'main/step-0120.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'fixture-only-completed-state')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    execution = root / ('execution-' + post.TRAINING_JOB)
    write(execution / 'terminal.json', dict(actual_exit_code=0, job_id=post.TRAINING_JOB,
          stage='train', source=post.TRAINING_SOURCE))
    write(execution / 'final-restore-terminal.json', dict(actual_exit_code=0,
          source=post.TRAINING_SOURCE, execution_id='restore'))
    write(root / 'main/training-terminal.json', dict(completed=True, actual_new_updates=120,
          restored_saved_updates=8, new_updates_this_job=112, total_optimizer_updates=646,
          complete_epochs=2, TRAIN=240, DEV=64))
    write(root / 'main/final-native-reference.json', dict(source=post.TRAINING_SOURCE,
          checkpoint_sha256=digest, training_execution_id='training'))
    write(root / 'main/final-native-restore.json', dict(passed=True, source=post.TRAINING_SOURCE,
          checkpoint_sha256=digest, new_optimizer_updates=0, actual_CUDA=True,
          Adam_RNG_OPAL_exact=True, native_logits_exact=True,
          training_execution_id='training', restore_execution_id='restore'))
    return root


def test_complete_phase_and_actual_distinct_final_restore_admit(tmp_path):
    root = completed(tmp_path)
    assert post.completed_training(root)['checkpoint_sha256'] == post.sha(root / 'main/step-0120.pt')


@pytest.mark.parametrize('file,field,value', [
    ('main/training-terminal.json', 'actual_new_updates', 119),
    ('main/training-terminal.json', 'new_updates_this_job', 120),
    ('main/final-native-restore.json', 'restore_execution_id', 'training'),
    ('execution-8156150/final-restore-terminal.json', 'actual_exit_code', 1),
    ('execution-8156150/terminal.json', 'source', 'relabelled'),
])
def test_partial_replayed_relabelled_or_failed_final_state_rejected(tmp_path, file, field, value):
    root = completed(tmp_path)
    path = root / file
    row = json.loads(path.read_text()); row[field] = value; write(path, row)
    with pytest.raises(ValueError):
        post.completed_training(root)


def test_changed_completed_weights_rejected(tmp_path):
    root = completed(tmp_path)
    (root / 'main/step-0120.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):
        post.completed_training(root)


def test_comparison_requires_all32_and_original_audio_without_rerunning_baseline(tmp_path, monkeypatch):
    baseline = tmp_path / 'original526'
    root = tmp_path / 'evaluation646'
    monkeypatch.setattr(post, 'BASELINE_ROOT', baseline)
    for collection, source in ((baseline, post.BASELINE_SOURCE), (root / 'native/main', 'new-source')):
        rows, cases = [], []
        for i in range(32):
            ident = 'case-%02d' % i
            wav = collection / ident / 'joined-emitted.wav'
            wav.parent.mkdir(parents=True)
            wav.write_bytes(('original-fixture-audio:' + str(i)).encode())
            rows.append(dict(id=ident, joined_wav_sha256=post.sha(wav)))
            cases.append(dict(id=ident, input_sha256=str(i)))
        write(collection / 'native-report.json', dict(complete=True, source=source, cases=rows))
        write(collection / 'results.json', dict(cases=cases))
    original = (baseline / 'native-report.json').read_bytes()
    post.compare(root, 'new-source')
    result = json.loads((root / 'native/continuation-comparison.json').read_text())
    assert result['outputs'] == 64 and result['baseline_reused'] and not result['qualified']
    assert (baseline / 'native-report.json').read_bytes() == original
    # Corruption of one original recording must not become a passing subset.
    (baseline / 'case-00/joined-emitted.wav').write_bytes(b'changed')
    with pytest.raises(ValueError):
        post.compare(root, 'new-source')


def test_fold_comparison_reuses_same646_and_rejects_checkpoint_or_reference_changes(tmp_path, monkeypatch):
    baseline=tmp_path/'unmerged646';root=tmp_path/'folded646'
    monkeypatch.setattr(post,'UNMERGED646_ROOT',baseline)
    for collection,source,fold in ((baseline,post.UNMERGED646_SOURCE,False),(root/'native/main','fold-source',True)):
        rows=[];cases=[]
        for i in range(32):
            ident='case-%02d'%i;wav=collection/ident/'joined-emitted.wav';wav.parent.mkdir(parents=True)
            wav.write_bytes(('unchanged-fixture:'+str(i)).encode());rows.append(dict(id=ident,joined_wav_sha256=post.sha(wav)));cases.append(dict(id=ident,input_sha256=str(i)))
        write(collection/'native-report.json',dict(complete=True,source=source,cases=rows))
        write(collection/'results.json',dict(cases=cases))
        write(collection/'profile.json',dict(checkpoint_sha256='actual-same646',voice_reference=dict(sha256='same-rp-reference'),inference_lora_fold=dict(enabled=fold)))
    post.compare(root,'fold-source','unmerged646')
    receipt=json.loads((root/'native/continuation-comparison.json').read_text())
    assert receipt['baseline_global_update']==receipt['candidate_global_update']==646
    assert receipt['baseline_source']==post.UNMERGED646_SOURCE and not receipt['qualified']
    p=root/'native/main/profile.json';row=json.loads(p.read_text());row['checkpoint_sha256']='different';write(p,row)
    with pytest.raises(ValueError):post.compare(root,'fold-source','unmerged646')
