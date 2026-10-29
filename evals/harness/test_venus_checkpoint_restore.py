"""Pinned upstream program1 restoration on CPU fixtures; not native9B proof."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from training.venus import train
from training.venus.checkpoint_restore import restore
from training.venus.lora import adapter_state
from training.venus.native_optimizer import NativeOpal, assert_optimizer_equal
from training.venus.opal_adapter import UPSTREAM_COMMIT
from training.venus.state import save, exact_state, restore_parent
from evals.harness.test_venus_native_opal import SOURCE, setup, make


def saved_program_one(monkeypatch, tmp_path):
    recipe = setup(monkeypatch)
    recipe['opal'] = dict(upstream_commit=UPSTREAM_COMMIT)
    model, base, rows = make()
    runtime = NativeOpal(base, model, source=SOURCE, run_dir=tmp_path/'parent-opal', recipe=recipe)
    train.train_step(model, runtime.optimizer, None, rows, 0, opal=runtime)
    files = dict(runtime.optimizer.files)
    files['updater.py'] = Path('training/venus/opal_programs/lr_half_v1/updater.py').read_text()
    # Exercise the actual pinned response/swap path, including rollback history.
    # This is a synthetic CPU fixture, not a production broker receipt.
    runtime.optimizer._apply(dict(k=1, requested_at=1, files=files,
                                  base=dict(runtime.optimizer.files)), 2)
    assert runtime.optimizer.version == 1
    train.train_step(model, runtime.optimizer, None, rows, 1, opal=runtime)
    runtime.diagnostics_state = dict(train_ce=1., dev_ce=2., last_review_step=2)
    path = tmp_path/'fixture-parent.pt'
    digest = save(path, model, runtime.optimizer, 2, dict(recipe=recipe),
                  diagnostics_state=runtime.diagnostics_state)
    return path, digest, runtime, model, rows


@pytest.mark.skipif(not SOURCE.exists(), reason='Exact upstream checkout required')
def test_complete_program_one_restore_preserves_next_update_and_adam(monkeypatch, tmp_path):
    path, digest, original, model, rows = saved_program_one(monkeypatch, tmp_path)
    fresh, base, fresh_rows = make()
    restored = None
    try:
        restored, proof = restore(path, digest, fresh, base, source=SOURCE, run_dir=tmp_path/'fresh-opal')
        assert proof['program_version'] == 1 and proof['update'] == 2
        assert proof['adam_counters'] == [2] * 4 and proof['new_optimizer_updates'] == 0
        assert not proof['phase_schedule_changed'] and not proof['training_authorized_by_this_helper']
        before = copy.deepcopy(restored.optimizer.state_dict())
        assert_optimizer_equal(original.optimizer.state_dict(), before)
        changed = copy.deepcopy(before)
        changed['dev'] = [dict(update=99, value=1.)]
        with pytest.raises(ValueError, match='dev'):
            assert_optimizer_equal(before, changed)
        expected = train.train_step(model, original.optimizer, None, rows, 2, opal=original)
        actual = train.train_step(fresh, restored.optimizer, None, fresh_rows, 2, opal=restored)
        assert exact_state(adapter_state(model), adapter_state(fresh))
        assert_optimizer_equal(original.optimizer.state_dict(), restored.optimizer.state_dict())
        assert expected['objective'] == actual['objective']
    finally:
        original.close()
        if restored is not None:
            restored.close()


@pytest.mark.skipif(not SOURCE.exists(), reason='Exact upstream checkout required')
def test_wrapped_parent_fails_before_plain_adam_mutation(monkeypatch, tmp_path):
    path, _, runtime, _, _ = saved_program_one(monkeypatch, tmp_path)
    fresh, base, _ = make()
    before = adapter_state(fresh)
    try:
        state = torch.load(path, map_location='cpu', weights_only=False)
        with pytest.raises(ValueError, match='Wrapped OPAL'):
            restore_parent(state, fresh, base)
        assert exact_state(before, adapter_state(fresh)) and not base.state
    finally:
        runtime.close()


@pytest.mark.skipif(not SOURCE.exists(), reason='Exact upstream checkout required')
def test_hash_and_progress_fail_before_model_mutation(monkeypatch, tmp_path):
    path, digest, runtime, _, _ = saved_program_one(monkeypatch, tmp_path)
    fresh, base, _ = make()
    before = adapter_state(fresh)
    try:
        with pytest.raises(ValueError, match='hash'):
            restore(path, '0'*64, fresh, base, source=SOURCE, run_dir=tmp_path/'wrong-hash')
        state = torch.load(path, map_location='cpu', weights_only=False)
        state['optimizer']['update'] += 1
        other = tmp_path/'fixture-invalid-progress.pt'
        torch.save(state, other)
        with pytest.raises(ValueError, match='progress'):
            restore(other, hashlib.sha256(other.read_bytes()).hexdigest(), fresh, base,
                    source=SOURCE, run_dir=tmp_path/'wrong-progress')
        reversed_parameters = list(reversed([p for p in fresh.parameters() if p.requires_grad]))
        wrong_base = torch.optim.AdamW(reversed_parameters, lr=.01)
        with pytest.raises(ValueError, match='parameter order'):
            restore(path, digest, fresh, wrong_base, source=SOURCE, run_dir=tmp_path/'wrong-binding')
        assert exact_state(before, adapter_state(fresh)) and not base.state
    finally:
        runtime.close()


@pytest.mark.skipif(not SOURCE.exists(), reason='Exact upstream checkout required')
def test_distinct_process_restores_program_one_and_next_update(monkeypatch, tmp_path):
    path, digest, runtime, model, rows = saved_program_one(monkeypatch, tmp_path)
    output = tmp_path/'child-fixture-result.pt'
    code = '''
import sys,os,torch,pytest
from pathlib import Path
from evals.harness.test_venus_native_opal import setup,make,SOURCE
from training.venus.checkpoint_restore import restore
from training.venus.lora import adapter_state
from training.venus import train
patch=pytest.MonkeyPatch();setup(patch);model,base,rows=make()
runtime,proof=restore(sys.argv[1],sys.argv[2],model,base,source=SOURCE,run_dir=Path(sys.argv[3]))
try:
    train.train_step(model,runtime.optimizer,None,rows,2,opal=runtime)
    torch.save(dict(adapter=adapter_state(model),optimizer=runtime.optimizer.state_dict(),proof=proof,pid=os.getpid()),sys.argv[4])
finally:
    runtime.close();patch.undo()
'''
    try:
        child_process = subprocess.run([sys.executable, '-c', code, str(path), digest,
                                        str(tmp_path/'child-opal'), str(output)],
                                       capture_output=True, text=True, timeout=30)
        assert child_process.returncode == 0, child_process.stderr[-4000:]
        child = torch.load(output, map_location='cpu', weights_only=False)
        import os
        assert child['pid'] != os.getpid()
        assert child['proof']['new_optimizer_updates'] == 0
        assert child['proof']['model_device'] == 'cpu'
        train.train_step(model, runtime.optimizer, None, rows, 2, opal=runtime)
        assert exact_state(adapter_state(model), child['adapter'])
        assert_optimizer_equal(runtime.optimizer.state_dict(), child['optimizer'])
    finally:
        runtime.close()
