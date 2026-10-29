"""New phase clocks must not silently reset inherited Adam/program state."""
import copy

import pytest
import torch

from training.venus import train
from training.venus.checkpoint_restore import restore
from training.venus.phase_transition import begin
from training.venus.state import exact_state, save, rng_state
from training.venus.native_optimizer import assert_optimizer_equal
from evals.harness.test_venus_checkpoint_restore import saved_program_one
from evals.harness.test_venus_native_opal import SOURCE, make


@pytest.mark.skipif(not SOURCE.exists(), reason='Pinned upstream required')
def test_declared_new_schedule_preserves_moments_rng_and_full_custom_program(monkeypatch, tmp_path):
    path, digest, original, _, _ = saved_program_one(monkeypatch, tmp_path)
    original.close()
    model, base, rows = make()
    restored, _ = restore(path, digest, model, base, source=SOURCE, run_dir=tmp_path/'restored-parent')
    parent = torch.load(path, map_location='cpu', weights_only=False)
    old_adam = copy.deepcopy(base.state_dict()['state'])
    old_rng = rng_state()
    recipe = dict(parent['metadata']['recipe'], updates=24, warmup_updates=4)
    phase, proof = begin(parent, restored, model, source=SOURCE, run_dir=tmp_path/'phase',
                         program_dir=tmp_path/'inherited-program', recipe=recipe)
    try:
        assert proof['new_optimizer_updates'] == 0 and proof['parent_program_version'] == 1
        assert phase.optimizer.update == 0 and phase.optimizer.version == 0
        assert phase.optimizer.files == parent['optimizer']['files']
        assert exact_state(old_adam, base.state_dict()['state']) and exact_state(old_rng, rng_state())
        assert phase.scheduler.last_epoch == 0 and base.param_groups[0]['lr'] == recipe['learning_rate']/4
        assert phase.optimizer.updater.last_update == parent['optimizer']['updater']['last_update']
        train.train_step(model, phase.optimizer, None, rows, 0, opal=phase)
        assert all(value['step'].item() == 3 for value in base.state_dict()['state'].values())
        meta = dict(recipe=recipe, parent_training_steps=2, initial_programs=parent['optimizer']['files'])
        phase_path = tmp_path/'new-phase-step1.pt'
        phase_hash = save(phase_path, model, phase.optimizer, 1, meta,
                          diagnostics_state=phase.diagnostics_state)
        fresh, fresh_base, _ = make()
        again, _ = restore(phase_path, phase_hash, fresh, fresh_base, source=SOURCE,
                           run_dir=tmp_path/'phase-independent-restore')
        try:
            assert_optimizer_equal(phase.optimizer.state_dict(), again.optimizer.state_dict())
        finally:
            again.close()
    finally:
        phase.close()
