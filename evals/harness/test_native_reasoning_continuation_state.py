"""Continuation provenance cannot silently alter the parent recipe or clocks.

These CPU tests cover metadata and rejection before tensor/runtime loading;
they do not claim to test GPU Adam/RNG restoration or a trained candidate.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals import native_reasoning_continuation_state as state


class ContinuationState(unittest.TestCase):
    def setUp(self):
        self.parent = dict(repo_commit=state.SOURCE, data_sha256='d' * 64,
                           recipe=dict(lr=2e-7, retention_kl_weight=1.,
                                       retention_hidden_mse_weight=10.,
                                       reasoning32=dict(teacher_ids=['preserve-all32'])))
        self.pin = patch.object(state, 'METADATA', state.canonical(self.parent))
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def metadata(self):
        return state.stage_metadata(self.parent, 'a' * 40, 'b' * 64)

    def optimizer(self, clock=528, rate=2e-7):
        return SimpleNamespace(param_groups=[dict(params=list(range(26)), lr=rate,
            betas=(.9, .999), eps=1e-8, weight_decay=0.)],
            state={i: dict(step=clock, exp_avg=.1, exp_avg_sq=.2) for i in range(26)})

    def test_only_source_rate_and_explicit_provenance_change(self):
        original = deepcopy(self.parent)
        with patch.dict(sys.modules, {'torch': None}):
            metadata = self.metadata()
            self.assertTrue(state.validate_metadata(metadata))
        self.assertEqual(self.parent, original)
        self.assertEqual(metadata['recipe']['lr'], 1e-6)
        self.assertEqual(metadata['recipe']['retention_hidden_mse_weight'], 10.)
        self.assertEqual(metadata['continuation']['initial_global_adam_step'], 528)
        self.assertEqual(metadata['continuation']['final_global_adam_step'], 536)
        self.assertEqual(metadata['continuation']['original_fit_actual_exit'], -15)
        self.assertFalse(metadata['continuation']['model_adopted'])

    def test_changed_data_loss_state_provenance_or_implementation_rejected(self):
        for mode in ('data', 'kl', 'hidden', 'teachers', 'lr', 'parent_state',
                     'clock', 'extra', 'objective_code', 'state_code'):
            metadata = self.metadata()
            if mode == 'data': metadata['data_sha256'] = 'c' * 64
            if mode == 'kl': metadata['recipe']['retention_kl_weight'] = 0.
            if mode == 'hidden': metadata['recipe']['retention_hidden_mse_weight'] = 0.
            if mode == 'teachers': metadata['recipe']['reasoning32']['teacher_ids'].pop()
            if mode == 'lr': metadata['recipe']['lr'] = 1e-5
            if mode == 'parent_state': metadata['continuation']['parent_state_sha256'] = 'c' * 64
            if mode == 'clock': metadata['continuation']['initial_global_adam_step'] = 520
            if mode == 'extra': metadata['new_loss_override'] = True
            if mode == 'objective_code': metadata['continuation']['objective_implementation_sha256'] = 'c' * 64
            if mode == 'state_code': metadata['continuation']['state_implementation_sha256'] = 'c' * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                state.validate_metadata(metadata)

    def test_bad_parent_or_noncommitted_identity_cannot_make_a_new_stage(self):
        for producer, authorization in ((state.SOURCE, 'b' * 64), ('short', 'b' * 64),
                                         ('a' * 40, None), ('a' * 40, 'B' * 64)):
            with self.subTest(producer=producer, authorization=authorization), self.assertRaises(ValueError):
                state.stage_metadata(self.parent, producer, authorization)
        changed = deepcopy(self.parent)
        changed['recipe']['lr'] = 1e-5
        with self.assertRaises(ValueError):
            state.stage_metadata(changed, 'a' * 40, 'b' * 64)

    def test_all_adam_clocks_and_hyperparameters_required(self):
        state._adam(self.optimizer(), 528, 2e-7)
        state._adam(self.optimizer(536, 1e-6), 536, 1e-6)
        for mode in ('missing', 'clock', 'moment', 'lr', 'betas', 'eps', 'decay'):
            optimizer = self.optimizer()
            if mode == 'missing': optimizer.state.pop(25)
            if mode == 'clock': optimizer.state[25]['step'] = 527
            if mode == 'moment': optimizer.state[0].pop('exp_avg_sq')
            if mode == 'lr': optimizer.param_groups[0]['lr'] = 1e-5
            if mode == 'betas': optimizer.param_groups[0]['betas'] = (.8, .999)
            if mode == 'eps': optimizer.param_groups[0]['eps'] = 1e-6
            if mode == 'decay': optimizer.param_groups[0]['weight_decay'] = .01
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                state._adam(optimizer, 528, 2e-7)

    def test_changed_parent_files_reject_before_torch_or_optimizer_creation(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / 'training-state.pt').write_bytes(b'wrong-parent')
            with patch.object(state, 'validate_context', return_value=self.parent), \
                 patch.dict(sys.modules, {'torch': None}), self.assertRaises(ValueError):
                state.restore_parent({}, dict(training=folder), 'a' * 40, 'b' * 64)

    def test_invalid_or_dirty_checkpoint_cannot_create_files(self):
        with tempfile.TemporaryDirectory() as name:
            for local_step in (True, 0, 9, 528):
                with self.subTest(local_step=local_step), self.assertRaises(ValueError):
                    state.save_update(name, {}, None, local_step, {})
            with patch.dict(sys.modules, {'torch': None}), self.assertRaises(ValueError):
                state.save_update(name, dict(p=SimpleNamespace(grad=object())),
                                  self.optimizer(529, 1e-6), 1, self.metadata())
            self.assertEqual(list(Path(name).iterdir()), [])

    def test_partial_checkpoint_and_changed_bytes_reject_before_loading(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            metadata = self.metadata()
            with self.assertRaises(FileNotFoundError):
                state.restore_final(folder, {}, None, metadata)
            for filename in ('training-state.pt', 'adapters.safetensors'):
                (folder / filename).write_bytes(filename.encode())
            receipt = dict(schema=state.SCHEMA, local_completed_steps=8,
                global_adam_step=536, metadata_sha256=state.canonical(metadata),
                files={filename:state._sha(folder / filename)
                       for filename in ('training-state.pt', 'adapters.safetensors')})
            (folder / 'training-state.pt').write_bytes(b'changed')
            (folder / 'checkpoint.json').write_text(json.dumps(receipt))
            with patch.dict(sys.modules, {'torch': None}), self.assertRaises(ValueError):
                state.restore_final(folder, {}, None, metadata)

    def test_interrupted_update_directory_is_never_reused(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name) / 'update-001'
            folder.mkdir()
            marker = folder / 'training-state.pt.tmp'
            marker.write_bytes(b'preserve-interrupted-state')
            learned = {str(i):SimpleNamespace(grad=None) for i in range(26)}
            with patch.dict(sys.modules, {'torch': None}), self.assertRaises(FileExistsError):
                state.save_update(name, learned, self.optimizer(529, 1e-6), 1, self.metadata())
            self.assertEqual(marker.read_bytes(), b'preserve-interrupted-state')
            self.assertEqual([p.name for p in folder.iterdir()], ['training-state.pt.tmp'])


if __name__ == '__main__':
    unittest.main()
