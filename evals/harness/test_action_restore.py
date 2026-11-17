"""Restore seam tests; fake OPAL transport is not upstream/GPU qualification."""
import copy
import unittest
from types import SimpleNamespace
import torch
from training.personaplex.action_fit import model
from training.personaplex.action_restore import restore_fit
from training.venus.opal_adapter import checkpoint_manifest
from training.venus.state import rng_state, exact_state


class SavedTransport:
    def __init__(self, state, corrupt=False):
        self.saved = copy.deepcopy(state)
        self.corrupt = corrupt

    def load_state_dict(self, state):
        self.saved = copy.deepcopy(state)
        if self.corrupt:
            self.saved['memory'] = b'changed'

    def state_dict(self):
        return self.saved

    def step(self, *args, **kwargs):
        raise AssertionError('No updates in a restore')


class ActionRestoreTest(unittest.TestCase):
    def fixture(self, mode='readout', corrupt=False):
        from evals.harness.test_action_inference import ActionInferenceTest
        _, identity, state = ActionInferenceTest().fixture(mode)
        state['rng'] = rng_state()
        state['diagnostics'] = dict(train_ce=None, dev_ce=None, last_review_step=0)
        for key in ('ledger', 'follows', 'next_analysis', 'decision_schedule', 'decisions', 'cost'):
            state['optimizer'][key] = []
        transport = SavedTransport(state['optimizer'], corrupt=corrupt)
        state['program'] = checkpoint_manifest(transport)
        head = model(5, mode, 'cpu')
        owned = [p for p in head.parameters() if p.requires_grad]
        runtime = SimpleNamespace(base=SimpleNamespace(param_groups=[dict(params=owned)]), optimizer=transport)
        runtime.restore_diagnostics = lambda value: setattr(runtime, 'diagnostics_state', copy.deepcopy(value))
        return state, head, runtime, identity

    def test_exact_readout_and_bias_restore_without_updates(self):
        for mode in ('readout', 'bias'):
            state, head, runtime, identity = self.fixture(mode)
            result = restore_fit(state, head, runtime, identity)
            self.assertTrue(exact_state(head.state_dict(), state['delta']))
            self.assertTrue(exact_state(rng_state(), state['rng']))
            self.assertEqual(result['new_optimizer_updates'], 0)
            self.assertFalse(result['native_free_running_qualified'])

    def test_wrong_parameter_owner_rejected_before_model_mutation(self):
        state, head, runtime, identity = self.fixture()
        before = copy.deepcopy(head.state_dict())
        runtime.base.param_groups[0]['params'] = [torch.nn.Parameter(torch.zeros(1))]
        with self.assertRaisesRegex(ValueError, 'exactly the selected'):
            restore_fit(state, head, runtime, identity)
        self.assertTrue(exact_state(before, head.state_dict()))

    def test_corrupt_complete_opal_state_is_not_accepted(self):
        state, head, runtime, identity = self.fixture(corrupt=True)
        with self.assertRaisesRegex(ValueError, 'OPAL state differs'):
            restore_fit(state, head, runtime, identity)

    def test_stale_program_version_is_rejected(self):
        state, head, runtime, identity = self.fixture()
        state['program']['program_version'] += 1
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            restore_fit(state, head, runtime, identity)

    def test_scalar_control_cannot_restore_with_trainable_hidden_weights(self):
        state, head, runtime, identity = self.fixture('bias')
        head.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'hidden weights frozen'):
            restore_fit(state, head, runtime, identity)
