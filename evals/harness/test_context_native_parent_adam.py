"""Optimizer clocks/moments survive a new-stage rate change without copying weights."""
import copy
import unittest

class ParentAdamPort(unittest.TestCase):
    def fixture(self):
        import torch
        p=torch.nn.Parameter(torch.tensor([.2,-.3]));learned={'adapter':p};opt=torch.optim.AdamW([p],lr=1e-5,weight_decay=0.)
        for i in range(520):p.grad=torch.tensor([.1,-.2]);opt.step();opt.zero_grad(set_to_none=True)
        state=dict(completed_steps=520,parameter_names=['adapter'],adapters={'adapter':p.detach().clone()},optimizer=copy.deepcopy(opt.state_dict()))
        return torch,p,learned,opt,state
    def test_only_rate_changes_and_adam_clock_advances_to528(self):
        from evals.context_native_parent_adam import restore_moments
        torch,p,learned,old,state=self.fixture();new=torch.optim.AdamW([p],lr=2e-7,weight_decay=0.);before=p.detach().clone();rng=torch.get_rng_state().clone()
        a=restore_moments(state,learned,new,2e-7)
        self.assertTrue(torch.equal(p,before));self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(a['all_named_moments_and_steps_exact'])
        for i in range(8):p.grad=torch.tensor([-.3,.5]);new.step();new.zero_grad(set_to_none=True)
        self.assertEqual(float(new.state[p]['step']),528.);self.assertEqual(float(state['optimizer']['state'][0]['step']),520.)
    def test_wrong_name_step_moment_or_rate_is_rejected_before_import(self):
        from evals.context_native_parent_adam import restore_moments
        torch,p,learned,old,state=self.fixture()
        for field in ['names','clock','moment','rate']:
            s=copy.deepcopy(state);new=torch.optim.AdamW([p],lr=2e-7,weight_decay=0.)
            if field=='names':s['parameter_names']=['wrong']
            if field=='clock':s['optimizer']['state'][0]['step'].fill_(519)
            if field=='moment':s['optimizer']['state'][0]['exp_avg_sq'][0]=-1
            with self.subTest(field=field),self.assertRaises(ValueError):restore_moments(s,learned,new,1e-5 if field=='rate' else 2e-7)
            self.assertFalse(new.state)
    def test_nonempty_optimizer_is_never_overwritten(self):
        from evals.context_native_parent_adam import restore_moments
        torch,p,learned,old,state=self.fixture()
        with self.assertRaises(ValueError):restore_moments(state,learned,old,2e-7)
