"""Interrupted optimization must preserve moments, weights and all CPU RNG streams."""
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from evals.context_training_state import save, restore


class TrainingState(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127); random.seed(128); np.random.seed(129)

    def test_resume_matches_uninterrupted_stochastic_optimization(self):
        def setup():
            p = torch.nn.Parameter(torch.randn(5,3)); opt = torch.optim.AdamW([p],lr=.01)
            return {'adapter':p},opt
        def step(params,opt):
            x = torch.randn(4,5)*random.random(); target = torch.tensor(np.random.randn(4,3),dtype=torch.float32)
            opt.zero_grad(); loss = (x@params['adapter']-target).square().mean();loss.backward();opt.step()
            return float(loss.detach())
        params,opt = setup(); meta = {'model':'pinned-test','data':'approved','recipe':{'lr':.01}}
        for _ in range(3):step(params,opt)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'training-state.pt';save(path,params,opt,3,meta)
            expected_losses = [step(params,opt) for _ in range(5)]; expected = params['adapter'].detach().clone()
            # Cold construction consumes Torch RNG; the restore must undo that.
            other,newopt = setup(); random.random();np.random.randn(23)
            state = restore(path,other,newopt,meta)
            self.assertEqual(state['completed_steps'],3)
            self.assertEqual(expected_losses,[step(other,newopt) for _ in range(5)])
            self.assertTrue(torch.equal(expected,other['adapter']))

    def test_changed_data_rejected_before_any_parameter_mutation(self):
        p = torch.nn.Parameter(torch.ones(2)); params = {'adapter':p}; opt = torch.optim.AdamW([p])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'state.pt';save(path,params,opt,1,{'data':'approved-A'})
            with torch.no_grad():p.fill_(7.)
            with self.assertRaisesRegex(RuntimeError,'identity'):
                restore(path,params,opt,{'data':'different-B'})
            self.assertTrue(torch.equal(p,torch.full((2,),7.)))


if __name__=='__main__':unittest.main()
