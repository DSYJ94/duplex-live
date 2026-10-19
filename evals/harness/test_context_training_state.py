"""Interrupted optimization must preserve moments, weights and all CPU RNG streams."""
import copy
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from evals.context_training_state import save, restore, _source_metadata_matches


def fixture_transition():
    return dict(schema='dl-native-source-transition-v1',
        from_repo_commit='3061dcba1de571d2dad58d466f7c3f3ca43e41ec', to_repo_commit='a'*40,
        training_state_sha256='3303bf1624cf3c36ff071e9fb69a239b3b93fda6a0d914db67848b7543cad675',
        authorization_sha256='b'*64, review_sha256='c'*64,
        starting_optimizer_steps=8, additional_optimizer_updates=32, final_optimizer_steps=40)


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

    def test_source_transition_changes_only_commit_and_adds_exact_provenance(self):
        t = fixture_transition()
        old = dict(repo_commit=t['from_repo_commit'], data='approved', voice='fixed',
            recipe={'lr':2e-6,'kl':1.,'hidden':10.,'word':64.,'clock':64.},
            frozen_hashes={'base':'frozen'}, targets=['all28','old25','new24'])
        new = dict(old, repo_commit=t['to_repo_commit'], source_transition=t)
        self.assertTrue(_source_metadata_matches(old,new,t))
        self.assertFalse(_source_metadata_matches(old,new,None))
        for key, value in [('data','changed'),('voice','other'),('recipe',{'lr':1e-5}),
                ('frozen_hashes',{}),('targets',['subset']),('source_transition',{})]:
            bad=copy.deepcopy(new);bad[key]=value
            with self.subTest(key=key):self.assertFalse(_source_metadata_matches(old,bad,t))
        for key, value in [('from_repo_commit','d'*40),('to_repo_commit','e'*40),
                ('authorization_sha256','f'*64),('review_sha256','f'*64),
                ('starting_optimizer_steps',7),('additional_optimizer_updates',64),
                ('final_optimizer_steps',72),('training_state_sha256','f'*64)]:
            bad=copy.deepcopy(t);bad[key]=value
            with self.subTest(key=key):self.assertFalse(_source_metadata_matches(old,new,bad))

    def test_fixture_source_migration_and_new_strict_reload_preserve_adam_and_rng(self):
        # SHA pin alone is mocked for this toy state, never for production files.
        t=fixture_transition();old=dict(repo_commit=t['from_repo_commit'],data='fixture',recipe={'lr':.01})
        new=dict(old,repo_commit=t['to_repo_commit'],source_transition=t)
        def setup():
            p=torch.nn.Parameter(torch.randn(5,3));return {'adapter':p},torch.optim.AdamW([p],lr=.01)
        def step(params,opt):
            x=torch.randn(4,5)*random.random();y=torch.tensor(np.random.randn(4,3),dtype=torch.float32)
            opt.zero_grad();loss=(x@params['adapter']-y).square().mean();loss.backward();opt.step()
            return float(loss.detach())
        params,opt=setup()
        for _ in range(8):step(params,opt)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'old.pt';save(path,params,opt,8,old)
            expected=[step(params,opt) for _ in range(4)];weight=params['adapter'].detach().clone()
            other,newopt=setup()
            with self.assertRaisesRegex(RuntimeError,'identity'):restore(path,other,newopt,new)
            with self.assertRaisesRegex(RuntimeError,'identity'):
                restore(path,other,newopt,new,source_transition=t)
            with patch('evals.moshi_model_download.sha256',return_value=t['training_state_sha256']):
                restore(path,other,newopt,new,source_transition=t)
            actual=[step(other,newopt) for _ in range(2)]
            migrated=Path(d)/'new.pt';save(migrated,other,newopt,10,new)
            third,thirdopt=setup();restore(migrated,third,thirdopt,new)
            actual += [step(third,thirdopt) for _ in range(2)]
            self.assertEqual(actual,expected)
            self.assertTrue(torch.equal(third['adapter'],weight))


if __name__=='__main__':unittest.main()
