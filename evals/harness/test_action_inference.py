import hashlib,unittest
import torch
from torch import nn
from evals.personaplex_adapters import AdapterLinear,adapter_parameters as old_adapters
from training.personaplex.action_inference import identity,install_fit,install_zero,adapter_parameters
from training.personaplex.native_action_readout import RetReadout

class ActionInferenceTest(unittest.TestCase):
    def fixture(self,mode='readout'):
        lm=nn.Module();lm.text_linear=AdapterLinear(nn.Linear(5,12,bias=False),rank=2)
        selection=dict(step=196,checkpoint_sha256='a'*64,trainer_source='b'*40);bound=identity(selection);files={'updater.py':'fixture-updater','analysis.py':'fixture-analysis'}
        state=dict(schema='dl-native-ret-fit-state-v1',step=128,writer_pid=123,rng={},diagnostics={},metadata=dict(identity=bound,mode=mode,input_features=5,backbone_updates=0,previous_saved_updates_replayed=0,trainable_parameters=6 if mode=='readout'else 1,recipe=dict(updates=128)),delta=dict(weight=torch.zeros(1,5),bias=torch.tensor([.2])),optimizer=dict(opal=1,update=128,version=0,files=files,updater={},user_optimizers=[],user_schedulers=[],observations={},memory=b''),program=dict(update=128,upstream_commit='8b16759a3e9930916ed582e28068267859f935f8',programs={k:hashlib.sha256(v.encode()).hexdigest()for k,v in files.items()}))
        return lm,bound,state
    def test_installed_residual_preserves_old_adapter_identity(self):
        lm,bound,state=self.fixture();x=torch.randn(1,8,5);before=lm.text_linear(x).detach();old={k:v.detach().clone()for k,v in old_adapters(lm).items()};proof=install_fit(lm,state,bound);after=lm.text_linear(x)
        self.assertIsInstance(lm.text_linear,RetReadout);self.assertTrue(torch.equal(before[...,:4],after[...,:4]));self.assertTrue(torch.equal(before[...,5:],after[...,5:]));self.assertFalse(torch.equal(before[...,4],after[...,4]));self.assertFalse(any(p.requires_grad for p in lm.parameters()))
        actual=adapter_parameters(lm);self.assertEqual(old.keys(),actual.keys());self.assertTrue(all(torch.equal(v,actual[k])for k,v in old.items()));self.assertFalse(proof['native_free_running_qualified'])
        with self.assertRaises(ValueError):install_fit(lm,state,bound)
    def test_invalid_identity_or_state_fails_before_freezing_or_replacing(self):
        for bad in ('identity','progress','nan','program'):
            lm,bound,state=self.fixture();projection=lm.text_linear;prior={n:p.requires_grad for n,p in lm.named_parameters()}
            if bad=='identity':state['metadata']['identity']=dict(bound,checkpoint_sha256='c'*64)
            if bad=='progress':state['step']=127
            if bad=='nan':state['delta']['bias'].fill_(float('nan'))
            if bad=='program':state['optimizer']['files']['updater.py']='changed'
            with self.assertRaises(ValueError):install_fit(lm,state,bound)
            self.assertIs(lm.text_linear,projection);self.assertEqual(prior,{n:p.requires_grad for n,p in lm.named_parameters()})
    def test_scalar_control_cannot_smuggle_hidden_weights(self):
        lm,bound,state=self.fixture('bias');state['delta']['weight'].fill_(.1)
        with self.assertRaises(ValueError):install_fit(lm,state,bound)

    def test_zero_head_is_distinct_from_trained_package(self):
        lm,bound,_=self.fixture();x=torch.randn(1,7,5);before=lm.text_linear(x).detach();proof=install_zero(lm,bound)
        self.assertTrue(torch.equal(before,lm.text_linear(x)));self.assertEqual(proof['new_readout_updates'],0);self.assertFalse(any(p.requires_grad for p in lm.parameters()))
