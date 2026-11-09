import copy
from contextlib import contextmanager
import math
from types import SimpleNamespace
import unittest
import torch
from torch import nn

from training.personaplex.matched_losses import ret_population,ret_loss,recorded_text,RET_GROUPS
from training.personaplex.matched_schedule import SIZES,epoch


class StreamingFixture(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(1,4);self.head=nn.Linear(4,8)

    @contextmanager
    def streaming(self,batch):
        self.state=SimpleNamespace(kv_cache=SimpleNamespace(cache=torch.zeros(1,1,1,1,4)))
        yield

    def get_streaming_state(self):return {'temporal':self.state}

    def forward_codes(self,x):
        hidden=self.input(x[:,0,:,None].float())+self.state.kv_cache.cache.reshape(1,1,4)
        self.state.kv_cache.cache=hidden[:,-1].reshape(1,1,1,1,4)
        return hidden,self.head(hidden)[:,None]


class NativeLossSemantics(unittest.TestCase):
    def test_recorded_loss_without_auxiliary_matches_original_values_and_gradients(self):
        from evals.context_native_stream_gradient import capture
        torch.manual_seed(17);a=StreamingFixture();b=copy.deepcopy(a)
        inputs=torch.arange(5).reshape(1,1,5).expand(1,17,5)
        mask=torch.tensor([False,False,True,True,True]);targets=torch.tensor([0,0,3,5,4])
        calls=[dict(start=0,frames=2,phase='prefix')]+[dict(start=i,frames=1,phase='live') for i in range(2,5)]
        old=capture(a,inputs,mask,targets,calls,backward=True,weight=.7)
        new=recorded_text(b,inputs,mask,targets,calls,.7)
        self.assertEqual(old['balanced_spoken_clock_ce'],new['text_ce'])
        self.assertEqual(new['ret_raw'],0.)
        for left,right in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
            torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_population_excludes_prefix_and_never_accepts_evaluation_fold(self):
        row=dict(fold='TRAIN',mask=torch.tensor([False,True,True]),targets=torch.tensor([4,4,3]))
        groups={g:[copy.deepcopy(row)] for g in RET_GROUPS};p=ret_population(groups)
        self.assertEqual((p['positive'],p['negative']),(5,5))
        self.assertAlmostEqual(p['positive']*p['positive_position_weight'],4.)
        self.assertAlmostEqual(p['negative']*p['negative_position_weight'],4.)
        groups['ordinary48'][0]['fold']='DEV'
        with self.assertRaisesRegex(ValueError,'evaluation'):ret_population(groups)

    def test_binary_ret_ce_has_correct_probability_and_finite_extreme_gradients(self):
        pop=dict(positive_position_weight=4.,negative_position_weight=4.)
        z=torch.zeros(2,6,requires_grad=True);labels=torch.tensor([4,3]);value=ret_loss(z,labels,pop)
        self.assertAlmostEqual(float(value.detach()),4*math.log(6)+4*math.log(6/5),places=5)
        z=torch.zeros(2,6,requires_grad=True)
        with torch.no_grad():z[0,4]=1000;z[1,4]=-1000
        value=ret_loss(z,torch.tensor([3,4]),pop);value.backward()
        self.assertTrue(torch.isfinite(value) and torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad[0,4]),3.99);self.assertLess(float(z.grad[1,4]),-3.99)

    def test_inverse_visits_keeps_total_ret_mass_eight_for_every_complete_epoch(self):
        groups={g:[dict(id=f'{g}:{i}',fold='TRAIN',kind='rare' if i==0 else 'common',
                    mask=torch.tensor([True,True]),targets=torch.tensor([4,3])) for i in range(n)] for g,n in SIZES.items()}
        p=ret_population(groups)
        self.assertEqual(p['positive'],197)
        for e in range(4):
            mass=0.
            for step in epoch(groups,e):
                for row in step:
                    if row['group'] in RET_GROUPS:
                        mass+=(p['positive_position_weight']+p['negative_position_weight'])/row['visits_this_epoch']
            self.assertAlmostEqual(mass,8.)


if __name__=='__main__':unittest.main()
