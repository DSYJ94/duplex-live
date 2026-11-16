import unittest
import torch
from torch import nn
from training.personaplex.action_features import selected,binary_logits,balanced_loss
from training.personaplex.native_action_readout import RetReadout

class ActionFeaturesTest(unittest.TestCase):
    def test_cached_native_binary_probability_and_gradients_match_actual_projection(self):
        torch.manual_seed(42);original=nn.Linear(7,13,bias=False,dtype=torch.bfloat16);h=torch.randn(1,9,7,dtype=torch.bfloat16);mask=torch.tensor([True,False,True,True,False,True,True,False,True]);targets=torch.tensor([4,0,8,4,0,7,9,0,2]);base=original(h).detach()[:,None];cache=selected(h,base,mask,targets);head=RetReadout(original,base_identity='a'*64)
        with torch.no_grad():head.delta.weight.fill_(.03);head.delta.bias.fill_(.11)
        native=head(h)[0,mask];actual=native[:,4].float()-torch.cat((native[:,:4],native[:,5:]),-1).float().logsumexp(-1)
        cached=binary_logits(cache,head.delta);self.assertTrue(torch.equal(actual,cached))
        loss=balanced_loss(cached,cache['positive'],2,4);loss.backward();g=head.delta.weight.grad.clone();head.zero_grad(set_to_none=True)
        balanced_loss(actual,cache['positive'],2,4).backward();self.assertTrue(torch.equal(g,head.delta.weight.grad));self.assertIsNone(original.weight.grad)
    def test_masks_and_population_are_not_silently_changed(self):
        h=torch.ones(1,3,2);z=torch.ones(1,1,3,8);mask=torch.ones(3,dtype=torch.bool);target=torch.tensor([4,0,1]);cache=selected(h,z,mask,target)
        self.assertEqual(int(cache['positive'].sum()),1)
        with self.assertRaises(ValueError):selected(h,z,mask.long(),target)
        with self.assertRaises(ValueError):balanced_loss(torch.ones(3),cache['positive'],0,2)

    def test_fit_rejects_dev_or_incomplete_population_and_bias_has_one_owner(self):
        from training.personaplex.action_fit import model,validate
        h=torch.ones(1,3,5);z=torch.ones(1,1,3,8);mask=torch.ones(3,dtype=torch.bool);target=torch.tensor([4,0,1]);row=dict(group='actual',id='fixture',fold='TRAIN',**selected(h,z,mask,target));cache=dict(schema='dl-native-ret-TRAIN-features-v1',optimizer_updates=0,native_backbone_updated=False,population=dict(positive=1,negative=2),rows=[row])
        self.assertEqual(validate(cache),5)
        with self.assertRaises(ValueError):validate(dict(cache,rows=[dict(row,fold='DEV')]))
        with self.assertRaises(ValueError):validate(dict(cache,population=dict(positive=1,negative=1)))
        bias=model(5,'bias','cpu');self.assertEqual(sum(p.numel()for p in bias.parameters()if p.requires_grad),1)
        balanced_loss(binary_logits(row,bias),row['positive'],1,2).backward();self.assertIsNone(bias.weight.grad);self.assertIsNotNone(bias.bias.grad)
