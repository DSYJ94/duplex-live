import unittest
import torch
from torch import nn
from training.personaplex.native_action_readout import install,RetReadout


class ActionReadoutTest(unittest.TestCase):
    def model(self):
        m=nn.Module();m.text_linear=nn.Linear(5,12,bias=False);return m

    def test_exact_zero_insertion_and_only_ret_can_change(self):
        m=self.model();x=torch.randn(2,7,5);before=m.text_linear(x).detach();readout=install(m,base_identity='a'*64)
        self.assertTrue(torch.equal(before,readout(x)))
        with torch.no_grad():readout.delta.weight.fill_(.2);readout.delta.bias.fill_(.5)
        after=readout(x);self.assertTrue(torch.equal(before[...,:4],after[...,:4]));self.assertTrue(torch.equal(before[...,5:],after[...,5:]));self.assertFalse(torch.equal(before[...,4],after[...,4]))
        self.assertEqual(sum(p.numel()for p in m.parameters()if p.requires_grad),6)

    def test_backward_and_fresh_restore_preserve_original_weights(self):
        m=self.model();x=torch.randn(8,5);readout=install(m,base_identity='a'*64);original=readout.original.weight.detach().clone();opt=torch.optim.AdamW(readout.delta.parameters(),lr=.01)
        readout(x)[...,4].sum().backward();self.assertIsNone(readout.original.weight.grad)
        opt.step();self.assertTrue(torch.equal(original,readout.original.weight));saved=readout.learned_state()
        other=RetReadout(readout.original,base_identity='a'*64);other.restore(saved);self.assertTrue(torch.equal(other(x),readout(x)))
        with self.assertRaises(ValueError):other.restore(dict(saved,token=5))
        with self.assertRaises(ValueError):other.restore(dict(saved,base_identity='b'*64))
        with self.assertRaises(ValueError):install(m,base_identity='a'*64)
