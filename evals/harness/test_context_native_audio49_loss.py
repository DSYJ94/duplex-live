import unittest
from types import SimpleNamespace
import torch
from evals.context_native_audio49_loss import loss

class AuxiliaryAudioOwnership(unittest.TestCase):
    def example(self):
        codes=torch.arange(17*3,dtype=torch.int64).reshape(1,17,3)%7
        logits=torch.zeros(1,16,3,7,requires_grad=True)
        output=SimpleNamespace(logits=logits,mask=torch.ones(1,16,3,dtype=torch.bool))
        return output,codes,torch.tensor([[False,True,False]])
    def test_only_approved_assistant_positions_receive_gradients(self):
        out,codes,mask=self.example();v,d=loss(out,codes,mask);v.backward()
        self.assertEqual(d['assistant_audio_targets'],[1]*8)
        self.assertEqual(int(out.logits.grad[:,8:].count_nonzero()),0)
        self.assertEqual(int(out.logits.grad[:,:,:1].count_nonzero()),0)
        self.assertEqual(int(out.logits.grad[:,:,2:].count_nonzero()),0)
        self.assertGreater(int(out.logits.grad[:,:8,1].count_nonzero()),0)
    def test_user_labels_never_become_targets(self):
        out,codes,mask=self.example();first=loss(out,codes,mask)[0];changed=codes.clone();changed[:,9:]=123456
        self.assertTrue(torch.equal(first,loss(out,changed,mask)[0]))
    def test_labels_are_already_aligned_and_are_not_shifted(self):
        out,codes,mask=self.example()
        with torch.no_grad():
            for q in range(8):out.logits[0,q,1,codes[0,q+1,1]]=25.
        self.assertLess(float(loss(out,codes,mask)[0]),1e-5)
    def test_every_assistant_head_needs_valid_targets(self):
        out,codes,mask=self.example();out.mask[:,7]=False
        with self.assertRaises(ValueError):loss(out,codes,mask)
