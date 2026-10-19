import copy
import unittest
from evals.context_warm_ordinary_retention import IDS,select,loss


class WarmOrdinaryRetentionTests(unittest.TestCase):
    def data(self):
        oldids=['teacher-recovery-train-%03d'%i for i in range(8,12)]+['other-%d'%i for i in range(21)]
        old=[(None,None,None,dict(id=i,kind='simple' if i in IDS else 'private',source_split='train')) for i in oldids]
        new=[(None,None,None,dict(id='teacher-native2-train-%03d'%i,kind='simple' if 8<=i<16 else 'private',source_split='train')) for i in range(24)]
        return old,new

    def test_all49_remain_and_exact10_predeclared_ordinary_anchors_selected(self):
        old,new=self.data();chosen=select(old,new)
        self.assertEqual([x[3]['id'] for x in chosen],list(IDS))
        self.assertNotIn('teacher-native2-train-014',[x[3]['id'] for x in chosen])
        self.assertNotIn('teacher-native2-train-015',[x[3]['id'] for x in chosen])
        with self.assertRaises(ValueError):select(old[:-1],new)
        with self.assertRaises(ValueError):select(old,new[:-1])

    def test_heldout_wrong_kind_or_missing_history_rejected(self):
        old,new=self.data()
        for field,value in [('source_split','validation'),('kind','private'),('id','native-fresh2-000')]:
            bad=copy.deepcopy(old);bad[0][3][field]=value
            with self.assertRaises(ValueError):select(bad,new)

    def test_exact_reference_has_zero_loss(self):
        import torch
        h=torch.zeros(1,3,2);z=torch.zeros(1,1,3,4);mask=torch.tensor([False,True,True])
        ref=(h[:,mask].clone(),z[0,0,mask].log_softmax(-1).clone())
        value,kl,mse=loss(h,z,mask,ref)
        self.assertEqual(float(value),0.);self.assertEqual(float(kl),0.);self.assertEqual(float(mse),0.)

    def test_regularizer_has_finite_gradients_only_at_approved_word_positions(self):
        import torch
        h=torch.ones(1,3,2,requires_grad=True);z=torch.zeros(1,1,3,4);z[0,0,1,0]=3.;z.requires_grad_();mask=torch.tensor([False,True,False])
        ref=(torch.zeros(1,1,2),torch.zeros(1,4).log_softmax(-1))
        value,_,_=loss(h,z,mask,ref);value.backward()
        self.assertGreater(float(value),0.);self.assertTrue(torch.isfinite(h.grad).all());self.assertTrue(torch.isfinite(z.grad).all())
        self.assertEqual(float(h.grad[:,0].abs().sum()),0.);self.assertEqual(float(z.grad[:,:,0].abs().sum()),0.)
        self.assertGreater(float(z.grad[:,:,1].abs().sum()),0.)


if __name__=='__main__':unittest.main()
