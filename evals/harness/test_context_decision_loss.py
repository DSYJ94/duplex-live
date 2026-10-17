import unittest
import torch
from evals.context_decision_loss import weights,policy,validate


class DecisionLossTest(unittest.TestCase):
    def test_default_preserves_existing_loss_weights_exactly(self):
        y=torch.tensor([0,3,4,5,3,6],dtype=torch.int64)
        self.assertTrue(torch.equal(weights(y),torch.where(y==3,.5,1.)))
        self.assertIsNone(policy(1.,[False,True]))

    def test_contextual_first_word_gradient_is_emphasized_not_later_words(self):
        y=torch.tensor([5,3,6,3],dtype=torch.int64);z=torch.zeros(4,10,requires_grad=True)
        w=weights(y,16.,True);loss=(torch.nn.functional.cross_entropy(z,y,reduction='none')*w).sum()/w.sum()
        loss.backward()
        self.assertAlmostEqual(float(z.grad[0,5]/z.grad[2,6]),16.,places=5)
        self.assertAlmostEqual(float(z.grad[1,3]/z.grad[2,6]),.5,places=5)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_private_requests_and_control_windows_are_unchanged(self):
        y=torch.tensor([3,0,4,7],dtype=torch.int64)
        self.assertTrue(torch.equal(weights(y,16.,False),weights(y)))
        p=policy(16.,[False,True,False]);self.assertEqual(p['component_indices'],[1])
        self.assertEqual(p['past_loss_targets']+p['audio_loss_targets']+p['user_loss_targets'],0)

    def test_duplicated_control_or_unmasked_past_cannot_acquire_decision_weight(self):
        for first in (0,3,4):
            with self.subTest(first=first),self.assertRaises(RuntimeError):
                weights(torch.tensor([first,5],dtype=torch.int64),16.,True)
        with self.assertRaises(RuntimeError):weights(torch.tensor([-1,5],dtype=torch.int64),16.,False)
        with self.assertRaises(RuntimeError):policy(16.,[False,False])

    def test_invalid_weight_is_rejected(self):
        for v in (float('nan'),float('inf'),0.,33.):
            with self.subTest(value=v),self.assertRaises(ValueError):validate(v)


if __name__=='__main__':unittest.main()
