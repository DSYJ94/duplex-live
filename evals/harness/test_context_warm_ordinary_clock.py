import copy
import unittest
from evals.context_warm_ordinary_clock import ACTUAL,CLOCK_IDS,select
from evals.harness import test_context_warm_ordinary_retention as fixtures


class OrdinaryClockTests(unittest.TestCase):
    def data(self):
        old,new=fixtures.WarmOrdinaryRetentionTests().data()
        old=[(object(),object(),object(),r) for _,_,_,r in old]
        new=[(object(),object(),object(),r) for _,_,_,r in new]
        batches=[(object(),object(),object()) for _ in range(28)]
        # Approved nonordinary actual records may omit target_kind.
        audits=[{} for _ in range(28)]
        for i,(job,scenario,manifest) in enumerate(ACTUAL):
            audits[i]=dict(target_kind='ordinary_answer',component='/projects/u6si/duplex-live/runs/context-training-proof-'+job,
                scenario=scenario,manifest_sha256=manifest,passed=True,past_loss_targets=0,audio_loss_targets=0)
        return old,new,batches,audits

    def test_all49_and28_kept_exact14_masks_and_inputs_unchanged(self):
        old,new,batches,audits=self.data();rows=select(old,new,batches,audits)
        self.assertEqual([r[3]['id'] for r in rows],list(CLOCK_IDS))
        self.assertEqual(len(rows),14)
        for original,row in zip(batches[:4],rows[10:]):
            for a,b in zip(original,row[:3]):self.assertIs(a,b)
        originals={r[3]['id']:r for r in old+new}
        for row in rows[:10]:
            original=originals[row[3]['id'].removeprefix('clock-')]
            for a,b in zip(original[:3],row[:3]):self.assertIs(a,b)

    def test_no_subset_or_extra_ordinary_capture(self):
        old,new,batches,audits=self.data()
        with self.assertRaises(ValueError):select(old,new,batches[:-1],audits[:-1])
        with self.assertRaises(ValueError):select(old[:-1],new,batches,audits)
        audits[4]=copy.deepcopy(audits[0])
        with self.assertRaises(ValueError):select(old,new,batches,audits)

    def test_changed_capture_split_or_prefix_audio_labels_rejected(self):
        old,new,batches,audits=self.data()
        for key,value in [('component','heldout-new-evaluation'),('manifest_sha256','changed'),
                ('scenario','recovery-validation-000'),('passed',False),('past_loss_targets',1),('audio_loss_targets',1)]:
            bad=copy.deepcopy(audits);bad[0][key]=value
            with self.assertRaises(ValueError):select(old,new,batches,bad)

    def test_regularizer_includes_original_pad_and_word_mask_positions(self):
        import torch
        from evals.context_warm_ordinary_retention import loss
        hidden=torch.ones(1,3,2,requires_grad=True);z=torch.zeros(1,1,3,4,requires_grad=True)
        mask=torch.tensor([False,True,True]);targets=torch.tensor([0,3,2])
        reference=(torch.zeros(1,2,2),torch.zeros(2,4).log_softmax(-1))
        value,_,_=loss(hidden,z,mask,reference);value.backward()
        self.assertEqual(float(hidden.grad[:,0].abs().sum()),0.)
        self.assertGreater(float(hidden.grad[:,targets==3].abs().sum()),0.)
        self.assertGreater(float(hidden.grad[:,targets==2].abs().sum()),0.)


if __name__=='__main__':unittest.main()
