import unittest
from evals.context_retention_teacher import validate_weight


class RetentionWeightTests(unittest.TestCase):
    def test_explicit_bounded_weight(self):
        for value in (0.,1.,2.):validate_weight(value)
        for value in (-1.,float('nan'),float('inf'),2.01):
            with self.assertRaises(ValueError):validate_weight(value)


class TeacherSignalTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch

    def test_wrong_base_has_zero_kl_gradient_but_teacher_corrects_it(self):
        from evals.context_retention_teacher import loss
        t=self.t;z=t.tensor([[8.,-8.,0.,0.,0.,0.]],requires_grad=True)
        reference=z.detach().softmax(-1)
        kl=t.nn.functional.kl_div(z.log_softmax(-1),reference,reduction='batchmean')
        kl.backward();self.assertLess(float(z.grad.abs().max()),1e-6)
        z.grad=None;ce=loss(z,t.tensor([1]),t.tensor([True]));ce.backward()
        self.assertLess(float(z.grad[0,1]),-.9);self.assertGreater(float(z.grad[0,0]),.9)

    def test_excluded_prefix_and_span_logits_cannot_affect_teacher_loss(self):
        from evals.context_retention_teacher import loss
        t=self.t;z=t.zeros(4,8,requires_grad=True);y=t.tensor([-1,5,-1,6]);m=t.tensor([False,True,False,True])
        expected=loss(z,y,m);changed=z.detach().clone();changed[~m]=float('nan')
        self.assertTrue(t.equal(expected.detach(),loss(changed,y,m)))
        expected.backward();self.assertEqual(int(t.count_nonzero(z.grad[~m])),0)
        with self.assertRaises(RuntimeError):loss(z,t.tensor([-1,8,-1,6]),m)

    def scene(self):
        t=self.t;codes=t.zeros(17,6,dtype=t.int64);codes[0]=t.tensor([-1,5,-1,6,3,0])
        r=dict(id='teacher-train-fixture',kind='simple',source_model='gpt-live-1',source_split='train',
            request_policy='native-prelead-epad',frames=6,real_frames=4,prefix_frames=1,
            real_to_virtual=[1,3,4,5],cache='fixture',cache_sha256='a',codes_sha256='b',wav_sha256='c',
            sidecar_sha256='d',ingestion_sha256='e')
        clean=codes[:,[1,3,4,5]].clone()
        return dict(record=r,codes=codes,clean_codes=clean,text_mask=t.tensor([False,True,False,True,True,True]))

    def test_only_complete_measured_teacher_text_is_eligible(self):
        from evals.context_retention_teacher import scope
        e=self.scene();_,_,a=scope(e);self.assertEqual(a['text_targets'],4);self.assertEqual(a['spoken_targets'],2)
        self.assertEqual(a['user_loss_targets'],0);self.assertEqual(a['span_loss_targets'],0)
        for key,value in [('source_split','validation'),('source_model','student')]:
            e=self.scene();e['record'][key]=value
            with self.assertRaises(RuntimeError):scope(e)
        e=self.scene();e['text_mask'][2]=True
        with self.assertRaises(RuntimeError):scope(e)
        e=self.scene();e['codes'][0,3]=7
        with self.assertRaises(RuntimeError):scope(e)


if __name__=='__main__':unittest.main()
