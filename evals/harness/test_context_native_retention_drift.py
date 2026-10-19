import copy
import unittest
from evals.context_native_retention_drift import validate_lineage,validate_inventory,differences


class RetentionDriftTests(unittest.TestCase):
    def test_wrong_parent_candidate_or_data_cannot_be_compared(self):
        parent=dict(optimizer_steps=520,data_sha256='data',frozen_hashes={'base':'hash'},voice_sha256='voice')
        child=dict(optimizer_steps=8,passed=True,additional_native_v2=dict(complete24_per8=True),
            warm_start=dict(checkpoint_sha256='parent'),data_sha256='data',frozen_hashes={'base':'hash'},voice_sha256='voice')
        validate_lineage(parent,child,'parent','data')
        for field,value in [('optimizer_steps',40),('passed',False),('data_sha256','changed'),
                            ('voice_sha256','changed'),('frozen_hashes',{'base':'changed'}),
                            ('additional_native_v2',{'complete24_per8':False}),('warm_start',{'checkpoint_sha256':'wrong'})]:
            changed=copy.deepcopy(child);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_lineage(parent,changed,'parent','data')

    def test_complete49_and_all4_references_required(self):
        old=[(None,None,None,dict(id='old%d'%i)) for i in range(25)]
        new=[(None,None,None,dict(id='teacher-native2-train-%03d'%i)) for i in range(24)]
        refs=[(None,None,dict(id='reference%d'%i)) for i in range(4)]
        validate_inventory(old,new,refs)
        for a,b,c in [(old[:-1],new,refs),(old,new[:-1],refs),(old,new,refs[:-1]),
                      (old[:-1]+[old[0]],new,refs),(old,new[:-1]+[new[0]],refs),(old,new,refs[:-1]+[refs[0]])]:
            with self.assertRaises(ValueError):validate_inventory(a,b,c)

    def test_zero_identity_and_excluded_frames_do_not_create_drift(self):
        import torch
        h=torch.zeros(1,4,2);z=torch.zeros(1,1,4,3);mask=torch.tensor([False,True,False,True])
        ch=h.clone();cz=z.clone();ch[:,0]=100.;cz[:,:,0]=100.
        r=differences(h,z,ch,cz,mask)
        self.assertEqual(r['warm_parent_kl'],0.);self.assertEqual(r['warm_parent_hidden_mse'],0.)
        self.assertEqual(r['text_top1_agreement'],1.);self.assertEqual(r['frames'],2)
        ch[:,1]=1.;cz[0,0,1,2]=5.
        r=differences(h,z,ch,cz,mask)
        self.assertGreater(r['warm_parent_kl'],0.);self.assertGreater(r['warm_parent_hidden_mse'],0.)
        self.assertEqual(r['text_top1_agreement'],.5)

    def test_nonfinite_comparison_is_rejected(self):
        import torch
        h=torch.zeros(1,1,2);z=torch.zeros(1,1,1,3);bad=z.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):differences(h,z,h,bad,torch.tensor([True]))


if __name__=='__main__':unittest.main()
