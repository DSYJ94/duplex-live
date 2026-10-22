"""Native control input contracts; no test qualifies actual CUDA Engine execution."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from evals import native_control_engine as control

VENDOR=Path('runs/cloud-export-20261016/tree/vendor/context-spanning-79df302fa2820cae10c61bb2b7f302c52edb6ed2')


def source():
    clean=torch.ones(17,5,dtype=torch.int64);clean[0]=torch.tensor([3,6,0,3,3]);clean[9:]=33
    prefix=torch.full((17,2),9,dtype=torch.int64);prefix[0]=3
    return clean,prefix


class NativeControlScope(unittest.TestCase):
    def test_streamed_user_changes_neither_teacher_audio_nor_complete_clock_mask(self):
        clean,prefix=source();before=clean.clone();user=torch.full((1,8,5),44,dtype=torch.int64)
        codes,mask=control.assemble(clean,prefix,user)
        self.assertTrue(torch.equal(clean,before))
        self.assertTrue(torch.equal(codes[:,:2],prefix))
        self.assertTrue(torch.equal(codes[:9,2:],clean[:9]))
        self.assertTrue(torch.all(codes[9:,2:]==44))
        self.assertEqual(mask.tolist(),[False,False,True,True,True,True,True])
        for bad in (user[:,:,:-1],user.float(),torch.full_like(user,2048)):
            with self.assertRaises(ValueError):control.assemble(clean,prefix,bad)
        clean[0,2]=4
        with self.assertRaises(ValueError):control.assemble(clean,prefix,user)

    def test_unapproved_full_cache_stops_before_engine_or_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            args=['audit','--cache','cache','--out',str(out),'--assets','assets','--base','base','--proof','proof']
            with patch.object(sys,'argv',args),patch.object(control,'verify_cache',side_effect=ValueError('Whole40 failed')):
                with self.assertRaisesRegex(ValueError,'Whole40 failed'):control.main()
            self.assertFalse(out.exists())


@unittest.skipUnless(VENDOR.exists(),'Pinned native vendor source needed for actual delay arithmetic')
class NativeControlDelay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(VENDOR.resolve()))
        from contextspan.moshi.models.lm import _delay_sequence
        cls.native_delay=staticmethod(_delay_sequence)

    def fixture(self):
        clean,prefix=source();user=torch.full((1,8,5),44,dtype=torch.int64)
        codes,mask=control.assemble(clean,prefix,user)
        initial=torch.full((1,17,1),2048,dtype=torch.int64);initial[:,0]=32000
        original,_=control.assemble(clean,prefix)
        saved=dict(inputs=control.delayed(codes,self.native_delay,initial),
                   offline_inputs=control.delayed(original,self.native_delay,initial),codes=codes,mask=mask,targets=codes[0],
                   streamed_user_codes=user,provided=torch.ones(1,17,5,dtype=torch.bool),current_targets=clean[None])
        calls=[dict(start=0,frames=2,phase='prefix',route='direct')]+[
            dict(start=2+i,frames=1,phase='live',route='graph') for i in range(5)]
        return clean,prefix,saved,calls,initial

    def test_pinned_native_delays_and_bootstrap_keep_all_real_frames(self):
        clean,prefix,saved,calls,initial=self.fixture()
        self.assertEqual(saved['inputs'].shape,(1,17,7))
        self.assertTrue(torch.equal(saved['inputs'][:,:,:1],initial))
        self.assertEqual(int(saved['inputs'][0,2,1]),2048)  # Delayed acoustic row.
        self.assertEqual(int(saved['inputs'][0,1,1]),9)     # Delay-zero semantic row.
        result=control.validate_trace(clean,prefix,saved,calls,self.native_delay,initial)
        self.assertEqual(result['text_targets'],5)
        self.assertEqual(result['prefix_text_loss_targets'],0)
        self.assertEqual(result['assistant_acoustic_source_frames_per_head'],5)

    def test_changed_teacher_target_or_prefix_ownership_is_rejected(self):
        clean,prefix,saved,calls,initial=self.fixture()
        for field in ('inputs','mask','provided','targets'):
            changed=copy.deepcopy(saved)
            if field=='inputs':changed[field][0,1,-1]+=1
            elif field=='mask':changed[field][0]=True
            elif field=='provided':changed[field][0,0,0]=False
            else:changed[field][-1]+=1
            with self.assertRaises(ValueError):control.validate_trace(clean,prefix,changed,calls,self.native_delay,initial)
        with self.assertRaises(ValueError):control.validate_trace(clean,prefix,saved,calls[:-1],self.native_delay,initial)


if __name__=='__main__':unittest.main()
