"""Use actual pinned undo-delay masks; never accept an arbitrary passing subset."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from evals import native_control_numerical as control

VENDOR=Path('runs/cloud-export-20261016/tree/vendor/context-spanning-79df302fa2820cae10c61bb2b7f302c52edb6ed2')


class ControlNumerical(unittest.TestCase):
    def test_all_outputs_must_have_same_shape_and_be_finite(self):
        x=torch.ones(1,7,3)
        self.assertEqual(control.differences(x,x)['maximum_absolute_error'],0.)
        for y in (x.reshape(1,3,7),torch.full_like(x,float('nan'))):
            with self.assertRaises(ValueError):control.differences(x,y)

    @unittest.skipUnless(VENDOR.exists(),'Pinned native source needed')
    def test_actual_native_un_delay_masks_keep_every_available_assistant_head(self):
        sys.path.insert(0,str(VENDOR.resolve()))
        from contextspan.moshi.models.lm import _undelay_sequence
        _,audio=_undelay_sequence([0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1],torch.zeros(1,16,7,4))
        _,text=_undelay_sequence([0],torch.zeros(1,1,7,4))
        output=SimpleNamespace(mask=audio,text_mask=text)
        codes=torch.ones(1,17,7,dtype=torch.int64);codes[0,0]=torch.tensor([3,3,3,7,0,3,3])
        mask=torch.tensor([False,False,True,True,True,True,True])
        positions,counts=control.target_positions(mask,codes,output)
        self.assertEqual(positions,[2,3,5]);self.assertEqual(counts,[5,4,4,4,4,4,4,4])
        # Keep the target count while selecting the wrong frame: still reject.
        output.mask=audio.clone();output.mask[0,1,2]=False;output.mask[0,1,-1]=True
        with self.assertRaisesRegex(ValueError,'no arbitrary target subset'):control.target_positions(mask,codes,output)


if __name__=='__main__':unittest.main()
