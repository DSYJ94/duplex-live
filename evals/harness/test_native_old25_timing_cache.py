import copy
import unittest
import torch
from evals.native_old25_timing_cache import validate_unchanged_payload


class CacheTimingInvariants(unittest.TestCase):
    def sample(self):
        return dict(codes=torch.arange(17*20).reshape(1,17,20),
                    reference=torch.zeros(1,20,4), markers=[2],
                    arrivals=[dict(ready_at_s=1.4, text='Checked value.', rows=torch.ones(2,4))])

    def test_only_text_lane_can_change(self):
        old=self.sample();fresh=copy.deepcopy(old)
        fresh['codes'][0,0]=torch.roll(fresh['codes'][0,0],1)
        validate_unchanged_payload(old,fresh)

    def test_rejects_any_audio_reference_receipt_shape_or_type_change(self):
        old=self.sample()
        for change in ('audio','reference','arrival','marker','frames','dtype'):
            fresh=copy.deepcopy(old)
            if change=='audio':fresh['codes'][0,8,4]+=1
            elif change=='reference':fresh['reference'][0,5,0]=1
            elif change=='arrival':fresh['arrivals'][0]['ready_at_s']=1.5
            elif change=='marker':fresh['markers']=[3]
            elif change=='frames':fresh['codes']=fresh['codes'][:,:,:19]
            else:fresh['codes']=fresh['codes'].float()
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                validate_unchanged_payload(old,fresh)


if __name__=='__main__':unittest.main()
