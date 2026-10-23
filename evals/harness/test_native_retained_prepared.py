import copy
import unittest
from types import SimpleNamespace
import torch

from evals.context_native_audio49_data import digest
from evals.harness.test_context_native_audio49_data import Complete49AudioOwnership
from evals.native_retained_prepared import validate_batches
from evals.context_native_audio49_training import validate_configuration


class Prepared49(unittest.TestCase):
    def fixture(self):
        rows=Complete49AudioOwnership().rows();batches=[]
        for row in rows:
            codes=torch.arange(17*8,dtype=torch.int64).reshape(1,17,8)
            tm=torch.ones((1,8),dtype=torch.bool);am=tm.clone();inputs=codes.clone()
            targets=codes[:,1:9].transpose(1,2)[am]
            for tensor,key in ((codes,'codes_sha256'),(tm,'text_mask_sha256'),(am,'audio_mask_sha256'),
                               (inputs,'native_input_sha256'),(targets,'assistant_targets_sha256')):
                row[key]=digest(tensor)
            batches.append(dict(codes=codes,text_mask=tm,audio_mask=am,delayed_inputs=inputs,record=row))
        return dict(cases=rows),batches

    def test_every_serialized_tensor_and_teacher_is_checked(self):
        data,batches=self.fixture();self.assertTrue(validate_batches(data,batches))
        for key in ('codes','text_mask','audio_mask','delayed_inputs'):
            changed=copy.deepcopy(batches);tensor=changed[28][key]
            tensor.flatten()[0]=not bool(tensor.flatten()[0]) if tensor.dtype==torch.bool else 123
            with self.subTest(key=key),self.assertRaises(ValueError):validate_batches(data,changed)
        with self.assertRaises(ValueError):validate_batches(data,batches[:-1])
        changed=copy.deepcopy(batches);changed[0],changed[1]=changed[1],changed[0]
        with self.assertRaises(ValueError):validate_batches(data,changed)

    def test_changed_target_annotation_is_rejected_independently(self):
        data,batches=self.fixture();data['cases'][0]['assistant_targets_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'assistant_targets'):validate_batches(data,batches)

    def test_cache_never_enables_legacy_or_partial_training_modes(self):
        for options in (dict(native_audio49_prepared='/cache'),
                        dict(native_audio49_prepared='/cache',native_audio49_source_inventory='/inventory'),
                        dict(native_audio49_prepared='/cache',native_audio49_source_inventory='/inventory',
                             control40_cache='/control',native_audio49_complete_batch=True)):
            with self.subTest(options=options),self.assertRaisesRegex(ValueError,'Prepared49'):
                validate_configuration(SimpleNamespace(**options))


if __name__=='__main__':unittest.main()
