import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import torch
from training.personaplex.native_optimizer_state import fingerprint,load_exact
from training.personaplex.distributed_state import load_checkpoint


class NativeOptimizerStateTest(unittest.TestCase):
    def optimizer(self):
        parameter=torch.nn.Parameter(torch.ones(3))
        optimizer=torch.optim.AdamW([parameter],lr=.001)
        parameter.square().sum().backward();optimizer.step();optimizer.zero_grad()
        return optimizer

    def test_complete_cpu_optimizer_restore_has_independent_exact_receipt(self):
        source=self.optimizer();state=copy.deepcopy(source.state_dict())
        target=self.optimizer();receipt=load_exact(target,state)
        self.assertTrue(receipt['local_Adam_exact'])
        self.assertEqual(receipt['expected_sha256'],receipt['actual_sha256'])
        self.assertEqual(receipt['optimizer_updates_performed'],0)

    def test_input_alias_cannot_hide_loader_corruption(self):
        state=copy.deepcopy(self.optimizer().state_dict());target=self.optimizer()
        original=target.load_state_dict
        def corrupt(value):
            original(value)
            next(iter(target.state.values()))['exp_avg'].add_(1)
        target.load_state_dict=corrupt
        with self.assertRaises(ValueError):load_exact(target,state)

    def test_dtype_shape_scalar_type_and_nonfinite_values_are_not_relaxed(self):
        left={'tensor':torch.tensor([1.]),'step':torch.tensor(0.),'flag':True}
        for changed in (dict(left,tensor=torch.tensor([1.],dtype=torch.float64)),
                        dict(left,tensor=torch.tensor([[1.]])),dict(left,flag=1)):
            self.assertNotEqual(fingerprint(left),fingerprint(changed))
        with self.assertRaises(ValueError):fingerprint({'state':torch.tensor(float('nan'))})

    def test_hash_bound_mmap_load_preserves_checkpoint_and_rejects_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.pt';original={'step':3,'tensor':torch.arange(8,dtype=torch.float32)}
            with path.open('wb') as stream:torch.save(original,stream)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            loaded=load_checkpoint(path,digest)
            self.assertEqual(loaded['step'],3);torch.testing.assert_close(loaded['tensor'],original['tensor'])
            loaded['tensor'].add_(1)  # Private mmap writes cannot mutate durable saved bytes.
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
            with self.assertRaises(ValueError):load_checkpoint(path,'0'*64)


if __name__=='__main__':unittest.main()
