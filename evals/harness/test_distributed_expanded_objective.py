import unittest
from unittest.mock import Mock,patch
from training.personaplex.distributed_expanded_objective import scale,backward
from training.personaplex.distributed_train import update


class ExpandedNativeContract(unittest.TestCase):
    def collective(self,flags):
        def gather(out,value):out[:]=[dict(value,valid=v) for v in flags]
        return patch('torch.distributed.all_gather_object',side_effect=gather)

    def test_partial_global_batch_keeps_equal_example_gradient_mass(self):
        with patch('torch.distributed.get_world_size',return_value=4),self.collective([True,True,False,False]):
            self.assertEqual(scale(True,3.),6.)
            self.assertEqual(scale(False,3.),0.)
            # DDP averages four replicas; two valid examples each receive1/2.
            self.assertEqual((6.*2+6.*4)/4,3.*(2+4)/2)

    def test_microbatches_normalize_over_whole_global_batch(self):
        with patch('torch.distributed.get_world_size',return_value=4),self.collective([True,False,False,False]):
            self.assertAlmostEqual(scale(True,21*49/1008,global_batch_count=21),4*49/1008,places=15)
            self.assertEqual(scale(False,21*49/1008,global_batch_count=21),0.)

    def test_empty_global_batch_is_rejected(self):
        with patch('torch.distributed.get_world_size',return_value=2),self.collective([False,False]):
            with self.assertRaises(ValueError):scale(False,1.)

    def test_saved_update_cannot_be_replayed(self):
        runtime,base,opal=Mock(),Mock(),Mock()
        with patch('training.personaplex.distributed_train.checkpoint_manifest',return_value={'update':8}):
            with self.assertRaises(ValueError):update(runtime,base,opal,step=7,retained_groups={},references={},exposures=[],expanded_row={},valid=True,expanded_coefficient=1.,identity={},checkpoint_path=None,log_path='unused')
        base.zero_grad.assert_not_called();opal.step.assert_not_called()

    def test_no_update_before_actual_native_admissions(self):
        runtime,base,opal=Mock(),Mock(),Mock()
        with patch('training.personaplex.distributed_train.checkpoint_manifest',return_value={'update':0}):
            with self.assertRaises(ValueError):update(runtime,base,opal,step=0,retained_groups={},references={},exposures=[],expanded_row={},valid=True,expanded_coefficient=1.,identity={},checkpoint_path=None,log_path='unused')
        base.zero_grad.assert_not_called();opal.step.assert_not_called()


if __name__=='__main__':unittest.main()
