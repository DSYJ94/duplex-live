import unittest
from evals.lychee_training_state import save


class StateBoundaryTests(unittest.TestCase):
    def test_heldout_history_rejected_before_tensor_access(self):
        for split in ('validation','test'):
            with self.assertRaisesRegex(RuntimeError,'private TRAIN'):
                save(None,None,None,{'split':split,'kind':'private'},None,None,None)

    def test_unreceived_result_cannot_enter_train_history(self):
        with self.assertRaisesRegex(RuntimeError,'actually received'):
            save(None,None,None,{'split':'train','kind':'private'},
                {'snapshot_elapsed_s':4.},{'status':'completed','received_at_s':5.},None)


if __name__=='__main__':unittest.main()
