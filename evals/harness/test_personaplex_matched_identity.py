from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from training.personaplex import matched_identity as i,matched_native as n


class LineageBoundary(unittest.TestCase):
    def args(self,origin=i.ORIGIN):return SimpleNamespace(source='b'*40,lineage_source=origin,stage='restore',base='personaplex',capacity='broad',inputs_sha='d'*64)

    def test_checkpoint_metadata_keeps_lineage_while_execution_producer_is_distinct(self):
        args=self.args();meta=n.metadata(args,{'input_sha256':'e'*64})
        self.assertEqual(meta['source'],i.ORIGIN);self.assertNotEqual(meta['source'],args.source)
        self.assertTrue(i.reference_source_matches(args,i.ORIGIN));self.assertTrue(i.reference_source_matches(args,args.source))
        self.assertFalse(i.reference_source_matches(args,'third-unverified-producer'))
        self.assertEqual(n.metadata(self.args(None),{'input_sha256':'e'*64})['source'],'b'*40)

    def test_reuse_forbids_repeating_native_admission_or_arbitrary_origin(self):
        args=self.args();args.stage='admit'
        with self.assertRaisesRegex(ValueError,'cannot repeat'):i.validate_lineage(args)
        with self.assertRaisesRegex(ValueError,'exact independently admitted'):i.lineage_source(self.args('c'*40))

    def original(self):return subprocess.check_output(['git','show',i.ORIGIN+':'+i.NATIVE],text=True)

    def test_actual_original_source_allows_only_declared_identity_changes(self):
        checked=i.compare_native(self.original(),Path(i.NATIVE).read_text())
        self.assertIn('run_training',checked);self.assertIn('restore',checked);self.assertIn('runtime',checked)

    def test_math_optimizer_recipe_or_restore_fingerprint_change_rejected(self):
        current=Path(i.NATIVE).read_text()
        for old,new in [('learning_rate=2e-6','learning_rate=3e-6'),('weight_decay=RECIPE[\'weight_decay\']','weight_decay=0.'),
                ("fingerprint(lm,groups)!=original['fingerprints']","False")]:
            self.assertIn(old,current)
            with self.subTest(edit=old),self.assertRaises(ValueError):i.compare_native(self.original(),current.replace(old,new))

    def test_uncommitted_native_or_identity_checker_never_claims_compatibility(self):
        with patch.object(i.subprocess,'check_output',return_value=b'uncommitted-mismatch'):
            with self.assertRaisesRegex(ValueError,'differs from execution producer'):i.validate_lineage(self.args())


if __name__=='__main__':unittest.main()
