import json
from pathlib import Path
import tempfile
import unittest

from portable.context.native_control_copyback import ROOTS,snapshot,verify


class ControlCopyback(unittest.TestCase):
    def test_material_progress_copies_exact_state_and_excludes_sensitive_or_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);numerical=base/ROOTS[0];numerical.mkdir(parents=True)
            (numerical/'numerical.json').write_text(json.dumps(dict(cases=[dict(passed=True)],passed=False)))
            training=base/ROOTS[2]/'training';training.mkdir(parents=True)
            (training/'training-state.pt').write_bytes(b'atomic saved optimizer and RNG placeholder')
            for name in ('session-token','core.123','dump.core','training-state.pt.tmp'):
                (training/name).write_bytes(b'excluded')
            private=training/'.private';private.mkdir();(private/'env').write_bytes(b'excluded')
            (training/'outside').symlink_to(base.parent)
            first=snapshot(base);out=Path(first['snapshot'])
            self.assertEqual(verify(out,first['inventory_sha256']),2)
            self.assertEqual((out/ROOTS[2]/'training/training-state.pt').read_bytes(),(training/'training-state.pt').read_bytes())
            self.assertFalse(snapshot(base,first['marker'])['changed'])
            (training/'training-state.pt').write_bytes(b'next saved step')
            self.assertTrue(snapshot(base,first['marker'])['changed'])
            (out/ROOTS[2]/'training/training-state.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify(out,first['inventory_sha256'])

    def test_symlinked_root_cannot_import_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            base=Path(directory);root=base/ROOTS[0];root.parent.mkdir(parents=True)
            root.symlink_to(other,target_is_directory=True)
            with self.assertRaises(ValueError):snapshot(base)

    def test_new_versioned_training_output_triggers_its_own_verified_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);old=base/ROOTS[2];old.mkdir(parents=True)
            (old/'controller.json').write_text('{"phase":"retired"}')
            first=snapshot(base)
            new=base/ROOTS[3]/'training';new.mkdir(parents=True)
            (new/'training-state.pt').write_bytes(b'new optimizer state')
            second=snapshot(base,first['marker']);self.assertTrue(second['changed'])
            copied=Path(second['snapshot'])/ROOTS[3]/'training/training-state.pt'
            self.assertEqual(copied.read_bytes(),b'new optimizer state')
            self.assertEqual(verify(Path(second['snapshot']),second['inventory_sha256']),2)


if __name__=='__main__':unittest.main()
