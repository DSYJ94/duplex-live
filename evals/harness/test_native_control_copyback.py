import json
from pathlib import Path
import tempfile
import unittest
import shlex
import shutil
import subprocess

from portable.context.native_control_copyback import ROOTS,snapshot,verify,ssh_command


class ControlCopyback(unittest.TestCase):
    def test_native49_receipt_triggers_backup_before_combined_stage_changes(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);controller=base/'native-control40-precise-pilot-v3-adoption'
            controller.mkdir();(controller/'controller.json').write_text('{"phase":"combined_readiness"}')
            first=snapshot(base)
            ready=base/'native-control40-precise-pilot-v2/combined-readiness';ready.mkdir(parents=True)
            report=ready/'native-stream49-gradient-readiness.json'
            report.write_text(json.dumps(dict(passed=True,cases=[dict(id=i)for i in range(49)])))
            second=snapshot(base,first['marker']);self.assertTrue(second['changed'])
            self.assertEqual(verify(Path(second['snapshot']),second['inventory_sha256']),2)
            self.assertEqual((Path(second['snapshot'])/report.relative_to(base)).read_bytes(),report.read_bytes())

    def test_current_v6_supervisor_metadata_is_included(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);controller=base/'native-control40-precise-pilot-v6-adoption'
            controller.mkdir();(controller/'controller.json').write_text('{"phase":"combined_readiness"}')
            result=snapshot(base)
            self.assertEqual(verify(Path(result['snapshot']),result['inventory_sha256']),1)

    def test_material_progress_copies_exact_state_and_excludes_sensitive_or_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);numerical=base/ROOTS[0];numerical.mkdir(parents=True)
            (numerical/'numerical.json').write_text(json.dumps(dict(cases=[dict(passed=True)],passed=False)))
            training=base/ROOTS[2]/'training';training.mkdir(parents=True)
            (training/'training-state.pt').write_bytes(b'atomic saved optimizer and RNG placeholder')
            for name in ('session-token','core.123','dump.core','training-state.pt.tmp','openai-backend.env'):
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

    def test_replacement_pilot_and_fresh_actor_progress_are_both_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            pilot=base/'native-control40-precise-pilot-v2/numerical';pilot.mkdir(parents=True)
            (pilot/'numerical.json').write_text(json.dumps(dict(cases=[dict(passed=True)],passed=False)))
            fresh=base/'native-control40-fresh-paired40-v1/baseline520';fresh.mkdir(parents=True)
            (fresh/'execution.json').write_text(json.dumps(dict(phase='collecting_fresh20',cases=[])))
            first=snapshot(base);self.assertEqual(verify(Path(first['snapshot']),first['inventory_sha256']),2)
            case=fresh/'case-00';case.mkdir();(case/'model.wav').write_bytes(b'new synthetic output')
            (fresh/'execution.json').write_text(json.dumps(dict(phase='collecting_fresh20',cases=[dict(index=0)])))
            second=snapshot(base,first['marker']);self.assertTrue(second['changed'])
            self.assertEqual(verify(Path(second['snapshot']),second['inventory_sha256']),3)
            self.assertEqual((Path(second['snapshot'])/case.relative_to(base)/'model.wav').read_bytes(),b'new synthetic output')
            speech=base/'native-control40-fresh-paired40-v1/independent-speech40/baseline520'
            speech.mkdir(parents=True);(speech/'report.json').write_text('{"results":[]}')
            third=snapshot(base,second['marker']);self.assertTrue(third['changed'])
            self.assertEqual(verify(Path(third['snapshot']),third['inventory_sha256']),4)

    def test_explicit_current_host_and_port_replace_deleted_pod_defaults(self):
        command=ssh_command(Path('/local/key'),Path('/local/known-hosts'),'root@192.0.2.10',22022)
        self.assertEqual(command[command.index('-p')+1],'22022')
        self.assertIn('StrictHostKeyChecking=yes',command)
        for host,port in [('-oProxyCommand=bad',22),('root@host;bad',22),('root@host',0),('root@host',65536),('root@host',True)]:
            with self.subTest(host=host,port=port),self.assertRaises(ValueError):
                ssh_command(Path('/key'),Path('/known-hosts'),host,port)

    @unittest.skipUnless(shutil.which('ssh'),'OpenSSH is required for effective configuration verification')
    def test_real_openssh_config_preserves_known_hosts_filename_with_spaces(self):
        filename=Path('/local/Shared GitHub Repos/duplex-live/known-hosts')
        command=ssh_command(Path('/local/key'),filename,'root@192.0.2.10',22022)
        # Exercise OpenSSH's own parser after the rsync remote-shell round trip.
        command=shlex.split(shlex.join(command))
        result=subprocess.run(command+['-G','-F','/dev/null','root@192.0.2.10'],
                              check=True,capture_output=True,text=True,timeout=5)
        actual=next(line for line in result.stdout.splitlines() if line.startswith('userknownhostsfile '))
        self.assertEqual(actual.partition(' ')[2],str(filename))


if __name__=='__main__':unittest.main()
