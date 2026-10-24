import json
from pathlib import Path
import tempfile
import unittest
import shlex
import shutil
import subprocess

from portable.context.native_control_copyback import ROOTS,snapshot,verify,ssh_command,inventory_rows,reuse_local_files,receive_snapshot,sha,marker


class ControlCopyback(unittest.TestCase):
    def test_large_file_hash_includes_multiple_chunks_and_non_aligned_tail(self):
        import hashlib
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory)/'tensor.pt';data=bytes(range(256))*(32768+1)+b'nonaligned tail'
            p.write_bytes(data)
            self.assertEqual(sha(p),hashlib.sha256(data).hexdigest())

    def test_adapter_file_triggers_backup_before_final_report(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);root=base/'native-control40-precise-pilot-v11-environment-retry'
            root.mkdir();(root/'controller.json').write_text('{"phase":"bounded8_training"}')
            first=snapshot(base);training=root/'training';training.mkdir()
            p=training/'adapters.safetensors';p.write_bytes(b'final standalone adapter')
            second=snapshot(base,first['marker']);self.assertTrue(second['changed'])
            self.assertEqual((Path(second['snapshot'])/p.relative_to(base)).read_bytes(),p.read_bytes())

    def test_training_and_restore_fingerprints_trigger_coarse_milestones_not_each_case(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);root=base/'native-control40-precise-pilot-v11-environment-retry';root.mkdir()
            path=root/'bounded8_training.log';path.write_text('')
            zero=marker(base)
            def write(n,c=0):
                path.write_text(('"native_stream_restore_fingerprint":\n'*n)+('"control40_restore_fingerprint":\n'*c))
            write(9);self.assertEqual(marker(base),zero)
            write(10);ten=marker(base);self.assertNotEqual(ten,zero)
            write(19);self.assertEqual(marker(base),ten)
            write(40);forty=marker(base)
            write(49);final49=marker(base);self.assertNotEqual(final49,forty)
            write(49,9);self.assertEqual(marker(base),final49)
            write(49,10);self.assertNotEqual(marker(base),final49)
            restore=root/'independent_restore.log';restore.write_text('')
            before=marker(base);restore.write_text('"native_stream_restore_fingerprint":\n'*10)
            self.assertNotEqual(marker(base),before)

    def test_quota_recovery_stages_snapshots_outside_source_volume_and_reuses_verified_bytes(self):
        with tempfile.TemporaryDirectory()as directory, tempfile.TemporaryDirectory()as disk:
            base=Path(directory);root=base/'native-control40-precise-pilot-v10-quota-retry/training'
            root.mkdir(parents=True);state=root/'training-state.pt';state.write_bytes(b'checkpoint1')
            snapshots=Path(disk)/'snapshots'
            first=snapshot(base,snapshot_root=snapshots)
            second=snapshot(base,snapshot_root=snapshots)
            self.assertFalse((base/'native-control40-backup-snapshots').exists())
            self.assertTrue((Path(first['snapshot'])/state.relative_to(base)).samefile(Path(second['snapshot'])/state.relative_to(base)))
            self.assertFalse(state.samefile(Path(first['snapshot'])/state.relative_to(base)))
            state.write_bytes(b'checkpoint2')
            third=snapshot(base,first['marker'],snapshot_root=snapshots)
            self.assertTrue(third['changed']);self.assertEqual(verify(Path(third['snapshot']),third['inventory_sha256']),1)
            self.assertEqual((Path(first['snapshot'])/state.relative_to(base)).read_bytes(),b'checkpoint1')

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

    def test_retained154_receipt_triggers_backup_with_same_controller_phase(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);root=base/'native-control40-precise-pilot-v8-recovery';root.mkdir()
            (root/'controller.json').write_text('{"phase":"combined_readiness"}')
            first=snapshot(base);ready=root/'combined-readiness';ready.mkdir()
            (ready/'complete-balanced-preflight.json').write_text('{"passed":true,"total_components":154}')
            second=snapshot(base,first['marker']);self.assertTrue(second['changed'])
            self.assertEqual(verify(Path(second['snapshot']),second['inventory_sha256']),2)

    def test_snapshots_reuse_verified_immutable_bytes_without_linking_live_outputs(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);root=base/'native-control40-precise-pilot-v8-recovery';root.mkdir()
            (root/'controller.json').write_text('{"phase":"combined_readiness"}')
            payload=root/'reference.pt';payload.write_bytes(b'unchanged reference tensors')
            first=snapshot(base);second=snapshot(base)
            a=Path(first['snapshot'])/payload.relative_to(base);b=Path(second['snapshot'])/payload.relative_to(base)
            self.assertTrue(a.samefile(b));self.assertFalse(payload.samefile(b))
            payload.write_bytes(b'new live output')
            self.assertEqual(a.read_bytes(),b'unchanged reference tensors')
            self.assertEqual(b.read_bytes(),b'unchanged reference tensors')
            self.assertEqual(verify(Path(second['snapshot']),second['inventory_sha256']),2)

    def test_local_reuse_links_only_matching_verified_bytes_and_rejects_path_escape(self):
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);root=base/'native-control40-precise-pilot-v8-recovery';root.mkdir()
            live=root/'reference.pt';live.write_bytes(b'immutable saved reference')
            previous=snapshot(base);reference=Path(previous['snapshot']);rows=inventory_rows(reference)
            target=base/'local-new';target.mkdir()
            missing,reused=reuse_local_files(reference,target,rows)
            self.assertEqual(missing,[]);self.assertEqual(reused,[rows[0]['path']])
            self.assertTrue((target/rows[0]['path']).samefile(reference/rows[0]['path']))
            second=base/'local-after-corruption';second.mkdir();(reference/rows[0]['path']).write_bytes(b'bad source')
            missing,reused=reuse_local_files(reference,second,rows)
            self.assertEqual(missing,[rows[0]['path']]);self.assertEqual(reused,[])
            bad=dict(rows[0],path='../outside')
            with self.assertRaises(ValueError):reuse_local_files(None,second,[bad])

    def test_receive_transfers_only_changed_files_but_verifies_entire_inventory(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as directory:
            base=Path(directory);root=base/'native-control40-precise-pilot-v8-recovery';root.mkdir()
            (root/'reference.pt').write_bytes(b'large immutable reference')
            (root/'controller.json').write_text('{"phase":"prepared"}')
            prior=snapshot(base);reference=Path(prior['snapshot'])
            (root/'controller.json').write_text('{"phase":"combined_readiness"}')
            current=snapshot(base);provider=Path(current['snapshot']);info=dict(current,snapshot='/workspace/migration/native-control40-backup-snapshots/'+provider.name)
            transferred=[]
            def transfer(command,**kwargs):
                if command[-2].endswith('/inventory.json'):
                    shutil.copyfile(provider/'inventory.json',Path(command[-1])/'inventory.json')
                else:
                    names=Path(command[command.index('--files-from')+1]).read_bytes().decode().strip('\0').split('\0')
                    transferred.extend(names)
                    for name in names:
                        dest=Path(command[-1])/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(provider/name,dest)
                return subprocess.CompletedProcess(command,0,'','')
            with patch('portable.context.native_control_copyback.subprocess.run',side_effect=transfer):
                count,receipt=receive_snapshot(info,base/'received',['ssh'],'root@example.test',reference)
            self.assertEqual(count,2);self.assertEqual(receipt['reused_verified_files'],1)
            self.assertEqual(transferred,[(root/'controller.json').relative_to(base).as_posix()])
            self.assertEqual(receipt['transferred_file_bytes'],(root/'controller.json').stat().st_size)
            (base/'received'/transferred[0]).write_text('corrupt')
            with self.assertRaises(ValueError):verify(base/'received',info['inventory_sha256'])

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
