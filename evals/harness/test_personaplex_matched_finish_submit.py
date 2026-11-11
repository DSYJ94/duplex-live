import json,tempfile,unittest,subprocess
from pathlib import Path
from unittest.mock import patch
from training.personaplex import matched_finish_submit as s


class FinalSubmission(unittest.TestCase):
    def test_budget_uses_actual_prior_and_rounds_down_with_reserve(self):
        self.assertEqual(s.walltime(135137),86400)
        self.assertEqual(s.walltime(307937),47520)
        for prior in (135137,307937,340000):
            seconds=s.walltime(prior)
            self.assertEqual(seconds%60,0);self.assertLessEqual(seconds,86400)
            self.assertLess(prior+2*seconds,112*3600)
        for invalid in (0,135136,True,1.5,403200):
            with self.assertRaises(ValueError):s.walltime(invalid)

    def test_complete_and_partial_routes_do_not_repeat_training(self):
        for complete,module in ((True,'matched_evaluation_host'),(False,'matched_finish')):
            cmd=s.command(Path('/allocated/controller'),'new-controller','phase-sha',complete,47520)
            self.assertIn(module,cmd[-1]);self.assertIn('--time=792',cmd)
            self.assertIn('--no-requeue',cmd);self.assertIn('--export=NONE',cmd)
            self.assertIn('--gpus=2',cmd);self.assertNotIn('matched_native --stage admit',cmd[-1])
            self.assertEqual('--inputs-sha' in cmd[-1],not complete)

    def test_missing_or_failed_parent_cannot_create_claim_or_submit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);claim=root/'claim'
            with patch.object(s,'ROOT',root),patch.object(s,'CLAIM',claim),patch.object(s.resume,'beneath',return_value=root):
                with self.assertRaises(FileNotFoundError):s.parent(root)
                (root/'phase-result.json').write_text(json.dumps(dict(all_four_complete=False)))
                with patch.object(s.resume,'chain',side_effect=ValueError('actual failed9')):
                    with self.assertRaisesRegex(ValueError,'failed9'):s.parent(root)
                self.assertFalse(claim.exists())

    def test_wrong_parent_identity_does_not_borrow_positive_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'phase-result.json').write_text(json.dumps(dict(all_four_complete=False)))
            old=dict(job_id='8317068',source=s.SOURCE,continuation_interval=9,prior_GPU_seconds=135137)
            old[s.startup.KEY]=s.startup.authority()
            with patch.object(s,'ROOT',root),patch.object(s.resume,'beneath',return_value=root),patch.object(s.resume,'chain',return_value=[dict(lease=old)]):
                self.assertEqual(s.parent(root)[2],False)
                for key,value in [('job_id','8314862'),('source','different'),('continuation_interval',8),('prior_GPU_seconds',0)]:
                    with patch.dict(old,{key:value}),self.assertRaises(ValueError):s.parent(root)


class ExclusiveFinalClaim(unittest.TestCase):
    def test_timeout_keeps_claim_and_second_call_cannot_submit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);work=root/'controller';work.mkdir();claim=root/'claim'
            old=dict(source=s.SOURCE,worker_worktree='/old/worker')
            def query(cmd,**kwargs):
                if cmd[:2]==['git','rev-parse']:return 'controller\n'
                return ''
            with patch.object(s,'DURABLE',root),patch.object(s,'CLAIM',claim),patch.object(s.Path,'cwd',return_value=work),\
                 patch.object(s,'parent',return_value=([dict(lease=old)],'phase-sha',False)),\
                 patch.object(s.resume,'verify_worker'),patch.object(s.resume,'allocation_carry',return_value=307937),\
                 patch.object(s,'storage_check',return_value={'passed':True}),patch.object(s,'forecast',return_value={'synthetic':'schedule'}),patch.object(s,'backup_proofs',return_value={'both':'synthetic fixture'}),\
                 patch.object(s.subprocess,'check_output',side_effect=query),\
                 patch.object(s.subprocess,'run',side_effect=subprocess.TimeoutExpired('sbatch',60)) as mutate:
                with self.assertRaises(subprocess.TimeoutExpired):s.submit('controller')
                self.assertEqual(json.loads((claim/'unknown.json').read_text())['state'],'UNKNOWN_DO_NOT_RESUBMIT')
                self.assertTrue((work/'runs').is_dir())
                self.assertEqual(json.loads((claim/'command.json').read_text())['maximum_seconds'],47520)
                with self.assertRaises(FileExistsError):s.submit('controller')
                self.assertEqual(mutate.call_count,1)

    def test_missing_backup_fails_before_durable_claim_or_submission(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);claim=root/'claim'
            with patch.object(s,'ROOT',root),patch.object(s,'CLAIM',claim),patch.object(s.subprocess,'run') as mutate:
                with self.assertRaises(FileNotFoundError):s.backup_proofs(root,'worker')
                mutate.assert_not_called();self.assertFalse(claim.exists())

class FullNewStateBackups(unittest.TestCase):
    def test_actual_manifest_files_and_source_pair_must_match(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'job';backups=Path(d)/'backup';records=[];checkpoints=set()
            for base in ('personaplex','context-spanning'):
                folder=root/base/'broad/train';folder.mkdir(parents=True)
                checkpoint=folder/'window.pt';checkpoint.write_bytes(('synthetic fixture '+base).encode());checkpoints.add(checkpoint)
                digest=s.g.sha(checkpoint);dest=backups/base;dest.mkdir(parents=True)
                (dest/checkpoint.name).write_bytes(checkpoint.read_bytes())
                value=dict(source='worker',actual_new_updates=1,total_lineage_updates=147,checkpoint=str(checkpoint),checkpoint_sha256=digest)
                (folder/'training-window-terminal.json').write_text(json.dumps(value))
                (dest/'saved.json').write_text(json.dumps(value))
                (dest/'window.pt.native-reference.json').write_text(json.dumps(dict(source='worker',step=147,checkpoint_sha256=digest)))
                files={p.name:dict(bytes=p.stat().st_size,sha256=s.g.sha(p)) for p in dest.iterdir()}
                record=dict(source='worker',checkpoint=str(checkpoint),destination=str(dest/'window.pt'),step=147,
                    sha256=digest,bytes=checkpoint.stat().st_size,files=files,actual_rsync_exit_code=0,hash_verified=True,separate_filesystem=True)
                (dest/'archive-receipt.json').write_text(json.dumps(record));records.append(record)
            (root/'independent-archive-status.json').write_text(json.dumps(dict(passed=True,job_id='8317068',source='worker',copies=records)))
            real_stat=Path.stat
            def stat(path,*args,**kwargs):
                value=real_stat(path,*args,**kwargs)
                return SimpleNamespace(st_dev=value.st_dev+1,st_size=value.st_size) if path in checkpoints else value
            with patch.object(s,'BACKUP_BASE',backups),patch.object(Path,'stat',stat):
                self.assertEqual(set(s.backup_proofs(root,'worker')),{'personaplex','context-spanning'})
                (backups/'personaplex/window.pt').write_bytes(b'corrupt fixture')
                with self.assertRaises(ValueError):s.backup_proofs(root,'worker')



class FinalSchedule(unittest.TestCase):
    def test_no_short_final_window_or_recalibration(self):
        original={'lineages':[dict(base=b,capacity=c,estimated_complete196_seconds=120000.) for b in ('personaplex','context-spanning') for c in ('narrow','broad')]}
        rows=[dict(base=b,capacity=c,restored_step=196 if c=='narrow' else 147) for b in ('personaplex','context-spanning') for c in ('narrow','broad')]
        with patch.object(s.resume,'preflight',return_value=original),patch.object(s.resume,'resume_plan',return_value=rows):
            result=s.forecast(Path('/synthetic-parent'),47520,False)
            self.assertEqual(result['required_seconds'],42000.)
            self.assertFalse(result['model_executed'])
            with self.assertRaisesRegex(ValueError,'do not fit'):s.forecast(Path('/synthetic-parent'),41000,False)

    def test_complete_parent_needs_no_training_forecast(self):
        with patch.object(s.resume,'preflight') as original,patch.object(s.resume,'resume_plan') as restore:
            self.assertEqual(s.forecast(Path('/synthetic-parent'),11400,True)['training_remaining_seconds'],0)
            original.assert_not_called();restore.assert_not_called()

if __name__=='__main__':unittest.main()
