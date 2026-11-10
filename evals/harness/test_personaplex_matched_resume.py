import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.personaplex import matched_resume as r,matched_guard as g
from training.personaplex.matched_reuse import DATA_SHA


class ResumeAccounting(unittest.TestCase):
    text='8216374|capture|brics.u6ada|COMPLETED|0:0|327|gres/gpu=1|\n8229989|duplex-matched-train|brics.u6ada|FAILED|1:0|951|gres/gpu=2|\n8234444|duplex-matched-training-only|brics.u6ada|FAILED|1:0|5|gres/gpu=2|\n90000|duplex-matched-training-only|brics.u6ada|COMPLETED|0:0|7200|gres/gpu=2|\n'
    ordered=[dict(lease=dict(job_id='90000',prior_GPU_seconds=2239,maximum_seconds=86400))]

    def test_actual_usage_includes_all_windows_and_prior_failed_admission(self):
        self.assertEqual(r.allocation_carry(self.text,self.ordered,'90001'),2239+14400)

    def test_postupdate_failure_and_prior_quota_failure_are_both_carried(self):
        text=self.text+'8235976|duplex-matched-prior|brics.u6ada|FAILED|1:0|644|gres/gpu=2|\n8260542|duplex-matched-quota-recovery|brics.u6ada|FAILED|1:0|15147|gres/gpu=2|\n'
        ordered=[dict(lease=dict(self.ordered[0]['lease'],prior_GPU_seconds=33821,postupdate_recovery_authorization_sha256='actual-grant'))]
        self.assertEqual(r.allocation_carry(text,ordered,'90001'),33821+14400)
        with self.assertRaises(ValueError):r.allocation_carry(text.replace('|15147|','|15146|'),ordered,'90001')

    def test_normal88_recovery_includes_failed_restore_interval4(self):
        text=self.text+'8235976|duplex-matched-prior|brics.u6ada|FAILED|1:0|644|gres/gpu=2|\n8260542|duplex-matched-quota|brics.u6ada|FAILED|1:0|15147|gres/gpu=2|\n8273944|duplex-matched-postupdate|brics.u6ada|FAILED|1:0|119|gres/gpu=2|\n'
        ordered=[dict(lease=dict(self.ordered[0]['lease'],prior_GPU_seconds=34059,normal88_authorization_sha256='actual-rollback-grant'))]
        self.assertEqual(r.allocation_carry(text,ordered,'90001'),34059+14400)
        with self.assertRaises(ValueError):r.allocation_carry(text.replace('|119|','|118|'),ordered,'90001')

    def test_missing_duplicate_unlisted_failed_and_reset_carry_rejected(self):
        for text in (self.text.replace('|7200|','|86401|'),self.text.replace('COMPLETED|0:0|7200','FAILED|1:0|7200'),
                     self.text+self.text,self.text+'90002|duplex-matched-new|brics.u6ada|COMPLETED|0:0|20|gres/gpu=2|\n',
                     self.text.replace('90000|','90003|')):
            with self.subTest(text=text),self.assertRaises(ValueError):r.allocation_carry(text,self.ordered,'90001')
        with self.assertRaises(ValueError):r.allocation_carry(self.text,[dict(lease=dict(self.ordered[0]['lease'],prior_GPU_seconds=327))],'90001')


class ResumeSelection(unittest.TestCase):
    def fixture(self,root):
        previous=root/'prior';previous.mkdir();original=[];entries=[]
        for base in ('personaplex','context-spanning'):
            for cap in ('narrow','broad'):
                leaf=root/(base+'-'+cap+'.pt');leaf.write_bytes(b'fixture untrained state')
                original.append(dict(base=base,capacity=cap,original_checkpoint=str(leaf),checkpoint_sha256=g.sha(leaf)))
                entries.append(dict(base=base,capacity=cap,checkpoint=str(leaf),checkpoint_sha256=g.sha(leaf),
                    training_child_terminal=str(previous/(base+'-'+cap+'-train.json')),
                    final_restore_terminal=str(previous/(base+'-'+cap+'-restore.json'))))
        (previous/'joint-admission.json').write_text(json.dumps(dict(source='worker-A',data_report_sha256=DATA_SHA,original_joint_packet_sha256=r.PACKET_SHA,entries=entries)))
        return previous,dict(lineages=original)

    def test_successful_postquota_window_uses_new_terminal_not_old_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);previous,original=self.fixture(root)
            packet=previous/'joint-admission.json';value=json.loads(packet.read_text())
            value['entries'][0]['quota_interrupted_state']='dl-moshi-quota-interrupted-v1';packet.write_text(json.dumps(value))
            folder=previous/'personaplex/narrow';(folder/'train').mkdir(parents=True);(folder/'final-restore').mkdir()
            leaf=folder/'train/window.pt';leaf.write_bytes(b'fixture trained98 state')
            (folder/'train/training-window-terminal.json').write_text(json.dumps(dict(checkpoint=str(leaf),checkpoint_sha256=g.sha(leaf))))
            (folder/'final-restore/restore.json').write_text('{"fixture":true}')
            def actual_step(entry,*args):
                if entry['checkpoint']==str(leaf):
                    self.assertNotIn('quota_interrupted_state',entry);return 98
                return 3 if entry.get('quota_interrupted_state') else 0
            with patch.object(r,'restored_entry',side_effect=actual_step):
                plan=r.resume_plan(previous,original,root/'next',root/'execution','worker-A')
            chosen=next(row for row in plan if row['restored_step']==98)
            self.assertNotIn('quota_interrupted_state',chosen)
            self.assertIn('previous_training_terminal',chosen)

    def test_positive_real_window_selected_and_other_arms_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);previous,original=self.fixture(root)
            folder=previous/'personaplex/narrow';(folder/'train').mkdir(parents=True);(folder/'final-restore').mkdir()
            leaf=folder/'train/window.pt';leaf.write_bytes(b'fixture trained98 state')
            (folder/'train/training-window-terminal.json').write_text(json.dumps(dict(checkpoint=str(leaf),checkpoint_sha256=g.sha(leaf))))
            (folder/'final-restore/restore.json').write_text('{"fixture":true}')
            def actual_step(entry,*args):return 98 if entry['checkpoint']==str(leaf) else 0
            with patch.object(r,'restored_entry',side_effect=actual_step):
                plan=r.resume_plan(previous,original,root/'new',root/'execution','worker-A')
            self.assertEqual(len(plan),4);self.assertEqual(sorted(row['restored_step'] for row in plan),[0,0,0,98])
            chosen=next(row for row in plan if row['restored_step']==98)
            self.assertEqual(chosen['checkpoint'],str(leaf));self.assertIn('previous_training_child_terminal',chosen)
            leaf.write_bytes(b'changed')
            with patch.object(r,'restored_entry',side_effect=actual_step),self.assertRaisesRegex(ValueError,'bytes changed'):
                r.resume_plan(previous,original,root/'other',root/'execution','worker-A')

    def test_zero_new_updates_or_discarding_saved_updates_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);previous,original=self.fixture(root)
            with patch.object(r,'restored_entry',return_value=0),self.assertRaisesRegex(ValueError,'zero-update'):
                r.resume_plan(previous,original,root/'new',root/'execution','worker-A')
            # Two observations per arm; the selected state cannot go backwards.
            with patch.object(r,'restored_entry',side_effect=[98,0]),self.assertRaisesRegex(ValueError,'discarded'):
                r.resume_plan(previous,original,root/'new',root/'execution','worker-A')

    def test_completed_lineages_skipped_after_all4_fresh_restores(self):
        from training.personaplex.matched_training_only import execute,initial_plan
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);previous,original=self.fixture(root);output=root/'next';output.mkdir();execution=root/'execution';execution.mkdir()
            plan=initial_plan(original,output,execution);calls=[]
            for row in plan:row['restored_step']=196 if row['capacity']=='narrow' else 98
            class Children:
                def parallel(self,action):
                    for base in ('personaplex','context-spanning'):action(base)
                def run(self,base,capacity,stage,folder,extra=(),label=None):
                    folder.mkdir();calls.append((base,capacity,stage,label));terminal=execution/(base+capacity+(label or stage)+'.json');terminal.write_text('{"fixture":true}')
                    if stage=='restore':(folder/'restore.json').write_text('{"fixture":true}')
                    else:
                        if capacity!='broad':raise AssertionError('Completed196 arm must not train again')
                        if sum(c[2]=='restore' and c[3] is None for c in calls)!=4:raise AssertionError('All4 fresh restores required first')
                        (folder/'training-window-terminal.json').write_text(json.dumps(dict(completed=False,total_lineage_updates=106,actual_new_updates=8,checkpoint=str(folder/'state.pt'),checkpoint_sha256='fixture')))
                    return terminal
            partial=execute(Children(),plan,output,output/'packet.json','worker-A')
            self.assertEqual(len(partial),2);self.assertEqual([c[1] for c in calls if c[2]=='train'],['broad','broad'])


class WorkerAndChain(unittest.TestCase):
    def test_unverified_worker_and_changed_controller_math_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'native.py').write_bytes(b'worker bytes')
            with patch.object(r,'BASE',root),patch.object(r,'CONTROL',('native.py',)),patch.object(r.subprocess,'check_output',return_value='wrong'):
                with self.assertRaisesRegex(ValueError,'immutable native'):r.verify_worker(root,'worker-A','controller-C')

    def test_failed_or_finished_window_cannot_resume(self):
        for completed,code in ((False,1),(True,0)):
            with tempfile.TemporaryDirectory() as directory:
                base=Path(directory);root=base/'initial';root.mkdir();execution=root/'execution-1';execution.mkdir()
                lease=dict(source='worker-A',job_id='1',continuation_interval=1)
                (execution/'lease.json').write_text(json.dumps(lease));(execution/'terminal.json').write_text(json.dumps(dict(source='worker-A',job_id='1',actual_exit_code=code)))
                result=root/'phase-result.json';result.write_text(json.dumps(dict(source='worker-A',all_four_complete=completed,partial_lineages=[dict(step=8)])))
                with patch.object(r,'BASE',base),patch.object(r,'INITIAL',root),self.assertRaisesRegex(ValueError,'successful incomplete'):
                    r.chain(root,g.sha(result))


class StartupIntervalSequence(unittest.TestCase):
    def test_positive_windows_carry_the_consumed_startup_interval_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);execution=root/'execution-90000';execution.mkdir()
            result=root/'phase-result.json';result.write_text(json.dumps(dict(source='worker',all_four_complete=False,partial_lineages=[dict(step=8)])))
            lease=execution/'lease.json';terminal=execution/'terminal.json'
            terminal.write_text(json.dumps(dict(source='worker',job_id='90000',actual_exit_code=0)))
            lease.write_text(json.dumps(dict(source='worker',job_id='90000',continuation_interval=2)))
            with patch.object(r,'BASE',root),patch.object(r,'INITIAL',root):
                self.assertEqual(r.chain(root,g.sha(result))[0]['lease']['continuation_interval'],2)
                lease.write_text(json.dumps(dict(source='worker',job_id='90000',continuation_interval=1)))
                with self.assertRaisesRegex(ValueError,'interval sequence'):
                    r.chain(root,g.sha(result))


if __name__=='__main__':unittest.main()
