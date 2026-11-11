import unittest,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex import matched_rpc_recovery as r
from training.personaplex import matched_rpc_job as job
from training.personaplex import matched_continue_policy as policy
from training.personaplex import matched_resume as resume


class ExactFailureStates(unittest.TestCase):
    def test_missing_amendment_stops_before_storage_or_model_preparation(self):
        args=SimpleNamespace(source='candidate',inputs_sha=job.DATA_SHA)
        with patch.object(job.subprocess,'check_output',return_value='candidate\n'),\
             patch.object(r,'authority',side_effect=ValueError('pending amendment')),\
             patch.object(job,'storage_check') as storage,patch.object(job,'preflight') as models:
            with self.assertRaisesRegex(ValueError,'pending amendment'):job.main(args)
            storage.assert_not_called();models.assert_not_called()

    def test_exact_selected_reference_requires_actual_owner_grant(self):
        for key in r.STATES:
            path,step,digest=r.selected(*key)
            args=SimpleNamespace(base=key[0],capacity=key[1],checkpoint=str(path),checkpoint_sha=digest)
            with patch.object(r,'authority',side_effect=ValueError('pending')):
                with self.assertRaisesRegex(ValueError,'pending'):r.reference_allowed(args,r.PREDECESSOR)
            with patch.object(r,'authority',return_value='actual-grant'):
                self.assertTrue(r.reference_allowed(args,r.PREDECESSOR))
                args.checkpoint_sha='0'*64;self.assertFalse(r.reference_allowed(args,r.PREDECESSOR))

    def test_normal56_rollback_and_unrelated_sources_are_not_allowed(self):
        path,step,digest=r.selected('personaplex','broad')
        args=SimpleNamespace(base='personaplex',capacity='broad',checkpoint=str(path.with_name('live.pt')),checkpoint_sha=digest)
        with patch.object(r,'authority') as grant:
            self.assertFalse(r.reference_allowed(args,r.PREDECESSOR))
            args.checkpoint=str(path);self.assertFalse(r.reference_allowed(args,'another-source'))
            grant.assert_not_called()

    def test_steps_pin_both_completed_narrows_and_latest_broads(self):
        self.assertEqual({k:v[0] for k,v in r.STATES.items()},
            {('personaplex','narrow'):196,('context-spanning','narrow'):196,
             ('personaplex','broad'):62,('context-spanning','broad'):63})

    def test_carried196_uses_new_restore_and_original_training_receipt(self):
        # Synthetic CPU protocol fixture; never a claim of a CUDA restore.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'personaplex/narrow/train';folder.mkdir(parents=True)
            checkpoint=folder/'epoch-04.pt';checkpoint.write_bytes(b'synthetic fixture')
            digest=job.g.sha(checkpoint);old='original-training';fresh='new-restore'
            def write(path,value):path.write_text(json.dumps(value));return str(path)
            write(Path(str(checkpoint)+'.native-reference.json'),dict(source=r.PREDECESSOR,step=196,checkpoint_sha256=digest,execution_id=old))
            original=dict(source=r.PREDECESSOR,total_lineage_updates=196,checkpoint_sha256=digest,
                final_checkpoint=str(checkpoint),completed=True,complete_epochs=4)
            old_terminal=folder/'training-terminal.json';write(old_terminal,original)
            restore=root/'fresh-restore.json';write(restore,dict(source='new-producer',lineage_source=policy.ORIGIN,
                base='personaplex',capacity='narrow',data_report_sha256=policy.DATA_SHA,checkpoint_sha256=digest,
                passed=True,actual_CUDA=True,all_Adam_RNG_OPAL_fields_exact=True,native_logits_exact=True,
                execution_id=fresh,reference_execution_id=old,restored_training_step=196))
            terminal=root/'fresh-child.json';write(terminal,dict(source='new-producer',base='personaplex',capacity='narrow',
                stage='restore',execution_id=fresh,actual_exit_code=0))
            row=dict(base='personaplex',capacity='narrow',checkpoint=str(checkpoint),checkpoint_sha256=digest,
                restore=str(restore),restore_sha256=job.g.sha(restore),restore_terminal=str(terminal),rpc_failure_recovery=r.MARKER)
            with patch.object(r,'FAILED_ROOT',root),patch.dict(r.STATES,{('personaplex','narrow'):(196,digest,'epoch-04.pt')}),patch.object(r,'authority',return_value='grant'):
                self.assertEqual(policy.restored_entry(row,'new-producer',policy.DATA_SHA,{}),196)
                self.assertEqual(json.loads(old_terminal.read_text()),original)
                bad=dict(row);bad.pop('rpc_failure_recovery')
                with self.assertRaises(ValueError):policy.restored_entry(bad,'new-producer',policy.DATA_SHA,{})

    def test_proposal_does_not_itself_authorize_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            missing=Path(directory)/'no-owner-grant.json'
            with patch.object(r,'AUTH',missing):
                with self.assertRaises(FileNotFoundError):r.authority()

    def test_successful7_carries_failed6_without_relabelling_or_reset(self):
        lines=[]
        for ident,value in job.HISTORY.items():
            code='0:0' if value['state']=='COMPLETED' else '1:0'
            lines.append('|'.join([ident,'duplex-matched-fixture','brics.u6ada',value['state'],code,str(value['elapsed']),'gres/gpu='+str(value['GPUs'])]))
        lines.append('90007|duplex-matched-fixture|brics.u6ada|COMPLETED|0:0|7200|gres/gpu=2')
        text='\n'.join(lines)
        ordered=[dict(lease=dict(job_id='90007',prior_GPU_seconds=134991,maximum_seconds=86400,
            rpc_recovery_authorization_sha256='fixture-grant'))]
        self.assertEqual(resume.allocation_carry(text,ordered,'90008'),149391)
        bad=text.replace('8285192|duplex-matched-fixture|brics.u6ada|FAILED|1:0',
            '8285192|duplex-matched-fixture|brics.u6ada|COMPLETED|0:0')
        with self.assertRaises(ValueError):resume.allocation_carry(bad,ordered,'90008')
        ordered[0]['lease']['prior_GPU_seconds']=44709
        with self.assertRaises(ValueError):resume.allocation_carry(text,ordered,'90008')


if __name__=='__main__':unittest.main()
