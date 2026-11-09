import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.personaplex import matched_training_only as host,matched_continue_policy as policy
from training.personaplex.matched_reuse import ORIGIN,DATA_SHA


class TrainingOnlyHost(unittest.TestCase):
    def original(self):return dict(lineages=[dict(base=b,capacity=c,original_checkpoint='/old/'+b+'/'+c+'/init.pt',checkpoint_sha256=b+c) for b in ('personaplex','context-spanning') for c in ('narrow','broad')])

    def test_actual_allocation_carry_and_unlisted_gpu_job_fail_closed(self):
        text='8216374|capture|brics.u6ada|COMPLETED|327|gres/gpu=1|\n8229989|duplex-matched-train-v1|brics.u6ada|FAILED|951|gres/gpu=2|\n90000|duplex-matched-training-only-v1|brics.u6ada|RUNNING|2|gres/gpu=2|\n'
        with patch.object(host.subprocess,'check_output',return_value=text):self.assertEqual(host.prior_usage('90000'),2229)
        with patch.object(host.subprocess,'check_output',return_value=text):
            with self.assertRaisesRegex(ValueError,'uncounted'):host.prior_usage('90001')

    def test_orchestration_restores_all4_before_training_and_preserves_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);execution=root/'execution';execution.mkdir();events=[]
            class Children:
                def parallel(self,action):
                    for base in ('personaplex','context-spanning'):action(base)
                def run(self,base,cap,stage,folder,extra=(),label=None):
                    folder.mkdir();events.append((base,cap,stage,label))
                    terminal=execution/(base+'-'+cap+'-'+(label or stage)+'-terminal.json');terminal.write_text('{"actual_exit_code":0}')
                    if stage=='restore':(folder/'restore.json').write_text('{"fixture":true}')
                    elif stage=='train':
                        self_test.assertEqual(sum(e[2]=='restore' and e[3] is None for e in events),4)
                        final=dict(source='fixture',completed=False,actual_new_updates=8,total_lineage_updates=8,checkpoint=str(folder/'window-state.pt'),checkpoint_sha256='fixture-state')
                        (folder/'training-window-terminal.json').write_text(json.dumps(final))
                    else:raise AssertionError('No repeated admission allowed')
                    return terminal
            self_test=self;rows=host.initial_plan(self.original(),root,execution)
            partial=host.execute(Children(),rows,root,root/'joint.json','fixture')
            self.assertEqual(len(partial),2);self.assertTrue(all(p['total_lineage_updates']==8 for p in partial))
            self.assertEqual(sum(e[2]=='train' for e in events),2)
            self.assertEqual(sum(e[3]=='final-restore' for e in events),2)

    def test_initial_controller_refuses_pending_grant_before_creating_state(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory,patch.object(host,'ROOT',Path(directory)/'no-state'),patch.object(host.g,'EXTENSION',Path(directory)/'missing'),patch.object(host.subprocess,'check_output',return_value='source'):
            with self.assertRaises(FileNotFoundError):host.main(SimpleNamespace(source='source',inputs_sha=DATA_SHA))
            self.assertFalse(host.ROOT.exists())


class FreshRestoreBinding(unittest.TestCase):
    def fixture(self,root):
        checkpoint=root/'init.pt';checkpoint.write_bytes(b'fixture')
        row=dict(base='personaplex',capacity='broad',checkpoint=str(checkpoint),checkpoint_sha256='state',restore=str(root/'restore.json'),restore_terminal=str(root/'terminal.json'))
        reference=dict(checkpoint_sha256='state',source=ORIGIN,execution_id='original',step=0)
        Path(str(checkpoint)+'.native-reference.json').write_text(json.dumps(reference))
        value=dict(source='new',lineage_source=ORIGIN,base='personaplex',capacity='broad',data_report_sha256=DATA_SHA,checkpoint_sha256='state',passed=True,actual_CUDA=True,all_Adam_RNG_OPAL_fields_exact=True,native_logits_exact=True,execution_id='new-restore',reference_execution_id='original',restored_training_step=0)
        Path(row['restore']).write_text(json.dumps(value));row['restore_sha256']=policy.g.sha(row['restore'])
        Path(row['restore_terminal']).write_text(json.dumps(dict(actual_exit_code=0,source='new',base='personaplex',capacity='broad',stage='restore',execution_id='new-restore')))
        return row,dict(original_checkpoint=str(checkpoint),checkpoint_sha256='state')

    def test_distinct_native_restore_bound_to_original_state(self):
        with tempfile.TemporaryDirectory() as directory:
            row,original=self.fixture(Path(directory));self.assertEqual(policy.restored_entry(row,'new',DATA_SHA,original),0)
            original['checkpoint_sha256']='swapped'
            with self.assertRaises(ValueError):policy.restored_entry(row,'new',DATA_SHA,original)

    def test_claimed_partial_or_missing_fingerprint_or_wrong_terminal_rejected(self):
        for key,value in [('restored_training_step',1),('native_logits_exact',False),('lineage_source','fake'),('execution_id','original')]:
            with tempfile.TemporaryDirectory() as directory:
                row,original=self.fixture(Path(directory));path=Path(row['restore']);r=json.loads(path.read_text());r[key]=value;path.write_text(json.dumps(r));row['restore_sha256']=policy.g.sha(path)
                with self.subTest(key=key),self.assertRaises(ValueError):policy.restored_entry(row,'new',DATA_SHA,original)
        with tempfile.TemporaryDirectory() as directory:
            row,original=self.fixture(Path(directory));Path(row['restore_terminal']).write_text('{"actual_exit_code":143}')
            with self.assertRaises(ValueError):policy.restored_entry(row,'new',DATA_SHA,original)

    def test_real_partial_step_requires_successful_previous_training_and_native_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row,original=self.fixture(root)
            reference=Path(str(row['checkpoint'])+'.native-reference.json')
            ref=json.loads(reference.read_text());ref.update(source='new',step=98,execution_id='actual-train');reference.write_text(json.dumps(ref))
            restore=Path(row['restore']);value=json.loads(restore.read_text());value.update(restored_training_step=98,reference_execution_id='actual-train');restore.write_text(json.dumps(value));row['restore_sha256']=policy.g.sha(restore)
            prior=root/'window-terminal.json';prior.write_text(json.dumps(dict(source='new',total_lineage_updates=98,checkpoint_sha256='state',checkpoint=row['checkpoint'],completed=False)))
            child=root/'training-child.json';child.write_text(json.dumps(dict(source='new',actual_exit_code=0,base='personaplex',capacity='broad',stage='train',execution_id='actual-train')))
            row.update(previous_training_terminal=str(prior),previous_training_child_terminal=str(child))
            self.assertEqual(policy.restored_entry(row,'new',DATA_SHA,original),98)
            v=json.loads(child.read_text());v['actual_exit_code']=143;child.write_text(json.dumps(v))
            with self.assertRaisesRegex(ValueError,'actual successful checkpointed'):policy.restored_entry(row,'new',DATA_SHA,original)


class CompletePolicyBudget(unittest.TestCase):
    def fixture(self,root):
        from types import SimpleNamespace
        entries=[];old=[]
        for base in ('personaplex','context-spanning'):
            for cap in ('narrow','broad'):
                folder=root/base/cap;folder.mkdir(parents=True)
                row,original=FreshRestoreBinding().fixture(folder);row.update(base=base,capacity=cap,training_terminal=str(folder/'not-yet-trained.json'))
                path=Path(row['restore']);value=json.loads(path.read_text());value.update(base=base,capacity=cap);path.write_text(json.dumps(value));row['restore_sha256']=policy.g.sha(path)
                path=Path(row['restore_terminal']);value=json.loads(path.read_text());value.update(base=base,capacity=cap);path.write_text(json.dumps(value))
                entries.append(row);old.append(dict(original,base=base,capacity=cap,estimated_complete196_seconds=3600.))
        path=root/'joint.json';path.write_text(json.dumps(dict(schema='dl-moshi-training-only-joint-v1',source='new',lineage_source=ORIGIN,data_report_sha256=DATA_SHA,original_joint_packet_sha256=policy.PACKET_SHA,entries=entries)))
        return SimpleNamespace(source='new',inputs_sha=DATA_SHA,joint_admission=str(path),joint_admission_sha=policy.g.sha(path),lease='fixture'),dict(lineages=old)

    def test_complete_matched_plan_and_carried_budget_are_required_before_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            args,old=self.fixture(Path(directory));lease=dict(continuation_authorization_sha256='fixture',prior_GPU_seconds=2229,start_unix=1000)
            with patch.object(policy,'preflight',return_value=old),patch.object(policy.g,'check',return_value=lease),patch.object(policy.g,'phase_limit',return_value=112*3600),patch.object(policy.time,'time',return_value=1000):
                report=policy.evaluate(args)
                self.assertEqual(report['required_GPU_seconds'],36000)
                self.assertEqual(report['available_GPU_seconds'],112*3600-2229)
                self.assertEqual(set(report['actual_verified_steps'].values()),{0})
                lease.pop('continuation_authorization_sha256')
                with self.assertRaisesRegex(ValueError,'recorded amendment'):policy.evaluate(args)
                lease['continuation_authorization_sha256']='fixture';lease['prior_GPU_seconds']=112*3600-100
                with self.assertRaisesRegex(RuntimeError,'remaining matched schedule'):policy.evaluate(args)
                path=Path(args.joint_admission);packet=json.loads(path.read_text());packet['entries'].pop();path.write_text(json.dumps(packet));args.joint_admission_sha=policy.g.sha(path)
                with self.assertRaisesRegex(ValueError,'four-lineage'):policy.evaluate(args)


if __name__=='__main__':unittest.main()
