import copy,json,tempfile,unittest
from pathlib import Path
from training.personaplex.adaptive_policy import digest
from training.personaplex.adaptive_evidence import read
from training.personaplex.adaptive_handover import prepare,validate_plan,restore_identity,target_identity,native_restore_packet,all_native_restored
from training.personaplex.adaptive_native import publish_decision


class Handover(unittest.TestCase):
    def fixture(self,root):
        parent=root/'parent';back=root/'backup';target=root/'new';newback=root/'newbackup'
        folder=parent/'job-123'/'train';folder.mkdir(parents=True)
        checkpoint=parent/'checkpoints'/'step-0060.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'common-state')
        h=read_hash(checkpoint);metadata=dict(file=checkpoint.name,sha256=h,step=60,source='a'*40)
        Path(str(checkpoint)+'.sha256.json').write_text(json.dumps(metadata))
        terminal=dict(actual_new_step=60,training_complete=False,continuation_ready=True,checkpoint=str(checkpoint),checkpoint_sha256=h,source='a'*40)
        (folder/'terminal.json').write_text(json.dumps(terminal));(folder.parent/'host-terminal.json').write_text(json.dumps(dict(success=True,training=terminal)))
        (parent/'admission.json').write_text(json.dumps(dict(actual_CUDA=True,all_four_ranks=[{}]*4,total_new_updates=196)))
        destination=back/'step-0060';destination.mkdir(parents=True)
        files=[]
        for name,blob in [(checkpoint.name,checkpoint.read_bytes()),(checkpoint.name+'.sha256.json',Path(str(checkpoint)+'.sha256.json').read_bytes())]+[(checkpoint.name+'.rank-%04d.pt'%r,b'rank')for r in range(4)]:
            p=destination/name;p.write_bytes(blob);files.append(dict(file=name,sha256=read_hash(p),bytes=len(blob)))
        (destination/'backup.json').write_text(json.dumps(dict(complete=True,source=str(checkpoint),files=files,step=60)))
        provider=dict(JobId='123',JobState='COMPLETED',Account='brics.u6ada')
        return (parent,'123',back,target,newback,'b'*40,provider)

    def test_live_failed_or_complete_parent_never_admits(self):
        for state in ('RUNNING','PENDING','FAILED','TIMEOUT','UNKNOWN'):
            with self.assertRaises(ValueError):prepare('/absent','123','/absentbackup','/new','/newbackup','b'*40,dict(JobId='123',JobState=state,Account='brics.u6ada'))

    def test_exact_latest_parent_and_independent_backup_binding(self):
        with tempfile.TemporaryDirectory()as d:
            args=self.fixture(Path(d));plan=prepare(*args);self.assertEqual(plan['step'],60)
            self.assertEqual(validate_plan(plan,args[-1]),plan)
            changed=copy.deepcopy(plan);changed['step']=59
            with self.assertRaises(ValueError):validate_plan(changed,args[-1])
            newer=args[0]/'checkpoints'/'step-0061.pt';newer.write_bytes(b'newer')
            with self.assertRaises(ValueError):prepare(*args)
            newer.unlink();metadata=args[2]/'step-0060'/'step-0060.pt.sha256.json';metadata.write_text('{}')
            with self.assertRaises(ValueError):prepare(*args)

    def test_subsequent_resume_is_latest_successful_and_provider_terminal(self):
        from training.personaplex.adaptive_handover import validate_resume
        with tempfile.TemporaryDirectory()as d:
            args=self.fixture(Path(d));checkpoint=args[0]/'checkpoints'/'step-0060.pt'
            plan=dict(root=str(args[0]),backup=str(args[2]),source='a'*40,step=49)
            self.assertEqual(validate_resume(plan,checkpoint,args[-1])['actual_new_step'],60)
            with self.assertRaises(ValueError):validate_resume(plan,checkpoint,dict(args[-1],JobState='RUNNING'))
            with self.assertRaises(ValueError):validate_resume(dict(plan,step=60),checkpoint,args[-1])
            newer=checkpoint.with_name('adaptive-step-0061.pt');newer.write_bytes(b'newer')
            with self.assertRaises(ValueError):validate_resume(plan,checkpoint,args[-1])

    def test_math_recipe_data_cannot_change_during_source_transition(self):
        parent=dict(source='a'*40,recipe={'LR':1e-5},data='fixed',capture_provenance=dict(current_training_source='a'*40,producer='c'*40))
        current=copy.deepcopy(parent);current['source']='b'*40;current['capture_provenance']['current_training_source']='b'*40
        plan=dict(parent_source='a'*40,source='b'*40,schema='dl-adaptive-handover-v1',parent_checkpoint_sha256='d'*64,step=60,parent_backup_sha256='e'*64,policy_sha256='f'*64)
        before=copy.deepcopy(parent);self.assertEqual(restore_identity(parent,current,plan),parent);self.assertEqual(parent,before)
        self.assertEqual(target_identity(current,plan)['adaptive_lineage']['parent_step'],60)
        for key,value in (('recipe',{'LR':2e-5}),('data','changed')):
            altered=copy.deepcopy(current);altered[key]=value
            with self.assertRaises(ValueError):restore_identity(parent,altered,plan)

    def test_every_native_restore_field_and_every_rank_are_required(self):
        flags=dict(optimizer_exact=True,rng_exact=True,recipe_exact=True,diagnostics_exact=True,fingerprint_exact=True,common_adam_exact=True)
        def packet(rank,**changes):
            return native_restore_packet(source='b'*40,step=60,execution='new'+str(rank),pid=200+rank,rank=rank,value=dict(writer_execution='old'+str(rank),writer_pid=100+rank),**dict(flags,**changes))
        rows=[packet(r)for r in range(4)];self.assertTrue(all_native_restored(rows,60)['passed'])
        reused=native_restore_packet(source='b'*40,step=60,execution='fresh',pid=100,rank=0,value=dict(writer_execution='prior-allocation',writer_pid=100),**flags)
        self.assertTrue(reused['native_fresh_restore_passed'])
        with self.assertRaises(ValueError):native_restore_packet(source='b'*40,step=60,execution='same',pid=200,rank=0,value=dict(writer_execution='same',writer_pid=100),**flags)
        for field in flags:
            with self.assertRaises(ValueError):packet(0,**{field:False})
        for bad in (rows[:3],rows[:3]+[rows[0]]):
            with self.assertRaises(ValueError):all_native_restored(bad,60)

    def test_decision_publishes_and_backs_up_before_receipt_or_learning(self):
        events=[]
        class Controller:
            state={'enabled':False}
            def enable(self,c):events.append('enable');self.state['enabled']=True
            def evaluate(self,r,c):events.append('decision');return dict(applied=True)
        def checkpoint(step,name):events.append('checkpoint');return Path(name)
        def backup(path):events.append('backup');return Path('backup.json')
        publish_decision(Controller(),checkpoint=checkpoint,backup_checkpoint=backup,context_value={},request={},step=60,write_receipt=lambda r:events.append('receipt'))
        self.assertEqual(events,['enable','decision','checkpoint','backup','receipt'])
        events.clear()
        def failure(path):events.append('backup-failed');raise OSError('disk full')
        with self.assertRaises(OSError):publish_decision(Controller(),checkpoint=checkpoint,backup_checkpoint=failure,context_value={},request={},step=60,write_receipt=lambda r:events.append('receipt'))
        self.assertNotIn('receipt',events)


def read_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':unittest.main()


class LauncherStages(unittest.TestCase):
    def test_success_and_failed_native_restore_order(self):
        from training.personaplex.expanded_launch import adaptive_stages
        for success in (True,False):
            with tempfile.TemporaryDirectory()as d:
                root=Path(d);folder=root/'job-new';checkpoint=root/'checkpoints'/'adaptive-step-0060.pt';checkpoint.parent.mkdir()
                Path(str(checkpoint)+'.sha256.json').write_text(json.dumps(dict(step=60,sha256='a'*64)))
                calls=[]
                def execute(module,args,**kwargs):
                    self.assertEqual(kwargs,dict(gpu=True,parallel=True));calls.append(args[0])
                    if args[0]=='transition':
                        path=folder/'transition'/'transition-terminal.json';path.parent.mkdir(parents=True)
                        path.write_text(json.dumps(dict(passed=True,new_updates=0,checkpoint=str(checkpoint))))
                    if args[0]=='verify-transition':
                        path=folder/'verify-transition'/'verify-terminal.json';path.parent.mkdir(parents=True)
                        path.write_text(json.dumps(dict(passed=success,new_updates=0,step=60,checkpoint_sha256='a'*64)))
                if success:
                    adaptive_stages(execute,[],folder,root,dict(parent_checkpoint='/original/step-0060.pt'))
                    self.assertEqual(calls,['transition','verify-transition','train-adaptive'])
                else:
                    with self.assertRaises(ValueError):adaptive_stages(execute,[],folder,root,dict(parent_checkpoint='/original/step-0060.pt'))
                    self.assertEqual(calls,['transition','verify-transition'])
    def test_external_resume_checkpoint_rejected_before_any_model(self):
        from training.personaplex.expanded_launch import adaptive_stages
        with self.assertRaises(ValueError):adaptive_stages(lambda *a,**k:self.fail('No model should start'),[],Path('/output/job'),Path('/output'),{},Path('/original/step-0060.pt'))


class RunnerAdmission(unittest.TestCase):
    def test_adaptive_stages_cannot_fall_through_to_ordinary_training(self):
        from types import SimpleNamespace
        from training.personaplex.expanded_runner import run
        for stage in ('transition','verify-transition','train-adaptive'):
            with self.assertRaisesRegex(ValueError,'handover plan'):run(SimpleNamespace(stage=stage,adaptive_plan=None))
        with self.assertRaisesRegex(ValueError,'explicit adaptive'):run(SimpleNamespace(stage='train',adaptive_plan=Path('/present')))
