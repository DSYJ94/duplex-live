import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex.matched_generation import construct_engine,completed_states,load_inference_adapter,DATA_SHA,ORIGIN,g


class NativeGenerationBoundary(unittest.TestCase):
    def test_public_base_frozen_before_narrow_ownership_check_then_exact_state_loaded(self):
        class PublicModel:
            trainable=True
            def requires_grad_(self,value):self.trainable=value;return self
            def eval(self):return self
        lm=PublicModel();saved={'named-adapter':'fixture-state'};events=[]
        def configure(model,capacity):
            if model.trainable:raise ValueError('Unexpected frozen-base gradient ownership')
            self.assertEqual(capacity,'narrow');events.append('configured');model.trainable=True
        def load(model,state):self.assertIs(state,saved);events.append('exact-state-loaded')
        with patch('training.personaplex.matched_native.configure',side_effect=configure),patch('training.personaplex.matched_state.load_adapter',side_effect=load):
            self.assertIs(load_inference_adapter(lm,'narrow',saved),lm)
        self.assertEqual(events,['configured','exact-state-loaded']);self.assertFalse(lm.trainable)

    def test_adapter_loader_runs_before_constructor_graph_capture_and_is_restored(self):
        events=[];original=lambda:events.append('unadapted')
        module=SimpleNamespace(load_model=original)
        def loader():events.append('adapter-loaded');return 'actual-adapted-model'
        def Engine(**kwargs):
            lm=module.load_model();events.append(('graph-captured',lm));return lm
        module.Engine=Engine
        self.assertEqual(construct_engine(module,loader),'actual-adapted-model')
        self.assertEqual(events,['adapter-loaded',('graph-captured','actual-adapted-model')]);self.assertIs(module.load_model,original)
        def fail(**kwargs):raise RuntimeError('constructor failed')
        module.Engine=fail
        with self.assertRaises(RuntimeError):construct_engine(module,loader)
        self.assertIs(module.load_model,original)

    def test_partial_failed_or_wrong_source_training_cannot_enter_evaluation(self):
        for completed,code,source in ((False,0,'trainer'),(True,1,'trainer'),(True,0,'different')):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);execution=root/'execution-123';execution.mkdir()
                (root/'phase-result.json').write_text(json.dumps(dict(source=source,all_four_complete=completed,partial_lineages=[])))
                (execution/'terminal.json').write_text(json.dumps(dict(source=source,actual_exit_code=code)))
                with self.assertRaisesRegex(ValueError,'All four completed'):completed_states(root,'trainer')

    def test_four_actual_final_state_records_including_inherited_completed_arm(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ex=root/'execution-123';ex.mkdir();entries=[];original=[];restore_files=[]
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return str(path)
            write(root/'phase-result.json',dict(source='trainer',all_four_complete=True,partial_lineages=[]))
            write(ex/'terminal.json',dict(source='trainer',actual_exit_code=0))
            for base in ('personaplex','context-spanning'):
                for cap in ('narrow','broad'):
                    folder=root/base/cap;folder.mkdir(parents=True)
                    checkpoint=folder/'epoch-04.pt';checkpoint.write_bytes(b'fixture full state');digest=g.sha(checkpoint)
                    reference=dict(source='trainer',checkpoint_sha256=digest,step=196,execution_id='training-process')
                    write(Path(str(checkpoint)+'.native-reference.json'),reference)
                    restore=folder/'final-restore/restore.json';restore_files.append(restore)
                    write(restore,dict(source='trainer',lineage_source=ORIGIN,base=base,capacity=cap,data_report_sha256=DATA_SHA,
                        checkpoint_sha256=digest,passed=True,actual_CUDA=True,all_Adam_RNG_OPAL_fields_exact=True,
                        native_logits_exact=True,execution_id='distinct-restore-process',reference_execution_id='training-process',restored_training_step=196))
                    rt=write(folder/'restore-child.json',dict(source='trainer',base=base,capacity=cap,stage='restore',
                        execution_id='distinct-restore-process',actual_exit_code=0))
                    tt=write(folder/'train-child.json',dict(source='trainer',base=base,capacity=cap,stage='train',
                        execution_id='training-process',actual_exit_code=0))
                    # One arm is inherited from the preceding completed window;
                    # others have their successful terminal in this window.
                    terminal=folder/('previous-terminal.json' if base=='personaplex' and cap=='narrow' else 'train/training-terminal.json')
                    write(terminal,dict(source='trainer',total_lineage_updates=196,checkpoint_sha256=digest,
                        final_checkpoint=str(checkpoint),completed=True,complete_epochs=4))
                    entries.append(dict(base=base,capacity=cap,checkpoint=str(checkpoint),checkpoint_sha256=digest,
                        restore=str(restore),restore_sha256=g.sha(restore),restore_terminal=rt,
                        final_restore=str(restore),final_restore_terminal=rt,training_child_terminal=tt,
                        previous_training_terminal=str(terminal),previous_training_child_terminal=tt))
                    original.append(dict(base=base,capacity=cap))
            write(root/'joint-admission.json',dict(source='trainer',data_report_sha256=DATA_SHA,entries=entries))
            with patch('training.personaplex.matched_generation.preflight',return_value=dict(lineages=original)):
                self.assertEqual(len(completed_states(root,'trainer')),4)
                bad=restore_files[-1];value=json.loads(bad.read_text());value['native_logits_exact']=False;write(bad,value)
                with self.assertRaises(ValueError):completed_states(root,'trainer')


if __name__=='__main__':unittest.main()
