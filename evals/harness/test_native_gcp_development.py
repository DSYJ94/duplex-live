import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from portable.context.native_gcp_development import checked_upstream, collected, native_command, sha, speech_command


class GcpDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.args = SimpleNamespace(source=root/'source', source_commit='a'*40,
            proof=root/'proof', upstream=root/'recovery', asr_python=Path('/asr/python'),
            resource_id='resource1', recovery_unit='recovery-v6', recovery_invocation='invocation1',
            recovery_commit='b'*40, deadline_utc='2026-10-25T11:40:00Z')
        for folder in (self.args.proof, self.args.upstream):
            folder.mkdir()
        (self.args.proof/'report.json').write_text('{}')
        self.upstream = dict(phase='recovered_saved8_proof_complete',source_commit='b'*40,
            numerical_proof_passed=True,optimizer_updates_this_run=0,original_stage_updates=8,
            derived_proof=str(self.args.proof),children=[dict(phase='finalization',pid=12,actual_exit=0),
                dict(phase='independent_restore',pid=13,actual_exit=0)])
        (self.args.upstream/'controller.json').write_text(json.dumps(self.upstream))
        self.observation = dict(schema='dl-gcp-recovery-terminal-observation-v1',
            observed_unix=1000,resource_id='resource1',unit='recovery-v6',invocation_id='invocation1',
            actual_exit=0,main_pid=0,result='success',active_state='inactive',
            controller_sha256=sha(self.args.upstream/'controller.json'),
            proof_report_sha256=sha(self.args.proof/'report.json'))

    def test_requires_fresh_actual_same_resource_terminal_success(self):
        checked_upstream(self.args,self.observation,self.upstream,1001)
        for field,value in [('observed_unix',900),('observed_unix',1002),('actual_exit',1),
                ('actual_exit',False),('main_pid',50),('active_state','active'),
                ('resource_id','other'),('invocation_id','different'),('controller_sha256','0'*64)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                checked_upstream(self.args,dict(self.observation,**{field:value}),self.upstream,1001)

    def test_live_partial_failed_or_reused_restore_cannot_start_evaluation(self):
        for change in ('phase','exit','missing','same_pid','proof','updates'):
            value=copy.deepcopy(self.upstream)
            if change=='phase':value['phase']='independent_restore'
            elif change=='exit':value['children'][1]['actual_exit']=None
            elif change=='missing':value['children'].pop()
            elif change=='same_pid':value['children'][1]['pid']=12
            elif change=='proof':value['derived_proof']=str(self.args.proof/'other')
            else:value['optimizer_updates_this_run']=1
            with self.subTest(change=change),self.assertRaises(ValueError):
                checked_upstream(self.args,self.observation,value,1001)

    def test_commands_preserve_full_existing_collectors_and_recovered_proof_root(self):
        native=native_command(self.args,Path('/paired'))
        speech=speech_command(self.args,Path('/paired'),Path('/speech'),90,100)
        self.assertIn('portable.context.native_development',native)
        self.assertEqual(native[native.index('--reasoning32-candidate-proof')+1],str(self.args.proof))
        self.assertEqual(native[native.index('--gpu-mode')+1],'serial-one-gpu')
        self.assertEqual(native[native.index('--runtime-profile')+1],'container-default')
        self.assertIn('portable.context.native_speech_followup',speech)
        self.assertEqual(speech[speech.index('--controller-pid')+1],'90')
        self.assertEqual(speech[speech.index('--controller-birth')+1],'100')
        for command in (native,speech):
            for forbidden in ('--actual-temporal-fit','--reload','--resume-training','--role'):
                self.assertNotIn(forbidden,command)

    def outputs(self):
        paired=Path(self.directory.name)/'paired';speech=Path(self.directory.name)/'speech'
        paired.mkdir();speech.mkdir()
        native=dict(phase='complete',complete_paired20=True,source_commit='a'*40,
            gpu_mode='serial-one-gpu',proofs={'baseline520':'/baseline','candidate-reasoning32':str(self.args.proof)})
        audit=dict(phase='complete_speech20_only',complete20=True,source_commit='a'*40,
            paired_roles=['baseline520','candidate-reasoning32'])
        (paired/'execution.json').write_text(json.dumps(native))
        (speech/'controller.json').write_text(json.dumps(audit))
        return paired,speech,native,audit

    def test_collection_never_grants_behavior_or_adoption(self):
        paired,speech,_,_=self.outputs()
        result=collected(self.args,paired,speech)
        self.assertTrue(result['behavior_review_required'])
        self.assertFalse(result['behavior_approved'])
        self.assertFalse(result['model_adopted'])

    def test_partial_native_or_speech_and_wrong_candidate_are_rejected(self):
        paired,speech,native,audit=self.outputs()
        for change in ('native_partial','speech_partial','source','candidate','roles'):
            left,right=copy.deepcopy(native),copy.deepcopy(audit)
            if change=='native_partial':left['complete_paired20']=False
            elif change=='speech_partial':right['complete20']=False
            elif change=='source':right['source_commit']='c'*40
            elif change=='candidate':left['proofs']['candidate-reasoning32']='/wrong'
            else:right['paired_roles']=['baseline520']
            (paired/'execution.json').write_text(json.dumps(left));(speech/'controller.json').write_text(json.dumps(right))
            with self.subTest(change=change),self.assertRaises(ValueError):collected(self.args,paired,speech)


if __name__=='__main__':
    unittest.main()
