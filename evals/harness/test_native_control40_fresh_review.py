"""CPU artifact fixtures, never evidence of native behavior or GPU approval."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch

from evals.moshi_model_download import sha256
from evals.native_control40_fresh_review import SCHEMA,ROLES,validate_review,validate,implementations
from evals.native_fresh_inputs_v2 import PLAN_SHA256
from portable.context.native_development import artifact_hashes,CASE_FILES,PARITY_FILES,compare_native


class PortableFreshReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.proof=root/'pilot-v2/training';self.proof.mkdir(parents=True)
        self.baseline=root/'context-training-proof-7510331';self.baseline.mkdir()
        self.pair=root/'native20';self.speech=root/'speech20'
        self.pair.mkdir();self.speech.mkdir()
        plan=Path(__file__).resolve().parents[1]/'fixtures/native-v2-development-20261019.json'
        self.cases=json.loads(plan.read_text())['cases']
        self.identity=dict(checkpoint_sha256='candidate',training_state_sha256='state',reload_report_sha256='reload')
        self.parent=dict(schema='dl-native-control40-retained154-stream49-proof-v1',
            updates_this_run=8,adam_parameter_steps_after=[528],additional_native_v2=dict(complete24_per8=True),
            warm_start=dict(proof=str(self.baseline),checkpoint_sha256='baseline'),voice_sha256='same-voice')
        self.native=dict(schema='dl-runpod-paired-native10-v1',phase='complete',complete_paired20=True,
            gpu_mode='serial-one-gpu',runtime_profile='container-default',runtime_comparison_only=False,
            optimizer_updates=0,provider_credential_transferred=False,gpus=['actual-fixture-device'],
            proofs=dict(zip(ROLES,map(str,(self.baseline,self.proof)))),plan_sha256=sha256(plan),
            source_commit='a'*40,execution_implementation_sha256={'fixture-code':'digest'})
        self.audio=dict(schema='dl-runpod-paired20-independent-speech-followup-v1',
            phase='complete_speech20_only',complete20=True,paired_source='a'*40,source_commit='a'*40,
            gpu_mode='serial-one-gpu',paired_roles=list(ROLES),reports_sha256={},
            speech_implementation_sha256={'fixture-code':'digest'})
        self.review=dict(schema=SCHEMA,candidate_proof=str(self.proof),
            candidate_checkpoint_sha256='candidate',candidate_training_state_sha256='state',
            independent_reload_report_sha256='reload',evaluation_plan_sha256=sha256(plan),
            fresh_plan_sha256=PLAN_SHA256,all10_native_cases_reviewed=True,all4_ordinary_core_correct=True,
            fresh_evaluation_authorized=True,checkpoint_adopted=False,native_asr_placement='shared-gpu-evaluation',
            native_pair_root=str(self.pair),independent_speech_root=str(self.speech),
            native_execution_source_commit='a'*40,actor_execution_sha256={},cases=[])
        self.actors={};self.audits={}
        for role in ROLES:
            actor_root=self.pair/role;actor_root.mkdir();(self.speech/role).mkdir()
            for name in ('parity-cold','parity-intervening','parity-reused'):
                folder=actor_root/name;folder.mkdir()
                (folder/'packet-stream.wav').write_bytes(b'fixture-pcm')
                (folder/'events.jsonl').write_text('{"text_token":1}\n')
                self.write(folder/'report.json',dict(text='fixture',
                    scenario=self.cases[1 if name=='parity-intervening' else 0]['scenario']))
                torch.save(dict(delayed_inputs=torch.tensor([1,2])),folder/'student-history.pt')
            actor=dict(schema='dl-native-paired-development-actor-v1',phase='complete',complete10=True,
                parity=compare_native(actor_root/'parity-cold',actor_root/'parity-reused'),
                parity_artifact_sha256={name:artifact_hashes(actor_root/name,PARITY_FILES)
                    for name in ('parity-cold','parity-intervening','parity-reused')},
                source_commit='a'*40,gpu_mode='serial-one-gpu',role=role,optimizer_updates=0,
                provider_credential_present=False,runtime_profile='container-default',assigned_main_gpu=0,
                assigned_asr_gpu=0,proof=self.native['proofs'][role],cases=[])
            rows=[]
            for i,case in enumerate(self.cases):
                folder=actor_root/('case-%02d'%i);folder.mkdir()
                (folder/'model.wav').write_bytes(b'fixture-audio'+role.encode()+bytes([i]))
                (folder/'user_in.wav').write_bytes(b'fixture-user'+bytes([i]))
                raw=dict(audio_sha256=sha256(folder/'model.wav'),scenario=case['scenario'],text='Fixture words',
                    scheduled_dispatch=False,word_or_acoustic_tokens_forced=False,response_start_forced=False,
                    real_user_samples_processed=24000,real_user_samples_expected=24000,
                    distinct_lm_asr_gpu_processes=False,separate_native_and_asr_processes=True,
                    asr_environment=dict(reference_transcript_supplied=False,assigned_visible_gpu='0'),
                    loading=dict(adapter=dict(checkpoint_sha256='baseline' if role=='baseline520' else 'candidate',
                        voice_codes_sha256='same-voice',proof=self.native['proofs'][role],zero=False,disabled=False,
                        installed_before_native_graphs=True),native_asr_resource_placement=dict(
                        profile='shared-gpu-evaluation',distinct_physical_gpus=False,separate_native_and_asr_processes=True)))
                self.write(folder/'report.json',raw)
                actor['cases'].append(dict(index=i,exit_code=0,artifact_sha256=artifact_hashes(folder,CASE_FILES)))
                rows.append(dict(run=str(folder),wav_sha256=sha256(folder/'model.wav'),
                    model_text='Fixture words',asr_text='Fixture words',source_kind='native',
                    asr_segments=[dict(words=[dict(start=0.,end=1.,text='Fixture words')])],
                    asr_time_bounds_valid=True,asr_word_bounds_valid=True,word_starts_within_recorded_audio=True,
                    wer_against_model_text=0.))
                if role=='candidate-control40':self.review['cases'].append(dict(index=i,
                    run_report_sha256=sha256(folder/'report.json'),audio_sha256=sha256(folder/'model.wav'),
                    behavior_and_receipts_reviewed=True,speech_quality_issue_reviewed=True,
                    behavior_verdict='pass',ordinary_core_correct=True,reason='CPU fixture only'))
            self.actors[role]=actor
            self.audits[role]=dict(schema='dl-native-speech-audit-v1',device='cuda',results=rows)
        self.seal()
        self.patcher=patch('evals.native_control40_fresh_review.implementations',return_value={'fixture-code':'digest'})
        self.patcher.start();self.addCleanup(self.patcher.stop)

    @staticmethod
    def write(path,data):path.write_text(json.dumps(data))

    def seal(self):
        self.write(self.pair/'execution.json',self.native)
        self.review['native_pair_execution_sha256']=sha256(self.pair/'execution.json')
        self.audio['paired_execution_sha256']=self.review['native_pair_execution_sha256']
        for role in ROLES:
            self.write(self.pair/role/'execution.json',self.actors[role])
            self.review['actor_execution_sha256'][role]=sha256(self.pair/role/'execution.json')
            self.write(self.speech/role/'report.json',self.audits[role])
            self.audio['reports_sha256'][role]=sha256(self.speech/role/'report.json')
        self.write(self.speech/'controller.json',self.audio)
        self.review['independent_speech_controller_sha256']=sha256(self.speech/'controller.json')

    def check(self):return validate_review(self.review,self.proof,self.parent,self.identity)

    def edit_raw(self,role,index,mutate):
        folder=self.pair/role/('case-%02d'%index);path=folder/'report.json'
        raw=json.loads(path.read_text());mutate(raw);self.write(path,raw)
        self.actors[role]['cases'][index]['artifact_sha256']=artifact_hashes(folder,CASE_FILES)
        if role=='candidate-control40':self.review['cases'][index]['run_report_sha256']=sha256(path)
        self.seal()

    def test_complete_non_slurm_candidate_authorizes_measurement_only(self):
        self.assertEqual(self.proof.name,'training')
        self.assertFalse(self.check()['checkpoint_adopted'])

    def test_partial_review_restore_plan_or_adoption_rejected(self):
        for key,value in [('candidate_checkpoint_sha256','other'),('candidate_training_state_sha256','other'),
                ('independent_reload_report_sha256','other'),('fresh_plan_sha256','other'),
                ('evaluation_plan_sha256','other'),('all10_native_cases_reviewed',False),
                ('all4_ordinary_core_correct',False),('fresh_evaluation_authorized',False),('checkpoint_adopted',True)]:
            with self.subTest(key=key):
                saved=self.review[key];self.review[key]=value
                with self.assertRaises(ValueError):self.check()
                self.review[key]=saved
        self.review['cases'].pop()
        with self.assertRaises(ValueError):self.check()

    def test_incomplete_actor_or_speech_cannot_pass(self):
        for role in ROLES:
            saved=copy.deepcopy(self.actors[role]);self.actors[role]['cases'].pop();self.seal()
            with self.subTest(role=role),self.assertRaises(ValueError):self.check()
            self.actors[role]=saved;self.seal()
            row=self.audits[role]['results'].pop();self.seal()
            with self.subTest(role=role),self.assertRaises(ValueError):self.check()
            self.audits[role]['results'].append(row);self.seal()

    def test_changed_source_physical_resource_or_speech_binding_rejected(self):
        for target,key,value in [(self.native,'execution_implementation_sha256',{}),
                (self.native,'runtime_profile','archive-tf32-high-v1'),(self.native,'gpus',['0','1']),
                (self.native,'gpu_mode','paired-two-gpu'),(self.audio,'source_commit','b'*40),
                (self.audio,'speech_implementation_sha256',{})]:
            saved=target[key];target[key]=value;self.seal()
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            target[key]=saved;self.seal()
        self.audio['paired_execution_sha256']='changed';self.write(self.speech/'controller.json',self.audio)
        self.review['independent_speech_controller_sha256']=sha256(self.speech/'controller.json')
        with self.assertRaises(ValueError):self.check()

    def test_baseline_checkpoint_must_be_the_actual_parent(self):
        self.edit_raw('baseline520',0,lambda r:r['loading']['adapter'].update(checkpoint_sha256='candidate'))
        with self.assertRaises(ValueError):self.check()

    def test_tenth_candidate_case_requires_semantic_and_audio_review(self):
        for key,value in [('behavior_and_receipts_reviewed',False),('speech_quality_issue_reviewed',False),
                ('behavior_verdict','unreviewed'),('reason','')]:
            saved=self.review['cases'][9][key];self.review['cases'][9][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.check()
            self.review['cases'][9][key]=saved

    def test_silence_cannot_vacuously_pass_ordinary_speech(self):
        row=self.audits['candidate-control40']['results'][0]
        row.update(asr_text='',asr_segments=[],wer_against_model_text=0.);self.seal()
        with self.assertRaises(ValueError):self.check()

    def test_nan_or_regressed_ordinary_speech_rejected(self):
        row=self.audits['candidate-control40']['results'][3]
        for value in (float('nan'),float('inf'),-.01,.081):
            row['wer_against_model_text']=value;self.seal()
            with self.subTest(value=value),self.assertRaises(ValueError):self.check()

    def test_diagnostic_forcing_or_reference_transcript_rejected(self):
        rawpath=self.pair/'candidate-control40/case-09/report.json'
        original=json.loads(rawpath.read_text())
        mutations=[lambda r:r.update(scheduled_dispatch=True),
            lambda r:r.update(response_start_forced=True),lambda r:r.update(word_or_acoustic_tokens_forced=True),
            lambda r:r['loading'].update(native_sampling_diagnostic={'enabled':True}),
            lambda r:r['loading']['adapter'].update(runtime_precision_diagnostic={'enabled':True}),
            lambda r:r['asr_environment'].update(reference_transcript_supplied=True),
            lambda r:r['asr_environment'].update(assigned_visible_gpu='1')]
        for mutate in mutations:
            self.write(rawpath,original);self.edit_raw('candidate-control40',9,mutate)
            with self.assertRaises(ValueError):self.check()

    def test_altered_baseline_or_candidate_audio_rejected(self):
        for role in ROLES:
            path=self.pair/role/'case-08/model.wav';original=path.read_bytes();path.write_bytes(b'altered')
            with self.subTest(role=role),self.assertRaises(ValueError):self.check()
            path.write_bytes(original)

    def test_different_user_pcm_even_with_rehashed_actor_rejected(self):
        folder=self.pair/'baseline520/case-09';(folder/'user_in.wav').write_bytes(b'other question')
        self.actors['baseline520']['cases'][9]['artifact_sha256']=artifact_hashes(folder,CASE_FILES);self.seal()
        with self.assertRaises(ValueError):self.check()

    def test_saved_parity_true_cannot_hide_changed_native_tensor(self):
        folder=self.pair/'candidate-control40/parity-reused'
        torch.save(dict(delayed_inputs=torch.tensor([1,3])),folder/'student-history.pt')
        self.actors['candidate-control40']['parity_artifact_sha256']['parity-reused']=artifact_hashes(folder,PARITY_FILES)
        self.seal()
        with self.assertRaises(RuntimeError):self.check()

    def test_intervening_artifact_is_bound_even_when_cold_reused_match(self):
        (self.pair/'baseline520/parity-intervening/events.jsonl').write_text('{"text_token":2}\n')
        with self.assertRaises(ValueError):self.check()

    def test_actual_proof_lock_and_strict_restore_precede_review(self):
        restored=dict(identity=self.identity,report_sha256='not-the-parent-report')
        self.write(self.proof/'report.json',self.parent);restored['report_sha256']=sha256(self.proof/'report.json')
        with patch('portable.context.native_development.restored_control40',return_value=restored) as strict:
            self.assertTrue(validate(self.review,self.proof,self.baseline)['fresh_evaluation_authorized'])
            with self.assertRaises(ValueError):validate(self.review,self.proof,self.proof.parent/'other')
            self.assertEqual(strict.call_count,2)
        with patch('portable.context.native_development.restored_control40',side_effect=ValueError('missing restore')):
            with self.assertRaises(ValueError):validate(self.review,self.proof,self.proof)

    def test_public_router_keeps_new_schema_out_of_legacy_numeric_guard(self):
        with patch('evals.native_fresh_review.committed_document',return_value=self.review), \
                patch('evals.native_control40_fresh_review.validate',return_value=self.review) as portable:
            from evals.native_fresh_review import validate as route
            self.assertTrue(route('committed-review',self.proof,self.proof)['fresh_evaluation_authorized'])
            portable.assert_called_once_with(self.review,self.proof.resolve(),self.proof)


class ImplementationIdentityTests(unittest.TestCase):
    def test_real_committed_file_and_malformed_commit(self):
        root=Path(__file__).resolve().parents[2]
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        name='evals/moshi_model_download.py'
        self.assertEqual(implementations(commit,[name]),{name:sha256(root/name)})
        for bad in ('main','abc','x'*40):
            with self.assertRaises(ValueError):implementations(bad,[name])

    def test_speech_source_rejects_dirty_checkout_and_records_all_imported_files(self):
        from portable.context.native_speech_followup import SPEECH_FILES,source_identity
        root=Path(__file__).resolve().parents[2]
        def git(cmd,**kwargs):return 'a'*40+'\n' if cmd[-2:]==['rev-parse','HEAD'] else ''
        with patch('portable.context.native_speech_followup.subprocess.check_output',side_effect=git):
            commit,files=source_identity(root)
        self.assertEqual(commit,'a'*40)
        self.assertEqual(files,{name:sha256(root/name) for name in SPEECH_FILES})
        with patch('portable.context.native_speech_followup.subprocess.check_output',side_effect=['a'*40,' M changed']):
            with self.assertRaises(RuntimeError):source_identity(root)


if __name__=='__main__':unittest.main()
