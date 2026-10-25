"""CPU collector fixtures only: no real model, GPU, backend, or behavior approval."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals.moshi_model_download import sha256
from evals.native_control40_fresh_session import cases
from portable.context.native_fresh_development import (ROLES,CASE_FILES,command,budget_guard,
    preflight_document,raw_case,compare_fresh,controller_identity,actor,complete_actor,speech_report)


class FreshCollectorFixtures(unittest.TestCase):
    roles=ROLES
    preflight_schema='dl-native-control40-fresh-paired40-preflight-v1'

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.out=self.root/'evaluation';self.out.mkdir()
        self.proof=self.root/'pilot/training';self.baseline=self.root/'context-training-proof-7510331'
        # Use authored placeholders, never the sealed real question plan/audio.
        fixture=self.root/'placeholder-plan.json'
        self.write(fixture,dict(do_not_train=True,training_approved=False,cases=[
            dict(id='fixture-%02d'%i,kind='ordinary' if i<4 else 'private',
                backend='observe-only' if i<4 else 'native-asr',minimum_actual_lookup_delay_s=1.)
            for i in range(20)]))
        for target,value in [('PLAN',fixture),('PLAN_SHA256',sha256(fixture))]:
            patched=patch('evals.native_control40_fresh_session.'+target,value)
            patched.start();self.addCleanup(patched.stop)
        self.review=self.root/'review.json';self.review.write_text('{}')
        self.quality=self.root/'quality.json';self.quality.write_text('{}')
        self.plan=dict(schema=self.preflight_schema,complete40=True,
            cases=cases(self.proof,self.baseline,self.roles[1]),candidate_proof=str(self.proof),baseline_proof=str(self.baseline),
            baseline_identity=dict(checkpoint_sha256='base'),candidate_identity=dict(checkpoint_sha256='candidate'),
            gpu_mode='serial-one-gpu',maximum_gpus=1,behavior_review_sha256=sha256(self.review),
            source_quality_sha256=sha256(self.quality),source_profile='owner-approved-turbo-spelling-v1',
            evaluation_only=True,training_approved=False,checkpoint_adopted=False)
        self.args=SimpleNamespace(source=self.root,source_commit='a'*40,out=self.out,proof=self.proof,
            review=self.review,quality=self.quality,data=self.root/'inputs',asr_python=Path('/isolated/asr/python'),
            deadline=4102444800.,role=self.roles[0])
        self.seal()

    @staticmethod
    def write(path,value):path.write_text(json.dumps(value))

    def seal(self):
        self.write(self.out/'preflight.json',self.plan)
        self.args.preflight_sha256=sha256(self.out/'preflight.json')

    def make_case(self,folder,row,identity):
        folder.mkdir()
        for name in ('model.wav','user_in.wav','packet-stream.wav'):
            (folder/name).write_bytes(name.encode()+row['scenario'].encode())
        (folder/'events.jsonl').write_text('{"text_token":12}\n')
        (folder/'worker-events.json').write_text('[]')
        loading=dict(adapter=dict(checkpoint_sha256=identity['checkpoint_sha256'],proof=row['proof'],
                zero=False,disabled=False,installed_before_native_graphs=True),
            native_asr_resource_placement=dict(profile='shared-gpu-evaluation',distinct_physical_gpus=False,
                separate_native_and_asr_processes=True),lookup_delivery_minimum_delay_s=row['minimum_lookup_delay_s'])
        self.write(folder/'loading.json',loading)
        raw=dict(scenario=row['scenario'],text='CPU fixture words',loading=loading,
            source_split='fresh-evaluation',fresh_holdout=True,evaluation_only=True,training_approved=False,
            quality_qualified=False,full20_task_qualification_passed=False,actual_lm_and_pcm_executed_together=True,
            student_history=None,scheduled_dispatch=False,word_or_acoustic_tokens_forced=False,response_start_forced=False,
            real_user_samples_processed=24000,real_user_samples_expected=24000,audio_sha256=sha256(folder/'model.wav'),
            asr_environment=dict(reference_transcript_supplied=False,assigned_visible_gpu='0'),
            distinct_lm_asr_gpu_processes=False,separate_native_and_asr_processes=True)
        self.write(folder/'report.json',raw)
        return raw

    def fake_probe(self,argv,session=None):
        self.assertEqual(session.scope,'fresh-full20')
        self.assertNotIn('--train-history',argv)
        fields=dict(zip(argv[::2],argv[1::2]))
        row=next(r for r in self.plan['cases'] if r['scenario']==fields['--scenario']
                 and r['proof']==fields['--adapter-proof'])
        identity=self.plan['baseline_identity' if row['role']==self.roles[0] else 'candidate_identity']
        self.make_case(Path(fields['--out']),row,identity)

    def run_actor(self,side_effect=None):
        with patch('portable.context.native_fresh_development.clean'), \
                patch('portable.context.native_fresh_development.controller_identity'), \
                patch('evals.context_spanning_fresh_probe.main',side_effect=side_effect or self.fake_probe) as probe:
            actor(self.args)
            return probe.call_count

    def test_command_whitelists_runtime_inputs_without_answers_or_training(self):
        row=dict(self.plan['cases'][20],expected='SECRET SCORING ANSWER',worker_only_world={'secret':'never supplied'})
        argv=command(self.args,row,self.out/'case')
        text=' '.join(argv)
        self.assertNotIn('SECRET',text);self.assertNotIn('never supplied',text)
        for flag in ('--train-history','--ret-logit-bias','--native-runtime-profile','--native-text-temperature'):
            self.assertNotIn(flag,argv)
        self.assertEqual(argv[argv.index('--adapter-proof')+1],str(self.proof))
        self.assertEqual(argv[argv.index('--native-fresh-candidate-proof')+1],str(self.proof))
        self.assertEqual(argv[argv.index('--native-asr-placement')+1],'shared-gpu-evaluation')

    def test_preflight_hash_review_quality_and_subset_cannot_change(self):
        preflight_document(self.args)
        self.quality.write_text('{"changed":true}')
        with self.assertRaises(ValueError):preflight_document(self.args)
        self.quality.write_text('{}')
        self.plan['cases'].pop();self.seal()
        with self.assertRaises(ValueError):preflight_document(self.args)

    def test_complete_actor_executes_all20_plus_cold_intervening_reused_parity(self):
        self.assertEqual(self.run_actor(),23)
        state=json.loads((self.out/self.roles[0]/'execution.json').read_text())
        self.assertEqual([c['question_index'] for c in state['cases']],list(range(20)))
        self.assertEqual([c['index'] for c in state['cases']],list(range(20)))
        self.assertTrue(state['complete20']);self.assertFalse(state['behavior_approved'])
        self.assertFalse(state['parity']['native_delayed_input_tensors_recorded_or_compared'])
        self.assertEqual(complete_actor(self.out,self.roles[0],self.plan,'a'*40,self.args.preflight_sha256),
                         sha256(self.out/self.roles[0]/'execution.json'))
        self.assertFalse(list(self.out.rglob('student-history.pt')))

    def test_candidate_actor_uses_all20_indices_and_exact_candidate_proof(self):
        self.args.role=self.roles[1];self.assertEqual(self.run_actor(),23)
        state=json.loads((self.out/self.roles[1]/'execution.json').read_text())
        self.assertEqual([c['index'] for c in state['cases']],list(range(20,40)))
        self.assertEqual(Path(state['proof']),self.proof)
        complete_actor(self.out,self.roles[1],self.plan,'a'*40,self.args.preflight_sha256)

    def test_final_case_failure_is_preserved_and_cannot_be_complete(self):
        def probe(argv,session):
            if argv[argv.index('--out')+1].endswith('case-19'):raise RuntimeError('fixture final failure')
            self.fake_probe(argv,session)
        with self.assertRaisesRegex(RuntimeError,'fixture final failure'):self.run_actor(probe)
        state=json.loads((self.out/self.roles[0]/'execution.json').read_text())
        self.assertEqual(state['phase'],'failed');self.assertEqual(len(state['cases']),19)
        self.assertFalse(state['complete20'])
        with self.assertRaises(ValueError):complete_actor(self.out,self.roles[0],self.plan,'a'*40,self.args.preflight_sha256)

    def test_changed_parity_or_final_case_artifact_cannot_hide_behind_saved_success(self):
        self.run_actor()
        path=self.out/self.roles[0]/'parity-reused/events.jsonl';original=path.read_text();path.write_text('{"text_token":13}\n')
        with self.assertRaises(ValueError):complete_actor(self.out,self.roles[0],self.plan,'a'*40,self.args.preflight_sha256)
        path.write_text(original)
        (self.out/self.roles[0]/'case-19/user_in.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):complete_actor(self.out,self.roles[0],self.plan,'a'*40,self.args.preflight_sha256)

    def test_cold_reused_output_parity_is_actual_PCM_tokens_and_user_audio(self):
        row=self.plan['cases'][0]
        for name in ('cold','reused'):self.make_case(self.out/name,row,self.plan['baseline_identity'])
        self.assertTrue(compare_fresh(self.out/'cold',self.out/'reused')['passed'])
        for name in ('user_in.wav','packet-stream.wav','events.jsonl'):
            path=self.out/'reused'/name;original=path.read_bytes()
            path.write_text('{"text_token":99}\n' if name=='events.jsonl' else 'changed PCM')
            with self.subTest(name=name),self.assertRaises(RuntimeError):compare_fresh(self.out/'cold',self.out/'reused')
            path.write_bytes(original)

    def test_train_recorder_forcing_diagnostic_or_wrong_delay_rejected(self):
        row=next(r for r in self.plan['cases'] if r['backend']=='native-asr')
        folder=self.out/'one';raw=self.make_case(folder,row,self.plan['baseline_identity'])
        changes=[lambda r:r.update(student_history={'teacher':'forbidden'}),
            lambda r:r.update(quality_qualified=True),lambda r:r.update(full20_task_qualification_passed=True),
            lambda r:r.update(response_start_forced=True),lambda r:r.update(scheduled_dispatch=True),
            lambda r:r['loading'].update(lookup_delivery_minimum_delay_s=row['minimum_lookup_delay_s']+1),
            lambda r:r['loading']['adapter'].update(checkpoint_sha256='wrong'),
            lambda r:r['loading']['adapter'].update(disabled=True),
            lambda r:r['loading'].update(native_sampling_diagnostic={'on':True}),
            lambda r:r['asr_environment'].update(reference_transcript_supplied=True)]
        for mutate in changes:
            changed=copy.deepcopy(raw);mutate(changed);self.write(folder/'report.json',changed)
            with self.assertRaises(ValueError):raw_case(folder,row,self.plan['baseline_identity'])
        self.write(folder/'report.json',raw);(folder/'student-history.pt').write_bytes(b'forbidden')
        with self.assertRaises(ValueError):raw_case(folder,row,self.plan['baseline_identity'])

    def test_speech_audit_keeps_all20_even_if_they_fail_word_or_silence_checks(self):
        folders=[];rows=[]
        for i,row in enumerate(self.plan['cases'][:20]):
            folder=self.out/('audio-%02d'%i);self.make_case(folder,row,self.plan['baseline_identity']);folders.append(folder)
            rows.append(dict(run=str(folder),source_kind='native',wav_sha256=sha256(folder/'model.wav'),
                model_text='CPU fixture words',wer_against_model_text=1.,audit_failure='fixture silent failure'))
        path=self.out/'speech.json';audit=dict(schema='dl-native-speech-audit-v1',device='cuda',results=rows)
        self.write(path,audit)
        self.assertEqual(speech_report(path,folders),sha256(path))
        audit['results'].pop();self.write(path,audit)
        with self.assertRaises(ValueError):speech_report(path,folders)

    def test_actor_rejects_changed_source_before_any_fresh_case(self):
        with patch('portable.context.native_fresh_development.clean',side_effect=ValueError('source changed')), \
                patch('evals.context_spanning_fresh_probe.main') as probe:
            with self.assertRaises(ValueError):actor(self.args)
            probe.assert_not_called()
        self.assertFalse((self.out/self.roles[0]).exists())

    def test_actor_requires_actual_parent_birth_active_role_and_preflight(self):
        self.args.controller_pid=41001;self.args.controller_birth=123;self.args.pod_id='fixture-pod'
        self.args.deadline_utc='2100-01-01T00:00:00Z'
        state=dict(schema='dl-native-portable-fresh40-controller-v1',pid=41001,
            source_commit=self.args.source_commit,pod_id='fixture-pod',preflight_sha256=self.args.preflight_sha256,
            phase='paired_fresh40',active_actor_role=self.roles[0],deadline_utc=self.args.deadline_utc,
            cost_guard_deadline_utc='2100-01-01T00:03:00Z')
        path=self.out/'execution.json';self.write(path,state)
        with patch('portable.context.native_fresh_development.os.getppid',return_value=41001), \
                patch('portable.context.native_fresh_development.process_birth',return_value=123):
            controller_identity(self.args)
            for key,value in [('active_actor_role',self.roles[1]),('phase','failed'),('preflight_sha256','changed'),
                    ('pod_id','other'),('cost_guard_deadline_utc','2100-01-01T00:01:00Z')]:
                self.write(path,dict(state,**{key:value}))
                with self.subTest(key=key),self.assertRaises(ValueError):controller_identity(self.args)
            self.write(path,state)
        with patch('portable.context.native_fresh_development.os.getppid',return_value=999):
            with self.assertRaises(ValueError):controller_identity(self.args)
        with patch('portable.context.native_fresh_development.os.getppid',return_value=41001), \
                patch('portable.context.native_fresh_development.process_birth',return_value=999):
            with self.assertRaises(ValueError):controller_identity(self.args)


class ReasoningFreshCollectorFixtures(FreshCollectorFixtures):
    roles=('baseline520','candidate-reasoning32')
    preflight_schema='dl-native-reasoning32-fresh-paired40-preflight-v1'


class FreshBudgetGuardTests(unittest.TestCase):
    def test_same_actual_pod_fresh_live_guard_and_bounded_deadline_required(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'guard.json'
            now=1_800_000_000.
            def stamp(seconds):return datetime.fromtimestamp(seconds,tz=timezone.utc).isoformat()
            good=dict(phase='armed',pod_id='actual-pod',at_utc=stamp(now-10),deadline_utc=stamp(now+400))
            path.write_text(json.dumps(good));budget_guard(path,'actual-pod',now+200,now=now)
            for key,value in [('phase','failed'),('pod_id','other'),('at_utc',stamp(now-91)),
                    ('at_utc',stamp(now+1)),('deadline_utc',stamp(now+319))]:
                path.write_text(json.dumps(dict(good,**{key:value})))
                with self.subTest(key=key),self.assertRaises(ValueError):budget_guard(path,'actual-pod',now+200,now=now)


class GcpFreshBudgetGuardTests(unittest.TestCase):
    def setUp(self):
        from portable.context.gcp_budget_watchdog import AUTH_SHA
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'actual-gcp-lease.json'
        self.now=1_800_000_000.;self.deadline=self.now+3600
        self.resource='951004838667287080'
        self.good=dict(provider='gcp',resource_id=self.resource,authorization_sha256=AUTH_SHA,
            execution_authorized=True,live_provider_deadline_verified=True,
            live_external_cost_guard_verified=True,single_gpu_across_providers_verified=True,
            checked_unix=self.now-1,deadline_unix=self.now+7200,
            remaining_budget_usd=50.,planned_maximum_cost_usd=20.)

    def check(self,receipt=None,resource=None):
        self.path.write_text(json.dumps(self.good if receipt is None else receipt))
        with patch('time.time',return_value=self.now):
            return budget_guard(self.path,self.resource if resource is None else resource,self.deadline)

    def test_actual_gcp_lease_is_validated_without_rewriting_it_as_a_runpod_receipt(self):
        value=self.check();original=self.path.read_bytes()
        self.assertEqual(value['provider'],'gcp')
        self.assertEqual(value['resource_id'],self.resource)
        self.assertEqual(value['pod_id'],self.resource)
        self.assertEqual(value['authorization_sha256'],self.good['authorization_sha256'])
        self.assertEqual(datetime.fromisoformat(value['at_utc']).timestamp(),self.good['checked_unix'])
        self.assertEqual(datetime.fromisoformat(value['deadline_utc']).timestamp(),self.good['deadline_unix'])
        self.assertEqual(self.path.read_bytes(),original)

    def test_legacy_looking_fields_cannot_bypass_gcp_authorization(self):
        value=dict(self.good,execution_authorized=False,phase='armed',pod_id=self.resource,
            at_utc=datetime.fromtimestamp(self.now,tz=timezone.utc).isoformat(),
            deadline_utc=datetime.fromtimestamp(self.now+7200,tz=timezone.utc).isoformat())
        with self.assertRaises(ValueError):self.check(value)

    def test_same_gcp_authority_resource_freshness_budget_and_deadline_remain_required(self):
        for key,value in [('authorization_sha256','0'*64),('checked_unix',self.now-31),
                ('checked_unix',self.now+1),('deadline_unix',self.deadline+119),
                ('deadline_unix',self.now+12*3600+1),('remaining_budget_usd',101.),
                ('planned_maximum_cost_usd',51.),('planned_maximum_cost_usd',True),
                ('execution_authorized',False),('live_provider_deadline_verified',False),
                ('live_external_cost_guard_verified',False),('single_gpu_across_providers_verified',False)]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                self.check(dict(self.good,**{key:value}))
        with self.assertRaises(ValueError):self.check(resource='different-resource')

    def test_only_guard_file_is_read_and_no_fresh_plan_or_model_is_opened(self):
        original_read=Path.read_text
        def read(path,*args,**kwargs):
            if path!=self.path:raise AssertionError('Only the actual guard receipt may be read')
            return original_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',read):self.check()


if __name__=='__main__':unittest.main()
