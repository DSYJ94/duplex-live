import unittest
from pathlib import Path
from types import SimpleNamespace
from evals.lychee_context import BASE_INSTRUCTIONS
from evals.native_persona_diagnostic import OLD_SUFFIX,NEW_SUFFIX,PROFILE,case,scoped_persona
from evals.native_text_sampling_diagnostic import policy


class PersonaDiagnostic(unittest.TestCase):
    def args(self,index=0):
        c=case(index)
        return SimpleNamespace(native_persona_diagnostic=True,native_persona_diagnostic_case=index,
            native_sampling_diagnostic=False,native_seen_retention_diagnostic=False,native_text_temperature=.7,
            backend_mode=c['backend'],expected_kind=c['kind'],train_history=index<6,scenario=c['scenario'],
            data=Path('/projects/u6si/duplex-live/runs/'+('native-curriculum-v2-data-7541875' if c['source']=='native-v2' else 'recovery-data-7407554')),
            adapter_proof=Path('/projects/u6si/duplex-live/runs/context-training-proof-7558824'),
            private_value=c.get('value','red circle'),voice='accepted-synthetic',persona_profile=PROFILE,
            adapter_disabled=False,adapter_zero=False,warm_audio=False,ret_logit_bias=0.,pending_status=False,lookup_delivery_delay_s=0.)

    def test_only_the_unqualified_final_instruction_changes(self):
        original=BASE_INSTRUCTIONS
        self.assertEqual(scoped_persona(),original[:-len(OLD_SUFFIX)]+NEW_SUFFIX)
        self.assertTrue(BASE_INSTRUCTIONS.endswith(OLD_SUFFIX))
        self.assertIn('only when the user requests a project record lookup',scoped_persona())

    def test_all10_original_tasks_are_retained_with_no_model_or_output_forcing(self):
        for i in range(10):
            r=policy(self.args(i));self.assertFalse(r['forced_reference_words'])
            self.assertTrue(r['diagnostic_outputs_must_not_be_trained_on'])
            self.assertFalse(r['native_quality_or_extension_approved'])

    def test_wrong_case_model_source_value_or_collector_scope_rejected(self):
        for field,value in [('scenario','recovery-train-008'),('data',Path('/other/data')),
                ('adapter_proof',Path('/projects/u6si/duplex-live/runs/context-training-proof-7510331')),
                ('private_value','blue square'),('train_history',False),('persona_profile','released-default')]:
            a=self.args();setattr(a,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):policy(a)

    def test_no_sampling_or_attention_warmup_substitution(self):
        for field,value in [('native_sampling_diagnostic',True),('native_seen_retention_diagnostic',True),
                ('native_text_temperature',0.),('adapter_zero',True),('warm_audio',True),('voice','released')]:
            a=self.args();setattr(a,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):policy(a)

    def test_launch_receipt_preserves_native_v2_dataset_selector(self):
        script=Path(__file__).parents[1]/'native-persona-diagnostic-step.sh'
        text=script.read_text()
        # The receipt previously overwrote case.source with a Git SHA, selecting
        # the old dataset for both quiet/pause cases before any model ran.
        self.assertIn("payload_source=os.environ['DL_TASK_COMMIT']",text)
        self.assertNotIn(",source=os.environ['DL_TASK_COMMIT']",text)
        for i in (4,5):self.assertEqual(case(i)['source'],'native-v2')

    def test_no_implicit_persona_index_or_invalid_case(self):
        a=self.args();a.native_persona_diagnostic=False
        with self.assertRaises(ValueError):policy(a)
        for i in [-1,10,True,None]:
            with self.assertRaises(ValueError):case(i)


if __name__=='__main__':unittest.main()
