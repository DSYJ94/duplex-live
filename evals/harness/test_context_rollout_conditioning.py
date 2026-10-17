import json
from pathlib import Path
import tempfile
import unittest
from evals.context_rollout_conditioning import verify
from evals.moshi_model_download import sha256
from evals.context_spanning_download import SOURCE_COMMIT


class RolloutConditioningTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        for f in ('voice.pt','adapters.safetensors','training-state.pt'):(self.p/f).write_bytes(f.encode())
        self.put('data.json',{'all_component_audits_passed':True,'past_loss_targets':0,'audio_loss_targets':0})
        self.put('approved-temporal-audit.json',{'schema':'dl-context-actual-temporal-audit-v1','passed':True,
            'optimizer_steps':0,'data_sha256':sha256(self.p/'data.json'),
            'components':[{'passed':True,'causal_prefix_maximum_error':0.,
                'comparisons':{'text_logits':{'maximum_absolute_error':.0001}}}]})
        protocol={'name':'bootstrap-preserve-provided-v1','attention_backend':'math',
            'code_sha256':sha256(Path(__file__).parents[1]/'context_native_layout.py')}
        self.report={'schema':'dl-context-actual-adapter-proof-v1','passed':True,
            'frozen_weights_unchanged':True,'base_audio_parameters_frozen':True,
            'all_actual_temporal_audits_passed':True,'all_component_audits_passed':True,
            'zero_adapter_native_batch_logits_exact':True,'source_commit':SOURCE_COMMIT,
            'last_layers':4,'rank':8,'scaling':2.,'past_loss_targets':0,'audio_loss_targets':0,
            'optimizer_steps':136,'expected_logits':{'a':'digest'},'native_protocol':protocol}
        for f,k in (('voice.pt','voice_sha256'),('adapters.safetensors','checkpoint_sha256'),
                ('training-state.pt','training_state_sha256'),('data.json','data_sha256'),
                ('approved-temporal-audit.json','actual_temporal_audit_sha256')):self.report[k]=sha256(self.p/f)
        self.put('report.json',self.report)
        self.reload={'schema':'dl-context-actual-adapter-reload-v1','passed':True,'fresh_process':True,'all_valid_logits_exact':True,
            'optimizer_rng_state_restored':True,'actual':self.report['expected_logits'],
            **{k:self.report[k] for k in ('checkpoint_sha256','training_state_sha256','data_sha256')}}
        self.put('reload-report.json',self.reload)
        self.loading={'adapter':{'proof':str(self.p),'zero':False,'disabled':False,
            'installed_before_native_graphs':True,'native_protocol':protocol,
            'checkpoint_sha256':self.report['checkpoint_sha256'],'voice_codes_sha256':self.report['voice_sha256']},
            'layout':{'native_protocol':protocol},'input_warmup':None}

    def put(self,f,value):(self.p/f).write_text(json.dumps(value))

    def test_exact_trained_collector_is_bound(self):
        self.assertEqual(verify(self.loading)['optimizer_steps'],136)

    def test_weight_or_voice_changes_fail_closed(self):
        for name in ('adapters.safetensors','voice.pt'):
            with self.subTest(name=name):
                original=(self.p/name).read_bytes();(self.p/name).write_bytes(b'changed')
                with self.assertRaises(RuntimeError):verify(self.loading)
                (self.p/name).write_bytes(original)

    def test_zero_disabled_and_unverified_reload_are_rejected(self):
        for flag in ('zero','disabled'):
            self.loading['adapter'][flag]=True
            with self.assertRaises(RuntimeError):verify(self.loading)
            self.loading['adapter'][flag]=False
        self.reload['optimizer_rng_state_restored']=False;self.put('reload-report.json',self.reload)
        with self.assertRaises(RuntimeError):verify(self.loading)


if __name__=='__main__':unittest.main()
