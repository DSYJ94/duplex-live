import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import unittest
import tempfile
import torch

from evals.native_eval_session import NativeEvalSession
from portable.context.native_development import command, compare_native


def options(tmp_path):
    proof=tmp_path/'proof';proof.mkdir();(proof/'report.json').write_text('{}')
    return argparse.Namespace(adapter_proof=proof,assets=tmp_path/'assets',base=tmp_path/'base',
        voice_reference=tmp_path/'voice.wav',voice_provenance=tmp_path/'voice.json',
        layout_audit=None,ret_logit_bias=0,lookup_delivery_delay_s=0,native_text_temperature=.7,
        persona_profile='recorded-records',backend_reference_profile='answer-only',
        event_log_mode='immediate',voice='accepted-synthetic')


class NativeEvaluationSession(unittest.TestCase):
    def setUp(self):
        folder=tempfile.TemporaryDirectory();self.addCleanup(folder.cleanup)
        self.tmp_path=Path(folder.name)

    def test_hash_cache_rejects_replaced_verified_asset(self):
        tmp_path=self.tmp_path
        path=tmp_path/'model';path.write_bytes(b'original')
        session=NativeEvalSession();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        session.verify_asset(path,digest);session.verify_asset(path,digest)
        path.write_bytes(b'changed!')
        with self.assertRaisesRegex(RuntimeError,'identity changed'):session.verify_asset(path,digest)


    def test_reuse_preserves_one_factory_and_rejects_weight_mutation(self):
        tmp_path=self.tmp_path
        args=options(tmp_path);session=NativeEvalSession();calls=[]
        engine=SimpleNamespace(lm=torch.nn.Linear(2,2).requires_grad_(False),inject_context_span=lambda _:0)
        def factory():calls.append(1);return engine,[],{}
        with patch('evals.context_execution.snapshot',return_value={'math':True}):
            session.load(args,factory);session.finish_case(True)
            session.load(args,factory);session.finish_case(True)
            assert len(calls)==1 and session.receipt()['strictly_loaded_model_reused']
            with torch.no_grad():engine.lm.weight.add_(1)
            with self.assertRaisesRegex(RuntimeError,'changed between'):session.load(args,factory)


    def test_error_unwinds_recorder_and_forbids_reuse(self):
        tmp_path=self.tmp_path
        args=options(tmp_path);session=NativeEvalSession();calls=[]
        engine=SimpleNamespace(lm=torch.nn.Linear(2,2).requires_grad_(False),inject_context_span=lambda _:0)
        with patch('evals.context_execution.snapshot',return_value={}):
            session.load(args,lambda:(engine,[],{}))
            state=SimpleNamespace(graphed_main='wrapped')
            engine._dl_student_history=SimpleNamespace(forward='original',state=state,graph='original-graph',
                prefix_trace=SimpleNamespace(close=lambda:calls.append('closed')))
            session.finish_case(False)
            assert engine.lm.forward_codes=='original' and state.graphed_main=='original-graph'
            assert not hasattr(engine,'_dl_student_history') and calls==['closed']
            with self.assertRaisesRegex(RuntimeError,'Failed native session'):session.load(args,lambda:None)


    def test_model_command_excludes_scoring_answer(self):
        tmp_path=self.tmp_path
        row={'source':'old','scenario':'recovery-validation-000','kind':'private',
             'backend':'native-asr','value':'yellow circle','expected':'SECRET SCORING ANSWER'}
        argv=command(row,'7590052',tmp_path/'out',Path('/opt/asr/bin/python'))
        assert 'SECRET SCORING ANSWER' not in ' '.join(argv)
        assert argv[argv.index('--backend-mode')+1]=='native-asr'
        assert '--train-history' not in argv


    def test_parity_rejects_different_actual_native_input(self):
        tmp_path=self.tmp_path
        import json
        for name in ('cold','reused'):
            folder=tmp_path/name;folder.mkdir()
            (folder/'packet-stream.wav').write_bytes(b'identical PCM')
            (folder/'events.jsonl').write_text(json.dumps({'text_token':4})+'\n')
            (folder/'report.json').write_text(json.dumps({'text':'same'}))
            torch.save({'delayed_inputs':torch.tensor([1 if name=='cold' else 2])},folder/'student-history.pt')
        with self.assertRaisesRegex(RuntimeError,'parity failed'):compare_native(tmp_path/'cold',tmp_path/'reused')

if __name__=='__main__':unittest.main()
