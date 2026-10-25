"""Fail-closed construction and complete-set recovery contracts; no Torch."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals.native_saved_stage_context import bind,construction_tree
from evals.native_saved_stage_recovery import partial_fingerprints,validate_complete_fingerprints
from portable.context.native_saved_stage_runtime import arguments,canonical_command,guard_receipt,isolate,restore_recorded_precision


class ConstructionTests(unittest.TestCase):
    def source(self, before='', boundary='if args.reload:', second=''):
        return ("def run(args):\n    prepared = 'complete'\n    with scope():\n"
            +before+"        "+boundary+"\n            raise RuntimeError('reload ran')\n"
            +second+"        optimizer = forbidden_optimizer()\n        optimizer.step()\n"
            +"    raise RuntimeError('training tail ran')\n").encode()

    def compile_fixture(self, source, capture):
        from contextlib import nullcontext
        with patch('evals.native_saved_stage_context.ORIGINAL_SHA256',hashlib.sha256(source).hexdigest()):
            return bind(source,dict(scope=nullcontext),capture)

    def test_capture_runs_inside_scope_without_any_optimizer_or_reload(self):
        captured=[]
        callback=lambda context:captured.append(context['prepared']) or 'captured'
        function,receipt=self.compile_fixture(self.source(),callback)
        for reload in (False,True):
            self.assertEqual(function(SimpleNamespace(reload=reload)),'captured')
        self.assertEqual(captured,['complete','complete'])
        self.assertFalse(receipt['optimizer_branch_included'])
        self.assertFalse(receipt['gpu_execution_authorized'])

    def test_optimizer_before_boundary_is_rejected(self):
        for before in ("        optimizer = forbidden_optimizer()\n", "        p.backward()\n",
                "        mutate.step()\n", "        restore_state()\n"):
            with self.subTest(before=before),self.assertRaises(ValueError):
                self.compile_fixture(self.source(before=before),lambda context:None)

    def test_missing_or_ambiguous_reload_boundary_is_rejected(self):
        for source in (self.source(boundary='if other:'),
                self.source(second='        if args.reload:\n            pass\n')):
            with self.assertRaises(ValueError):self.compile_fixture(source,lambda context:None)

    def test_changed_original_source_is_rejected(self):
        original=(Path(__file__).resolve().parents[1]/'context_actual_train.py').read_bytes()
        tree,receipt=construction_tree(original)
        self.assertEqual(receipt['derived_ast_sha256'],
            '7907f6de7ddbdaa1a22111829a67cfb1a4a2e5df629482f1693c4676549d15f0')
        self.assertFalse(any(isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
            and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range'
            and any(isinstance(a,ast.Attribute) and a.attr=='actual_fit_updates' for a in n.iter.args)
            for n in ast.walk(tree)))
        with self.assertRaises(ValueError):construction_tree(original+b'\n# changed')

    def test_recovery_callback_has_no_optimizer_step_backward_or_state_save(self):
        source=(Path(__file__).resolve().parents[1]/'native_saved_stage_recovery.py').read_bytes()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Call):
                function=node.func
                name=function.id if isinstance(function,ast.Name) else getattr(function,'attr',None)
                self.assertNotIn(name,('step','backward','zero_grad','save_state'))


class FingerprintTests(unittest.TestCase):
    def setUp(self):
        self.ready=dict(native49_policy=dict(teachers=[dict(id='n'+str(i)) for i in range(49)]),
            control40_policy=dict(teachers=[dict(id='c'+str(i)) for i in range(40)]))
        self.rows=[dict(native_stream_restore_fingerprint='n'+str(i),sha256='a'*64) for i in range(49)]
        self.rows += [dict(control40_restore_fingerprint='c'+str(i),**self.native()) for i in range(31)]
        audio=dict(teacher_ids=['a'+str(i) for i in range(49)],
            native_stream_auxiliary=self.ready['native49_policy'])
        control=dict(self.ready['control40_policy'],target_counts={
            'c'+str(i):dict(text=3,audio=[2]*8) for i in range(40)})
        reasoning=dict(teachers=[dict(id='r'+str(i)) for i in range(32)],target_counts={
            'r'+str(i):dict(text=3,audio=[2]*8,audio_mask_frames=2) for i in range(32)})
        self.metadata=dict(recipe=dict(assistant_audio_auxiliary=audio,control40=control,reasoning32=reasoning))
        self.actual=dict(original28={'o'+str(i):'a'*64 for i in range(28)},
            audio49={i:'a'*64 for i in audio['teacher_ids']},
            native49={r['id']:'a'*64 for r in audio['native_stream_auxiliary']['teachers']},
            control40={r['id']:self.native() for r in control['teachers']},
            reasoning32={r['id']:dict(self.native(),audio_mask_frames=2) for r in reasoning['teachers']})

    def native(self):
        return dict(native_text_calls_sha256='b'*64,assistant_audio_logits_sha256='c'*64,
            assistant_audio_targets=[2]*8,text_targets=3)

    def text(self,rows=None):return '\n'.join(json.dumps(row) for row in (self.rows if rows is None else rows))

    def partials(self):return partial_fingerprints(self.text(),self.ready)

    def test_complete_recomputed_sets_match_preserved_prefix(self):
        partials=self.partials()
        self.assertEqual({k:len(v) for k,v in partials.items()},dict(native49=49,control40=31,reasoning32=0))
        self.assertTrue(validate_complete_fingerprints(self.actual,self.metadata,partials))

    def test_truncated_duplicate_reordered_or_wrong_prefix_rejected(self):
        for rows in (self.rows[:-1],self.rows+[self.rows[-1]],list(reversed(self.rows))):
            with self.assertRaises(ValueError):partial_fingerprints(self.text(rows),self.ready)
        rows=copy.deepcopy(self.rows);rows[-1]['control40_restore_fingerprint']='c39'
        with self.assertRaises(ValueError):partial_fingerprints(self.text(rows),self.ready)

    def test_missing_even_unpreserved_final_case_is_rejected(self):
        for name in self.actual:
            changed=copy.deepcopy(self.actual);changed[name].pop(next(reversed(changed[name])))
            with self.subTest(name=name),self.assertRaises(ValueError):
                validate_complete_fingerprints(changed,self.metadata,self.partials())

    def test_same_count_substitution_and_extra_group_are_rejected(self):
        for name in ('audio49','native49','control40','reasoning32'):
            changed=copy.deepcopy(self.actual);changed[name]['other']=changed[name].pop(next(iter(changed[name])))
            with self.subTest(name=name),self.assertRaises(ValueError):
                validate_complete_fingerprints(changed,self.metadata,self.partials())
        changed=copy.deepcopy(self.actual);changed['extra']={}
        with self.assertRaises(ValueError):validate_complete_fingerprints(changed,self.metadata,self.partials())

    def test_original_partial_value_must_match(self):
        for name in ('native49','control40'):
            changed=copy.deepcopy(self.actual);key=next(iter(changed[name]))
            if name=='native49':changed[name][key]='d'*64
            else:changed[name][key]['assistant_audio_logits_sha256']='d'*64
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'Original preserved'):
                validate_complete_fingerprints(changed,self.metadata,self.partials())

    def test_changed_unpreserved_target_counts_and_invalid_hashes_rejected(self):
        for name,key,field,value in [('reasoning32','r31','text_targets',4),
                ('control40','c39','assistant_audio_targets',[1]*8),
                ('reasoning32','r31','text_targets',True),
                ('reasoning32','r31','assistant_audio_logits_sha256','bad')]:
            changed=copy.deepcopy(self.actual);changed[name][key][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_complete_fingerprints(changed,self.metadata,self.partials())

    def test_actual_reasoning_producer_shape_and_pinned_audio_mask(self):
        # Shape and counts from the immutable original producer's real V6 log.
        row=dict(native_text_calls_sha256='2a31827df5fd550ffd9a243018d15e61301dc231b30917b65d9279a3fb827599',
            assistant_audio_logits_sha256='623ae2544e9666ec0638cf7395a9a042b72b265cb997f24f401baf1503476c12',
            assistant_audio_targets=[706]+[705]*7,text_targets=707,audio_mask_frames=706)
        self.actual['reasoning32']['r0']=row
        self.metadata['recipe']['reasoning32']['target_counts']['r0']=dict(
            text=707,audio=[706]+[705]*7,audio_mask_frames=706)
        self.assertTrue(validate_complete_fingerprints(self.actual,self.metadata,self.partials()))
        for value in (None,0,True,705,707):
            changed=copy.deepcopy(self.actual)
            if value is None:changed['reasoning32']['r0'].pop('audio_mask_frames')
            else:changed['reasoning32']['r0']['audio_mask_frames']=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_complete_fingerprints(changed,self.metadata,self.partials())
        for name in ('reasoning32','control40'):
            changed=copy.deepcopy(self.actual);changed[name][next(iter(changed[name]))]['unknown']=1
            with self.assertRaises(ValueError):validate_complete_fingerprints(changed,self.metadata,self.partials())


class RuntimeContractTests(unittest.TestCase):
    def test_ngc_default_precision_is_restored_to_original_policy(self):
        matmul=SimpleNamespace(allow_tf32=True)
        state=dict(precision='high')
        def set_precision(value):
            state['precision']=value;matmul.allow_tf32=False
        torch=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
            get_float32_matmul_precision=lambda:state['precision'],set_float32_matmul_precision=set_precision)
        result=restore_recorded_precision(torch)
        self.assertEqual(result['before'],dict(float32_precision='high',matmul_tf32=True))
        self.assertEqual(result['after'],dict(float32_precision='highest',matmul_tf32=False))
        self.assertEqual(result['optimizer_updates'],0)
        matmul.allow_tf32=True;torch.set_float32_matmul_precision=lambda value:None
        with self.assertRaises(ValueError):restore_recorded_precision(torch)

    def test_exact_command_only_changes_output_and_never_requests_reload(self):
        command=['fixture-python','-u','-m','portable.context.native_training_cached',
            '--actual-temporal-fit','--actual-fit-updates','8','--out','old']
        state=dict(planned_commands=dict(bounded8_training=command))
        with patch('portable.context.native_saved_stage_runtime.FIT_COMMAND_SHA256',canonical_command(command)):
            self.assertEqual(arguments(state,'new'),command[4:-1]+['new'])
            self.assertEqual(command[-1],'old')
            changed=copy.deepcopy(state);changed['planned_commands']['bounded8_training'].append('--reload')
            with self.assertRaises(ValueError):arguments(changed,'new')

    def guard(self):
        return dict(execution_authorized=True,live_provider_deadline_verified=True,
            live_external_cost_guard_verified=True,single_gpu_across_providers_verified=True,
            checked_unix=1000,deadline_unix=5000,provider='gcp',remaining_budget_usd=100.,
            planned_maximum_cost_usd=90.,resource_id='fixture-vm',authorization_sha256='a'*64)

    def test_fresh_bounded_guard_is_required(self):
        with patch('portable.context.native_saved_stage_runtime.time.time',return_value=1000):
            self.assertEqual(guard_receipt(self.guard)['provider'],'gcp')
            for key,value in [('execution_authorized',False),('live_provider_deadline_verified',False),
                    ('live_external_cost_guard_verified',False),('single_gpu_across_providers_verified',False),
                    ('checked_unix',900),('checked_unix',1001),('deadline_unix',999),
                    ('deadline_unix',50000),('remaining_budget_usd',101),
                    ('planned_maximum_cost_usd',101),('authorization_sha256',''),('resource_id','')]:
                changed=self.guard();changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):guard_receipt(lambda:changed)

    def test_other_provider_or_nan_budget_fails_closed(self):
        with patch('portable.context.native_saved_stage_runtime.time.time',return_value=1000):
            for key,value in [('provider','unapproved-provider'),('remaining_budget_usd',float('nan')),
                    ('planned_maximum_cost_usd',float('inf')),('planned_maximum_cost_usd',True)]:
                changed=self.guard();changed[key]=value
                with self.assertRaises(ValueError):guard_receipt(lambda:changed)

    def test_wrong_or_dirty_original_checkout_is_rejected_before_bootstrap(self):
        root=Path(__file__).resolve().parents[2]
        from evals.native_reasoning_saved_stage_audit import SOURCE
        for values in (['b'*40],[SOURCE,' M changed.py']):
            with patch('portable.context.native_saved_stage_runtime.git',side_effect=values),self.assertRaises(ValueError):
                isolate(root)


if __name__=='__main__':unittest.main()
