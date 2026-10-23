import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest

from evals.native_asr_placement import assign,report
from portable.context.native_development import command


class NativeASRPlacement(unittest.TestCase):
    def test_legacy_two_gpu_and_explicit_shared_gpu_are_distinct(self):
        self.assertEqual(assign('0,1','separate-gpu','/asr'),('0','1'))
        self.assertEqual(assign('0','shared-gpu-evaluation','/asr'),('0','0'))
        for visible,profile,python in [('0','separate-gpu','/asr'),('0,0','separate-gpu','/asr'),
                ('0,1','shared-gpu-evaluation','/asr'),('','shared-gpu-evaluation','/asr'),
                ('0','shared-gpu-evaluation',None),('0','unknown','/asr')]:
            with self.subTest(visible=visible,profile=profile),self.assertRaises(ValueError):assign(visible,profile,python)
        shared=report(SimpleNamespace(native_asr_placement='shared-gpu-evaluation'))
        self.assertFalse(shared['distinct_physical_gpus']);self.assertTrue(shared['separate_native_and_asr_processes'])
        self.assertTrue(shared['resource_timing_cohort_differs_from_two_gpu'])
        self.assertTrue(report(SimpleNamespace())['distinct_physical_gpus'])

    def test_single_gpu_command_keeps_all_native_options_except_declared_placement(self):
        case=dict(source='old',scenario='recovery-validation-000',kind='private',backend='native-asr')
        old=command(case,Path('/proof'),Path('/out'),Path('/asr'))
        shared=command(case,Path('/proof'),Path('/out'),Path('/asr'),gpu_mode='serial-one-gpu')
        i=shared.index('--native-asr-placement');self.assertEqual(shared[i+1],'shared-gpu-evaluation')
        self.assertEqual(shared[:i]+shared[i+2:],old)
        self.assertNotIn('--ret-logit-bias',shared);self.assertNotIn('--native-runtime-profile',shared)

    def test_native_requests_audio_words_receipts_and_clocks_are_unchanged(self):
        old=subprocess.check_output(['git','show','20a35783f0e38a5156a12bc503d4803ad810cd3d:evals/context_spanning_native.py'],text=True)
        current=Path('evals/context_spanning_native.py').read_text()
        class ResourceMetadata(ast.NodeTransformer):
            def visit_Dict(self,node):
                node=self.generic_visit(node)
                if any(isinstance(k,ast.Constant) and k.value=='distinct_lm_asr_gpu_processes' for k in node.keys):
                    kept=[]
                    for key,value in zip(node.keys,node.values):
                        if isinstance(key,ast.Constant) and key.value in ('separate_native_and_asr_processes','native_asr_resource_placement'):continue
                        if isinstance(key,ast.Constant) and key.value=='distinct_lm_asr_gpu_processes':value=ast.Constant(True)
                        kept.append((key,value))
                    node.keys=[x[0] for x in kept];node.values=[x[1] for x in kept]
                return node
        def normalized(source):
            tree=ast.parse(source);run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run')
            return ast.dump(ResourceMetadata().visit(run),include_attributes=False)
        self.assertEqual(normalized(old),normalized(current))


if __name__=='__main__':unittest.main()
