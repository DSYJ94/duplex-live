"""A stronger reference must preserve the entire old numerical calculation."""
import ast
import copy
from pathlib import Path
import unittest


class NormalizeReferenceMetadata(ast.NodeTransformer):
    def visit_ImportFrom(self,node):return None
    def visit_With(self,node):
        node=self.generic_visit(node)
        node.items=[i for i in node.items if not (isinstance(i.context_expr,ast.Call)
                    and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='attention_reference')]
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id=='dict' and any(k.arg=='engine_report_sha256' for k in node.keywords):
            extra={'attention_reference_code_sha256','original_numerical_code_sha256',
                   'original_failure_sha256','independent_reference_profile'}
            node.keywords=[k for k in node.keywords if k.arg not in extra]
            for k in node.keywords:
                if k.arg in ('schema','reference_code_sha256'):k.value=ast.Constant('explicit-reference-metadata')
        return node
    def visit_Assign(self,node):
        for t in node.targets:
            if (isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='result'
                    and isinstance(t.slice,ast.Constant) and t.slice.value=='attention_reference'):return None
        return self.generic_visit(node)


class PreciseWhole40(unittest.TestCase):
    def test_all_inputs_calls_targets_masks_losses_causality_depth_and_thresholds_are_unchanged(self):
        def run(name):
            tree=ast.parse((Path('evals')/name).read_text())
            f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            return ast.dump(NormalizeReferenceMetadata().visit(f),include_attributes=False)
        self.assertEqual(run('native_control_numerical.py'),run('native_control_numerical_precise.py'))

    def test_regression_check_detects_subset_mask_or_tolerance_weakening(self):
        source=Path('evals/native_control_numerical_precise.py').read_text()
        tree=ast.parse(source);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        expected=ast.dump(NormalizeReferenceMetadata().visit(copy.deepcopy(f)),include_attributes=False)
        for before,after in (("for record in manifest['cases']:","for record in manifest['cases'][:39]:"),
                             ('threshold=.001','threshold=.01'),
                             ("mask=saved['mask'].cuda();targets=", "mask=saved['mask'].cuda();mask[-1]=False;targets=")):
            self.assertIn(before,source)
            changed=ast.parse(source.replace(before,after))
            g=next(n for n in changed.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            self.assertNotEqual(expected,ast.dump(NormalizeReferenceMetadata().visit(g),include_attributes=False))


if __name__=='__main__':unittest.main()
