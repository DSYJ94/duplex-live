"""Prove the exact new proof-routing extension leaves old gradient code intact."""
import ast
import hashlib
from pathlib import Path
import subprocess

SCHEMA = 'dl-native-reasoning32-retained154-stream49-control40-proof-v1'


def loader_extension_only(old, current):
    expected = ast.parse("""if report['schema'] == 'dl-native-reasoning32-retained154-stream49-control40-proof-v1':
    from evals.native_reasoning_stage import verify_checkpoint as verify_reasoning32
    verify_reasoning32(proof)
    specific=True
""").body[0]
    counts = [0, 0]
    dump = lambda value: ast.dump(value, include_attributes=False)
    class RemoveExactRouting(ast.NodeTransformer):
        def visit_If(self, node):
            if dump(node.test) == dump(expected.test):
                if ([dump(n) for n in node.body] != [dump(n) for n in expected.body]
                        or len(node.orelse) != 1 or not isinstance(node.orelse[0], ast.If)):
                    raise ValueError('Changed reasoning proof-routing branch')
                counts[0] += 1
                return self.visit(node.orelse[0])
            return self.generic_visit(node)
        def visit_Tuple(self, node):
            strings = [n.value for n in node.elts if isinstance(n, ast.Constant)]
            if SCHEMA in strings:
                if 'dl-native-control40-retained154-stream49-proof-v1' not in strings:
                    raise ValueError('Unexpected native protocol schema extension')
                counts[1] += 1
                node.elts = [n for n in node.elts if not isinstance(n, ast.Constant) or n.value != SCHEMA]
            return self.generic_visit(node)
    normalized = RemoveExactRouting().visit(ast.parse(current))
    if counts != [1, 1] or dump(normalized) != dump(ast.parse(old)):
        raise ValueError('Old complete loader AST changed beyond exact proof-routing additions')
    return True


def verify(ready):
    from evals.native_control_gradient import implementations, reference_dependencies
    current = implementations()
    binding_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if ready['implementation_pins'] == current:
        return dict(mode='exact_all_implementation_hashes', current=current,
                    binding_implementation_sha256=binding_sha256)
    commit = ready['source_commit']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Exact original gradient source required')
    root = Path(__file__).resolve().parents[1]
    if set(ready['implementation_pins']) != set(current):
        raise ValueError('Changed gradient implementation dependency inventory')
    for name, digest in current.items():
        old = subprocess.check_output(['git', 'show', commit + ':evals/' + name], cwd=root)
        if hashlib.sha256(old).hexdigest() != ready['implementation_pins'][name]:
            raise ValueError('Original gradient source hash mismatch: ' + name)
        if digest != ready['implementation_pins'][name]:
            if name != 'context_adapter_load.py':
                raise ValueError('Changed actual gradient dependency: ' + name)
            loader_extension_only(old.decode(), (root / 'evals' / name).read_text())
    reference_dependencies(commit)
    return dict(mode='exact_original_hashes_plus_whole_loader_AST_routing_extension',
                source_commit=commit, original=ready['implementation_pins'], current=current,
                binding_implementation_sha256=binding_sha256,
                exact_old_loader_AST_after_removing_only_declared_two_additions=True,
                numerical_or_gradient_thresholds_changed=False)
