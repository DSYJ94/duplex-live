"""Construct the exact original context, returning before its optimizer branch.

The original source is immutable and SHA-pinned. Its construction prefix is
compiled separately with one explicit capture return at the reload boundary.
The derived syntax-tree hash must be recorded by a future recovery producer.
This module neither loads a model nor authorizes execution or training.
"""
import ast
import copy
import hashlib
import inspect

ORIGINAL_SHA256='2e504fac5d3171680e3277d1e436f7db51d9403b45a1e77f09d4c2c7ccc893c5'


def reload_boundary(node):
    return (isinstance(node,ast.If) and isinstance(node.test,ast.Attribute)
        and isinstance(node.test.value,ast.Name) and node.test.value.id=='args'
        and node.test.attr=='reload')


def construction_tree(source):
    if hashlib.sha256(source).hexdigest()!=ORIGINAL_SHA256:
        raise ValueError('Exact original actual-history source required')
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run']
    if len(functions)!=1:raise ValueError('One original construction function required')
    function=copy.deepcopy(functions[0])
    scopes=[(i,n) for i,n in enumerate(function.body) if isinstance(n,ast.With)
        and any(reload_boundary(child) for child in n.body)]
    if len(scopes)!=1:raise ValueError('One exact native construction scope required')
    index,scope=scopes[0]
    boundaries=[i for i,n in enumerate(scope.body) if reload_boundary(n)]
    if len(boundaries)!=1:raise ValueError('One exact reload boundary required')
    scope.body=scope.body[:boundaries[0]]
    function.body=function.body[:index+1]
    # The cut occurs before optimizer construction, reload or any backward call.
    # Inspect nested definitions too so later source changes cannot hide a step.
    forbidden={'backward','step','zero_grad','save_state','restore_state','AdamW'}
    for node in ast.walk(function):
        if isinstance(node,ast.Call):
            name=node.func.id if isinstance(node.func,ast.Name) else (
                node.func.attr if isinstance(node.func,ast.Attribute) else None)
            if name in forbidden:raise ValueError('Optimizer or state mutation before capture')
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            if any(isinstance(t,ast.Name) and t.id in ('optimizer','opt') for t in targets):
                raise ValueError('Optimizer construction before capture')
    scope.body.append(ast.Return(value=ast.Call(func=ast.Name(id='__saved_stage_capture',ctx=ast.Load()),
        args=[ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[])))
    derived=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    # Python 3.13+ omits empty fields by default; 3.12 always includes them.
    # Serialize all fields on both runtimes without changing the executable tree.
    options={'show_empty':True} if 'show_empty' in inspect.signature(ast.dump).parameters else {}
    digest=hashlib.sha256(ast.dump(derived,include_attributes=False,**options).encode()).hexdigest()
    return derived,dict(schema='dl-native-saved-stage-context-prefix-v1',
        original_source_sha256=ORIGINAL_SHA256,derived_ast_sha256=digest,
        cut_before='original args.reload branch',optimizer_branch_included=False,
        model_loaded=False,gpu_execution_authorized=False)


def bind(source, original_globals, capture):
    """Bind only after the caller verifies the isolated original-source checkout."""
    tree,receipt=construction_tree(source)
    namespace=dict(original_globals)
    if '__saved_stage_capture' in namespace:
        raise ValueError('Capture namespace already populated')
    namespace['__saved_stage_capture']=capture
    exec(compile(tree,'<pinned-native-construction-prefix>','exec'),namespace)
    return namespace['run'],receipt
