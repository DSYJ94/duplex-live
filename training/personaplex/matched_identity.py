"""Preserve checkpoint lineage separately from the actual execution producer.

No authorization or numerical proof is created here. Only the admitted origin
is reusable, with byte-identical dependencies and tightly specified identity
changes to the native wrapper. New actual native restoration remains required.
"""
import ast,hashlib,subprocess
from pathlib import Path

ORIGIN='8b822b0d85ccea4f291255204cbc358fa85469c4'
NATIVE='training/personaplex/matched_native.py'
DEPENDENCIES=('training/personaplex/matched_losses.py','training/personaplex/matched_state.py',
    'training/personaplex/matched_schedule.py','training/personaplex/capacity.py',
    'training/personaplex/matched_inputs.py','training/venus/native_optimizer.py',
    'training/venus/opal_adapter.py','training/venus/state.py','evals/personaplex_adapters.py')
META_EDIT=('source=a.source,base=a.base','source=lineage_source(a),base=a.base')
RESTORE_EDITS=[("original['source']!=a.source", "not reference_source_matches(a,original['source'])"),
    ("fingerprint(lm,groups)!=original['fingerprints']", "not compare_fingerprints(a,lm,groups,original,folder)"),
    ("dict(passed=True,source=a.source,actual_CUDA=True,", "dict(passed=True,source=a.source,lineage_source=lineage_source(a),base=a.base,capacity=a.capacity,\n            data_report_sha256=a.inputs_sha,actual_CUDA=True,")]
MAIN_EDITS=[('def main(a):\n    import torch','def main(a):\n    compatibility=validate_lineage(a)\n    import torch'),
    ("    groups,report=corpus(Path(a.inputs),a.inputs_sha)", "    if compatibility:write(folder,'source-compatibility.json',compatibility)\n    groups,report=corpus(Path(a.inputs),a.inputs_sha)")]


def lineage_source(a):
    origin=getattr(a,'lineage_source',None)
    if origin is not None and origin!=ORIGIN:raise ValueError('Only the exact independently admitted origin is reusable')
    return origin or a.source


def reference_source_matches(a,producer):
    from .matched_interrupted import reference_allowed
    from .matched_failed_state import reference_allowed as failed_allowed
    from .matched_normal88 import reference_allowed as normal_allowed
    from .matched_storage_recovery import reference_allowed as storage_allowed
    from .matched_rpc_recovery import reference_allowed as rpc_allowed
    return producer in {a.source,lineage_source(a)} or reference_allowed(a,producer) or failed_allowed(a,producer) or normal_allowed(a,producer) or storage_allowed(a,producer) or rpc_allowed(a,producer)


def compare_fingerprints(a,lm,groups,original,folder):
    """Persist the actual comparison; same single forward and exact equality."""
    from .matched_native import fingerprint,write
    actual=fingerprint(lm,groups);expected=original['fingerprints'];passed=actual==expected
    write(folder,'native-restore-comparison.json',dict(source=a.source,base=a.base,capacity=a.capacity,
        checkpoint_sha256=a.checkpoint_sha,expected=expected,actual=actual,passed=passed,
        criterion='unchanged exact native hidden/text-logit fingerprints',optimizer_updates=0))
    return passed


def function_sources(text):
    return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}


def compare_native(old,new):
    original=function_sources(old);current=function_sources(new);checked=[]
    for name,body in original.items():
        if name=='joint_schedule':continue  # budget/proof orchestration; not model math
        edits=([META_EDIT] if name=='metadata' else RESTORE_EDITS if name=='restore' else MAIN_EDITS if name=='main' else [])
        for before,after in edits:
            if body.count(before)!=1:raise ValueError('Original identity edit no longer uniquely matches: '+name)
            body=body.replace(before,after)
        if name not in current or ast.dump(ast.parse(body),include_attributes=False)!=ast.dump(ast.parse(current[name]),include_attributes=False):
            raise ValueError('Unapproved native math/state/wrapper change: '+name)
        checked.append(name)
    for variable in ('BASE','BASES','CAPACITIES','RECIPE'):
        def assignment(source):
            return next(ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==variable for t in n.targets))
        if assignment(old)!=assignment(new):raise ValueError('Original declared math/recipe changed: '+variable)
    return checked


def validate_lineage(a):
    if not getattr(a,'lineage_source',None):return None
    if a.stage=='admit':raise ValueError('Training-only reuse cannot repeat native admission')
    lineage_source(a)
    root=Path(__file__).resolve().parents[2]
    def git_file(revision,file):return subprocess.check_output(['git','-C',str(root),'show',revision+':'+file])
    identity_file='training/personaplex/matched_identity.py'
    identity_bytes=(root/identity_file).read_bytes()
    if identity_bytes!=git_file(a.source,identity_file):raise ValueError('Identity checker differs from execution producer')
    current=(root/NATIVE).read_bytes()
    if current!=git_file(a.source,NATIVE):raise ValueError('Actual native worker differs from execution producer')
    checked=compare_native(git_file(ORIGIN,NATIVE).decode(),current.decode());hashes={}
    for file in DEPENDENCIES:
        raw=(root/file).read_bytes()
        if raw!=git_file(ORIGIN,file) or raw!=git_file(a.source,file):
            raise ValueError('Original native math/state dependency changed: '+file)
        hashes[file]=hashlib.sha256(raw).hexdigest()
    for file in ('training/personaplex/matched_interrupted.py','training/personaplex/matched_continue_policy.py',
                 'docs/NATIVE-MOSHI-QUOTA-RECOVERY-AUTHORIZATION-2026-11-09.json',
                 'training/personaplex/matched_failed_state.py','training/personaplex/matched_guard.py',
                 'docs/NATIVE-MOSHI-POSTUPDATE-RECOVERY-AUTHORIZATION-2026-11-10.json',
                 'docs/NATIVE-MOSHI-HEARTBEAT-FAILURE-RESULT-2026-11-10.json',
                 'training/personaplex/matched_normal88.py',
                 'docs/NATIVE-MOSHI-NORMAL88-RECOVERY-AUTHORIZATION-2026-11-10.json',
                 'docs/NATIVE-MOSHI-RESTORE89-FAILURE-2026-11-10.json',
                 'training/personaplex/matched_storage_recovery.py','training/personaplex/matched_storage_quota.py',
                 'docs/NATIVE-MOSHI-STORAGE-RECOVERY-AUTHORIZATION-2026-11-10.json',
                 'docs/NATIVE-MOSHI-STORAGE-RECOVERY-PROPOSAL-2026-11-10.json',
                 'docs/NATIVE-MOSHI-STORAGE-STOP-2026-11-10.json',
                 'training/personaplex/matched_rpc_recovery.py','training/personaplex/matched_rpc_startup.py',
    'docs/NATIVE-MOSHI-RPC-MOUNT-CONTINUATION-PROPOSAL-2026-11-11.json',
    'docs/NATIVE-MOSHI-RPC-MOUNT-CONTINUATION-AUTHORIZATION-2026-11-11.json',
    'docs/NATIVE-MOSHI-RPC-STARTUP-CONTINUATION-PROPOSAL-2026-11-11.json',
    'docs/NATIVE-MOSHI-RPC-STARTUP-CONTINUATION-AUTHORIZATION-2026-11-11.json',
                 'docs/NATIVE-MOSHI-RPC-RECOVERY-PROPOSAL-2026-11-11.json',
                 'docs/NATIVE-MOSHI-RPC-RECOVERY-AUTHORIZATION-2026-11-11.json'):
        raw=(root/file).read_bytes()
        if raw!=git_file(a.source,file):raise ValueError('Recovery policy differs from actual producer: '+file)
        hashes[file]=hashlib.sha256(raw).hexdigest()
    return dict(lineage_source=ORIGIN,actual_producer=a.source,checked_native_functions=checked,
        original_dependency_hashes=hashes,identity_checker_sha256=hashlib.sha256(identity_bytes).hexdigest(),only_declared_identity_wrapper_edits=True,
        new_native_restore_required=True,training_authorized=False)
