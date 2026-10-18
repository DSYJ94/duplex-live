"""Reuse an exact historical input approval through its immutable Git validator.

Changing the assistant does not regenerate or relabel evaluation audio. The
original quality validator must belong to this Git repository, match its pinned
commit byte for byte and have a clean worktree. Its original source/quality/hash
checks execute unchanged. No code is copied to a remote host by this loader.
"""
import importlib.util
from pathlib import Path
import subprocess
from evals.moshi_model_download import sha256


def cases(data,quality,identities,source=None,commit=None):
    if (source is None)!=(commit is None):raise ValueError('Historical validator needs both source and commit')
    if source is None:
        from evals.duplex_fresh_inputs import checked_cases
        module=Path(__file__).with_name('duplex_fresh_inputs.py')
        return checked_cases(data,quality,identities),dict(source=str(module),sha256=sha256(module),commit=None)
    def call(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
    root=Path(__file__).resolve().parents[1];source=source.resolve()
    def common(folder):
        path=Path(call(folder,'rev-parse','--git-common-dir'))
        return (path if path.is_absolute() else folder/path).resolve()
    if (common(source)!=common(root) or call(source,'rev-parse','HEAD')!=commit
            or call(source,'status','--porcelain')):
        raise RuntimeError('Input approval validator is not the declared clean worktree in this repository')
    module=source/'evals/duplex_fresh_inputs.py'
    original=subprocess.check_output(['git','-C',str(source),'show',commit+':evals/duplex_fresh_inputs.py'])
    if module.read_bytes()!=original:raise RuntimeError('Historical input validator differs from immutable Git source')
    spec=importlib.util.spec_from_file_location('dl_pinned_input_validator_'+commit,module)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    selected=loaded.checked_cases(data,quality,identities)
    return selected,dict(source=str(module),sha256=sha256(module),commit=commit,original_quality_checks_unchanged=True)
