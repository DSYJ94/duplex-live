"""Reuse pure source readbacks within ONE unchanged, fully hashed input snapshot.

No result survives the context. Every distinct source/quality/strictness tuple
runs its original checker; every teacher recording still runs its full checker.
Frozen source modules and their artifact code hashes are never edited.
"""
from contextlib import contextmanager
from copy import deepcopy
import functools
import importlib
import inspect
from pathlib import Path
import sys

from evals.native_reasoning_source import sha

SOURCE_MODULES = ('native_reasoning_source', 'native_reasoning_source_recovery',
                  'native_reasoning_source_repair', 'native_reasoning_source_pacing',
                  'native_reasoning_source_quality', 'native_reasoning_source_quality_v4')


def inventory(roots):
    rows = []
    for root in sorted(set(Path(p).absolute() for p in roots)):
        if root.is_symlink() or not root.exists():
            raise ValueError('Existing nonsymlink verification input required')
        paths = [root] if root.is_file() else sorted(root.rglob('*'))
        for path in paths:
            if path.is_symlink():
                raise ValueError('Verification snapshot cannot follow symlinks')
            if path.is_file():
                rows.append((str(path), path.stat().st_size, sha(path)))
    return rows


class Memo:
    def __init__(self, roots):
        self.roots = tuple(Path(p).absolute() for p in roots)
        self.calls = {}; self.hits = {}; self.cache = {}

    def wrap(self, function):
        signature = inspect.signature(function)
        name = function.__module__ + '.' + function.__name__

        @functools.wraps(function)
        def checked(*args, **kwargs):
            bound = signature.bind(*args, **kwargs); bound.apply_defaults()
            key = []
            for parameter, value in bound.arguments.items():
                if isinstance(value, Path):
                    value = value.absolute()
                    if '..' in value.parts or not any(value == root or value.is_relative_to(root) for root in self.roots):
                        raise ValueError('Source readback escaped the complete input snapshot')
                    value = ('path', str(value))
                elif type(value) not in (bool, str, int, type(None)):
                    raise ValueError('Unexpected mutable source-verifier argument')
                key.append((parameter, type(value).__name__, value))
            key = (function, tuple(key))
            if key in self.cache:
                self.hits[name] = self.hits.get(name, 0) + 1
                return deepcopy(self.cache[key])
            self.calls[name] = self.calls.get(name, 0) + 1
            result = function(*args, **kwargs)
            self.cache[key] = deepcopy(result)
            return result
        return checked


@contextmanager
def source_readbacks(roots, *, include_old_speech=False):
    # Load the real history chain BEFORE replacing imported aliases.
    importlib.import_module('evals.native_reasoning_teacher_recovery_v6')
    importlib.import_module('evals.native_reasoning_teacher_cutoff_readback')
    modules = [importlib.import_module('evals.' + n) for n in SOURCE_MODULES]
    if include_old_speech:
        modules.append(importlib.import_module('evals.native_reasoning_teacher_quality'))
    code = [Path(m.__file__) for m in sys.modules.copy().values()
            if getattr(m, '__name__', '').startswith('evals.native_reasoning_')
            and getattr(m, '__file__', None)]
    code.append(Path(__file__).resolve().parents[1] / 'data/synthetic/filter.py')
    original = inventory([*roots, *code]); memo = Memo(roots)
    replacements = {m.verify: memo.wrap(m.verify) for m in modules}
    patched = []
    try:
        for name, module in list(sys.modules.items()):
            if not name.startswith('evals.native_reasoning_'): continue
            for attribute, value in list(vars(module).items()):
                if inspect.isfunction(value) and value in replacements:
                    patched.append((module, attribute, value))
                    setattr(module, attribute, replacements[value])
        yield memo
        if inventory([*roots, *code]) != original:
            raise ValueError('Verification inputs or source code changed during cached readback')
    finally:
        for module, attribute, value in reversed(patched): setattr(module, attribute, value)
        memo.cache.clear()
