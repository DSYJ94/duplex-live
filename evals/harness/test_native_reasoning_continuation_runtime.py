"""Runtime isolation and early rejection, without loading a model or provider."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from portable.context.native_reasoning_continuation_runtime import execute, _independent_identity


class ContinuationRuntime(unittest.TestCase):
    def test_original_grant_rejects_before_guard_source_or_model_work(self):
        root = Path(__file__).resolve().parents[2]
        def forbidden():
            self.fail('Provider/guard must not be touched for the old grant')
        with self.assertRaises(ValueError):
            execute(None, None, None, None, None,
                root / 'docs/GCP-NATIVE-CONTINUATION-AUTHORIZATION-2026-10-24.json',
                forbidden, None)

    def test_fresh_identity_requires_distinct_process_same_host_and_boot(self):
        first = dict(pid=10, birth=100, host='test-host', boot_id='test-boot')
        second = dict(first, pid=11, birth=101)
        _independent_identity(first, second)
        for changed in (first, dict(second, birth=100), dict(second, host='other'),
                        dict(second, boot_id='other')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _independent_identity(first, changed)

    def test_isolation_uses_original_modules_and_only_adds_missing_producer_helpers(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as name:
            source, producer = Path(name) / 'original', Path(name) / 'producer'
            for folder in (source, producer):
                (folder / 'evals').mkdir(parents=True)
            (source / 'evals/path_probe.py').write_text("VALUE='ORIGINAL'\n")
            (producer / 'evals/path_probe.py').write_text("VALUE='WRONG_PRODUCER_OVERRIDE'\n")
            (producer / 'evals/new_continuation_probe.py').write_text("VALUE='NEW_HELPER'\n")
            script = '''
import importlib,sys
from pathlib import Path
from unittest.mock import patch
from portable.context import native_reasoning_continuation_runtime as runtime
def git(root,*args):
    return runtime.SOURCE if args==('rev-parse','HEAD') else ''
with patch.object(runtime,'git',side_effect=git):
    runtime._isolate(Path(sys.argv[1]),Path(sys.argv[2]))
assert importlib.import_module('evals.path_probe').VALUE=='ORIGINAL'
assert importlib.import_module('evals.new_continuation_probe').VALUE=='NEW_HELPER'
assert 'torch' not in sys.modules
'''
            result = subprocess.run([sys.executable, '-B', '-c', script, str(source), str(producer)],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_preloaded_model_runtime_cannot_be_silently_reused(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as name:
            script = '''
import sys,types
from pathlib import Path
from unittest.mock import patch
from portable.context import native_reasoning_continuation_runtime as runtime
def git(root,*args):
    return runtime.SOURCE if args==('rev-parse','HEAD') else ''
sys.modules['torch']=types.ModuleType('torch')
with patch.object(runtime,'git',side_effect=git):
    try:
        runtime._isolate(Path(sys.argv[1]),Path(sys.argv[1]))
    except ValueError as error:
        assert 'Fresh process' in str(error)
    else:
        raise AssertionError('Preloaded Torch was accepted')
'''
            result = subprocess.run([sys.executable, '-B', '-c', script, name],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
