"""Record files read by an offline validation command for migration planning.

This observes Python open events, not subprocess/C-level IO. A complete restore
test is still required; this manifest alone is not a dependency-closure proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--module', required=True)
    ap.add_argument('args', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    logical_root = args.root.absolute(); physical_root = args.root.resolve()
    observed = set(); active = True
    allowed = {'runs', 'models', 'vendor', 'worktrees', 'repo'}

    def on_open(event, values):
        if not active or event != 'open': return
        path, mode, flags = values
        if not isinstance(path, (str, bytes, os.PathLike)): return
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC): return
        path = Path(os.fsdecode(path)).absolute()
        for root in (logical_root, physical_root):
            try: relative = path.relative_to(root)
            except ValueError: continue
            if not relative.parts or relative.parts[0] not in allowed:
                raise RuntimeError('Refusing to inventory private/non-artifact root: '+str(relative))
            if path.is_file(): observed.add((str(relative), str(path)))
            break

    sys.addaudithook(on_open)
    argv = args.args[1:] if args.args[:1] == ['--'] else args.args
    sys.argv = [args.module, *argv]
    completed = False
    try:
        try: runpy.run_module(args.module, run_name='__main__', alter_sys=True)
        except SystemExit as error:
            if error.code not in (None, 0): raise
        completed = True
    finally:
        active = False
        files = []
        for relative, path in sorted(observed):
            digest = hashlib.sha256()
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
            files.append({'path': relative, 'bytes': Path(path).stat().st_size,
                'sha256': digest.hexdigest()})
        manifest = {'schema': 'dl-observed-artifact-reads-v1',
            'compatibility_root': str(logical_root), 'module': args.module,
            'command_completed': completed, 'files': files,
            'subprocess_and_native_io_covered': False,
            'complete_dependency_closure_verified': False,
            'restore_test_required': True}
        args.out.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__': main()
