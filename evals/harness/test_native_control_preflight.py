"""Actual subprocess boundary must retain the compiler check's CUDA paths."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from portable.context.native_control_preflight import native_environment


class NativePreflightEnvironment(unittest.TestCase):
    def test_separate_child_gets_the_verified_assembler_and_header_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);header=root/'include/cuda.h';header.parent.mkdir();header.write_text('test-header')
            assembler=root/'bin/ptxas';assembler.parent.mkdir();assembler.write_text('test-assembler')
            env=os.environ.copy();env.pop('TRITON_PTXAS_PATH',None);env['CPATH']='/existing/include'
            configured=native_environment(dict(header=str(header),assembler=str(assembler)),env)
            code="import os,json; print(json.dumps({k:os.environ.get(k) for k in ('TRITON_PTXAS_PATH','CPATH')}))"
            child=json.loads(subprocess.check_output([sys.executable,'-c',code],env=configured,text=True,timeout=5))
            self.assertEqual(child['TRITON_PTXAS_PATH'],str(assembler))
            self.assertEqual(child['CPATH'].split(os.pathsep),[str(header.parent),'/existing/include'])
            self.assertNotIn('TRITON_PTXAS_PATH',env)


if __name__=='__main__':unittest.main()
