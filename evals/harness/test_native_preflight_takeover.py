"""Actual Linux process handoff: preserve child exit before retiring its parent."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from portable.context.native_preflight_takeover import process,pause,resume,retire


@unittest.skipUnless(Path('/proc/self/stat').exists(),'Actual Linux process boundary required')
class NativeTakeover(unittest.TestCase):
    def test_exact_supervisor_pause_preserves_same_child_and_real_exit_status(self):
        code='import subprocess,sys,json; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(.3)"]); print(json.dumps({"child":p.pid}),flush=True); p.wait(timeout=10)'
        parent=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,text=True)
        child=json.loads(parent.stdout.readline())['child'];birth=process(parent.pid)['birth'];child_birth=process(child)['birth']
        try:
            with self.assertRaises(RuntimeError):pause(parent.pid,birth+1)
            pause(parent.pid,birth)
            self.assertIn(process(parent.pid)['state'],('T','t'))
            end=time.monotonic()+3
            while process(child)['state']!='Z' and time.monotonic()<end:time.sleep(.01)
            actual=process(child)
            self.assertEqual(actual['birth'],child_birth)
            self.assertEqual(actual['state'],'Z')
            self.assertEqual(__import__('os').waitstatus_to_exitcode(actual['exit_status']),0)
            retire(parent.pid,birth)
            self.assertEqual(parent.wait(timeout=3),-signal.SIGTERM)
        finally:
            if parent.poll() is None:
                resume(parent.pid,birth);parent.terminate();parent.wait(timeout=3)
            parent.stdout.close()


if __name__=='__main__':unittest.main()
