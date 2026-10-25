"""Bounded isolated GPU recognition for live native requests; no saved history."""
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import uuid


class IsolatedLiveASR:
    def __init__(self, python, source, timeout=90):
        self.temporary = tempfile.TemporaryDirectory(prefix='duplex-live-asr-')
        self.lock = threading.Lock()
        self.timeout = timeout
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
        env.pop('OPENAI_API_KEY', None)
        try:
            self.process = subprocess.Popen([str(python), '-u', '-m', 'evals.context_spanning_asr',
                '--root', self.temporary.name], env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        except BaseException:
            self.temporary.cleanup()
            raise
        try:
            self.ready = self._read()
            if not self.ready.get('ready'):
                raise RuntimeError('Isolated ASR did not start')
        except BaseException:
            self.close()
            raise

    def _read(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            self.process.terminate()
            raise TimeoutError('Isolated recognition timed out; no automatic restart')
        line = self.process.stdout.readline(1024*1024)
        if not line or not line.endswith('\n'):
            raise RuntimeError('Invalid isolated recognition response')
        return json.loads(line)

    def transcribe(self, pcm, sample_rate):
        import numpy as np
        import soundfile as sf
        if sample_rate != 24000 or pcm.ndim != 1 or not 0 < len(pcm) <= 60*24000 or not np.isfinite(pcm).all():
            raise ValueError('Actual finite arrived audio of at most sixty seconds required')
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError('Isolated recognition has stopped')
            identity = uuid.uuid4().hex
            path = Path(self.temporary.name)/(identity+'.wav')
            try:
                sf.write(path, pcm, sample_rate, subtype='FLOAT')
                self.process.stdin.write(json.dumps(dict(id=identity, path=str(path)))+'\n')
                self.process.stdin.flush()
                value = self._read()
                if value['id'] != identity or value['input_samples'] != len(pcm):
                    raise RuntimeError('Recognition/source identity mismatch')
                return value
            finally:
                path.unlink(missing_ok=True)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()
        self.temporary.cleanup()
