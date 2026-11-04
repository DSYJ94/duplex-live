"""Copy immutable, atomically published checkpoints while training continues."""
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time

from .large_inputs import sha


class CheckpointStream:
    def __init__(self,ssh_argv,remote,backup,receipts,source):
        self.ssh=list(ssh_argv);self.remote=remote;self.backup=Path(backup);self.receipts=Path(receipts)
        self.stop_event=threading.Event();self.verified={};self.bytes=0
        self.backup.mkdir(parents=True,exist_ok=False)
        binding=dict(source=source,remote_output=remote,backup=str(self.backup))
        with (self.receipts/'stream-backup-binding.json').open('x') as stream:json.dump(binding,stream)
        self.thread=threading.Thread(target=self.run,daemon=True)

    def invoke(self,args,seconds):
        proc=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        deadline=time.monotonic()+seconds
        try:
            while True:
                if self.stop_event.is_set():raise InterruptedError('Final copyback takes ownership')
                if time.monotonic()>deadline:raise TimeoutError('Checkpoint copy timed out; durable source retained')
                try:out,err=proc.communicate(timeout=1);break
                except subprocess.TimeoutExpired:continue
            if proc.returncode:raise RuntimeError(err[-1200:])
            return out
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.communicate()

    def record(self,value):
        with (self.receipts/'stream-backup.jsonl').open('a') as stream:stream.write(json.dumps(dict(at_unix=time.time(),**value))+'\n')

    def run(self):
        try:
            while not self.stop_event.is_set():
                code="""from pathlib import Path
import hashlib,json
root=Path(%r);seen=set(%r);rows=[]
for p in [root/'expanded-init.pt']+sorted(root.glob('main/step-*.pt'))+sorted(root.glob('control/step-*.pt')):
 if not p.is_file() or p.is_symlink():continue
 relative=str(p.relative_to(root))
 if relative in seen:continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 rows.append(dict(path=relative,bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(rows))
"""%(self.remote,list(self.verified))
                rows=json.loads(self.invoke(self.ssh+['/usr/bin/python3 -c '+shlex.quote(code)],600))
                for row in rows:
                    path=Path(row['path'])
                    if path.is_absolute() or '..' in path.parts or path.suffix!='.pt':raise ValueError('Exact checkpoint path required')
                    if self.bytes+row['bytes']>100_000_000_000:raise ValueError('Approved100GB artifact ceiling')
                    target=self.backup/path;target.parent.mkdir(parents=True,exist_ok=True)
                    args=['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--partial-dir=.rsync-partial',
                          '-e',shlex.join(self.ssh[:-1]),self.ssh[-1]+':'+self.remote+'/'+str(path),str(target)]
                    self.invoke(args,1800)
                    if target.stat().st_size!=row['bytes'] or sha(target)!=row['sha256']:raise ValueError('Streamed checkpoint hash differs')
                    self.verified[row['path']]=row;self.bytes+=row['bytes'];self.record(dict(checkpoint_verified=row))
                self.stop_event.wait(60)
        except InterruptedError:pass
        except Exception as error:self.record(dict(error_type=type(error).__name__,error=str(error),final_full_copyback_required=True))

    def start(self):self.thread.start();return self
    def stop(self):
        self.stop_event.set();self.thread.join(timeout=30)
        if self.thread.is_alive():raise RuntimeError('Wait for existing checkpoint transfer before final copyback')
