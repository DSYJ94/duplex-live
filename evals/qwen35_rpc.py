"""Resident local controller over a private Unix socket, with an isolated runtime.

The parent speech process keeps its existing Transformers version. Only this
child imports the newer model runtime. The channel transports actual local
model requests/results and errors; it is neither HTTP nor a remote API fallback.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from evals.recovery_worker import WorkerContractError
from evals.moshi_model_download import sha256

LIMIT=2*1024*1024


def send(stream,packet):
    encoded=json.dumps(packet,allow_nan=False).encode()+b'\n'
    if len(encoded)>LIMIT:raise RuntimeError('Controller RPC packet exceeds bound')
    stream.write(encoded);stream.flush()


def receive(stream):
    encoded=stream.readline(LIMIT+1)
    if not encoded or len(encoded)>LIMIT or not encoded.endswith(b'\n'):raise RuntimeError('Controller RPC closed or returned an incomplete/oversized packet')
    packet=json.loads(encoded)
    if not isinstance(packet,dict):raise RuntimeError('Controller RPC packet is not an object')
    return packet


class ResidentQwen35:
    provider_name='local-transformers-qwen35-9b'
    def __init__(self,assets,folder,device='cuda:1',pending_speech_protocol='explicit-actions'):
        import transformers
        self.parent_version=transformers.__version__
        if self.parent_version!='4.55.3':raise RuntimeError('Resident integration requires the verified parent speech runtime4.55.3')
        self.folder=Path(folder);self.socket_path=self.folder/'controller.sock';self.log_path=self.folder/'controller-process.log'
        if self.socket_path.exists() or len(str(self.socket_path).encode())>100:raise RuntimeError('Controller socket path is occupied or too long')
        self.lock=threading.RLock();self.counter=0;self.connection=None;self.stream=None;self.process=None
        self.log=self.log_path.open('wb');environment=os.environ.copy();environment.pop('OPENAI_API_KEY',None)
        environment.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2')
        command=[sys.executable,'-u','-m','evals.qwen35_rpc','--assets',str(assets),'--socket',str(self.socket_path),
            '--device',device,'--pending-speech-protocol',pending_speech_protocol]
        started=time.monotonic()
        try:
            self.process=subprocess.Popen(command,stdout=self.log,stderr=subprocess.STDOUT,env=environment)
            deadline=started+180
            while True:
                if self.process.poll() is not None:raise RuntimeError('Controller startup failed; full log: '+str(self.log_path))
                connection=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);connection.settimeout(180)
                try:connection.connect(str(self.socket_path));self.connection=connection;break
                except (FileNotFoundError,ConnectionRefusedError):
                    connection.close()
                    if time.monotonic()>=deadline:raise RuntimeError('Controller startup timed out')
                    time.sleep(.1)
            self.stream=self.connection.makefile('rwb');ready=receive(self.stream)
            if (ready.get('kind')!='ready' or ready.get('provider')!=self.provider_name
                    or ready.get('model')!='Qwen/Qwen3.5-9B' or not ready.get('load',{}).get('strict')
                    or not ready['load'].get('all_source_tensors_exact_on_cpu_before_device_copy')
                    or ready.get('pending_speech_protocol')!=pending_speech_protocol):raise RuntimeError('Unverified resident controller handshake')
            expected={name:sha256(Path(__file__).with_name(name)) for name in ('qwen35_rpc.py','qwen35_controller.py','qwen_response_executor.py','qwen_controller_probe.py')}
            if (ready['source_sha256']!=expected or ready['load']['assets_sha256']!=sha256(assets)
                    or ready['load']['device']!=device or ready['load']['trainable_parameters']!=0):raise RuntimeError('Resident controller source, assets or device mismatch')
            self.proof=ready['load'];self.warmup=ready['warmup'];self.model_id=ready['model']
            self.transport_info=dict(kind='private-unix-socket',controller_separate_process=True,pid=self.process.pid,
                parent_transformers_version=self.parent_version,child_transformers_version=ready['transformers_version'],
                startup_s=time.monotonic()-started,child_source_sha256=ready['source_sha256'],log=str(self.log_path),remote_api_used=False)
            self.connection.settimeout(90)
        except BaseException:
            self.close();raise

    def __call__(self,payload):
        with self.lock:
            if self.process.poll() is not None:raise RuntimeError('Resident controller exited; see '+str(self.log_path))
            self.counter+=1;identity=self.counter;send(self.stream,dict(kind='request',id=identity,payload=payload));packet=receive(self.stream)
            if packet.get('id')!=identity:raise RuntimeError('Controller RPC response identity mismatch')
            if packet.get('kind')=='error':raise WorkerContractError('Resident local controller: '+packet['message'],packet.get('result') or dict(error_type=packet['error_type']))
            if packet.get('kind')!='response' or not isinstance(packet.get('response'),dict):raise RuntimeError('Controller RPC response shape invalid')
            import transformers
            if transformers.__version__!=self.parent_version:raise RuntimeError('Parent speech runtime changed')
            return packet['response']

    def close(self):
        if self.connection is not None:self.connection.settimeout(2)
        if self.process is not None and self.process.poll() is None:
            try:
                if self.stream is not None:send(self.stream,dict(kind='shutdown',id=0));receive(self.stream)
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
        if self.stream is not None:self.stream.close();self.stream=None
        if self.connection is not None:self.connection.close();self.connection=None
        if not self.log.closed:self.log.close()
        import stat
        if self.socket_path.exists() and stat.S_ISSOCK(self.socket_path.stat().st_mode):self.socket_path.unlink()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--socket',type=Path,required=True)
    ap.add_argument('--device',default='cuda:1');ap.add_argument('--pending-speech-protocol',choices=('array','explicit-actions'),default='explicit-actions');args=ap.parse_args()
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    try:
        listener.bind(str(args.socket));os.chmod(args.socket,0o600);listener.listen(1)
        from evals.qwen35_controller import Qwen35ResponseExecutor
        executor=Qwen35ResponseExecutor(args.assets,device=args.device,pending_speech_protocol=args.pending_speech_protocol)
        import transformers
        connection,_=listener.accept()
        with connection,connection.makefile('rwb') as stream:
            send(stream,dict(kind='ready',provider=executor.provider_name,model=executor.model_id,load=executor.proof,warmup=executor.warmup,
                pending_speech_protocol=args.pending_speech_protocol,transformers_version=transformers.__version__,
                source_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                    ('qwen35_rpc.py','qwen35_controller.py','qwen_response_executor.py','qwen_controller_probe.py')}))
            while True:
                packet=receive(stream);identity=packet.get('id')
                if packet.get('kind')=='shutdown':send(stream,dict(kind='shutdown',id=identity));break
                if packet.get('kind')!='request' or not isinstance(identity,int) or isinstance(identity,bool) or not isinstance(packet.get('payload'),dict):
                    raise RuntimeError('Invalid controller request envelope')
                try:send(stream,dict(kind='response',id=identity,response=executor(packet['payload'])))
                except Exception as error:send(stream,dict(kind='error',id=identity,error_type=type(error).__name__,message=str(error),result=getattr(error,'result',None)))
    finally:
        listener.close()
        if args.socket.exists():args.socket.unlink()


if __name__=='__main__':main()
