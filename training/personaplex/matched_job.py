"""One actual two-GPU Slurm job: parallel bases, serial capacities per GPU.

Initial launch only. Later resumes must bind real post-update checkpoints and
all carried Slurm usage explicitly; this entry point cannot reset that budget.
Host Python3.6, with a separate scheduler guard and real child terminals.
"""
import argparse
from datetime import datetime,timezone
import json,os,signal,subprocess,sys,threading,time,uuid
from pathlib import Path
from . import matched_guard as g
from .matched_paths import BASE,command


def write(path,value):
    path=Path(path);temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(str(temporary),str(path))


def visible_pair(value):
    devices=value.split(',')
    if len(devices)!=2 or len(set(devices))!=2 or any(not d.strip() for d in devices):
        raise ValueError('Exactly two distinct actual scheduler-selected GPUs required')
    return devices


def prior_usage():
    # This initial-only entry admits exactly the already completed source48
    # allocation. A future resume cannot silently substitute327 again.
    path=Path('docs/NATIVE-PERSONAPLEX-CAPTURE48-RESULT-2026-11-08.json')
    r=json.loads(path.read_text())
    if (r['job']!='8216374' or r['source']!='96850d5029f98098d8144d534f44f0aefeabb57b'
            or r['actual_Slurm_state']!='COMPLETED' or r['actual_exit']!='0:0'
            or r['allocated_GPU_seconds']!=327 or not r['fresh_input_reconstruction']['passed']):
        raise ValueError('Exact actual prior source48 allocation required')
    actual=subprocess.check_output(['sacct','-n','-P','-j','8216374','--format=JobIDRaw,Account,State,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=20)
    row=next((line.split('|') for line in actual.splitlines() if line.split('|')[0]=='8216374'),None)
    if (not row or row[1]!=g.ACCOUNT or row[2]!='COMPLETED' or row[3]!='327'
            or 'gres/gpu=1' not in row[4].split(',')):
        raise ValueError('Actual scheduler accounting differs from preserved prior usage')
    return 327


class Children:
    def __init__(self,work,source,execution,lease,watcher,devices,inputs,inputs_sha):
        self.work,self.source,self.execution,self.lease,self.watcher=work,source,execution,lease,watcher
        self.guard_path=self.execution/'guard.json'
        self.devices,self.inputs,self.inputs_sha=devices,inputs,inputs_sha
        self.active={};self.lock=threading.Lock();self.failed=threading.Event()

    def ensure(self):
        if self.failed.is_set():raise RuntimeError('Another native stage failed; preserve without duplicating')
        if self.watcher.poll() is not None:raise RuntimeError('Independent matched scheduler guard exited')
        g.ensure(self.lease,json.loads(self.guard_path.read_text()))

    def run(self,base,capacity,stage,folder,extra=(),label=None):
        self.ensure();name=base+'-'+capacity+'-'+(label or stage);execution_id=uuid.uuid4().hex
        env=dict(SLURM_JOB_ID=self.lease['job_id'],SLURM_JOB_ACCOUNT=g.ACCOUNT,
            DUPLEX_LIVE_LEASE_FILE=str(self.execution/'lease.json'),
            DUPLEX_LIVE_GUARD_FILE=str(self.guard_path),
            DUPLEX_NATIVE_EXECUTION_ID=execution_id,CUBLAS_WORKSPACE_CONFIG=':4096:8')
        cmd=command(self.work,gpu=self.devices[base],environment=env)+[
            'training.personaplex.matched_native','--source',self.source,'--root',str(folder),
            '--lease',str(self.execution/'lease.json'),'--base',base,'--capacity',capacity,
            '--inputs',self.inputs,'--inputs-sha',self.inputs_sha,'--stage',stage]+list(extra)
        began=time.time();actual=None;child=None
        with (self.execution/(name+'.log')).open('x') as log:
            try:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                with self.lock:self.active[base]=child
                identity=dict(source=self.source,base=base,capacity=capacity,stage=stage,pid=child.pid,
                    execution_id=execution_id,started_unix=began,module='training.personaplex.matched_native')
                write(self.execution/(base+'-current.json'),identity)
                while child.poll() is None:self.ensure();time.sleep(5)
                actual=child.returncode
            finally:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=90)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    actual=child.returncode
                if child is not None:
                    write(self.execution/(name+'-terminal.json'),dict(identity,actual_exit_code=actual,
                        ended_unix=time.time(),seconds=time.time()-began))
                with self.lock:self.active.pop(base,None)
        if actual!=0:raise RuntimeError('Actual native child failed: '+name)
        return self.execution/(name+'-terminal.json')

    def parallel(self,action):
        errors=[];lock=threading.Lock()
        def run(base):
            try:action(base)
            except BaseException as exc:
                self.failed.set()
                with lock:errors.append((base,str(exc)))
        threads=[threading.Thread(target=run,args=(base,)) for base in ('personaplex','context-spanning')]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        if errors:raise RuntimeError('Native parallel phase failed: '+repr(errors))


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source:raise ValueError('Exact Git native host producer required')
    root=Path(a.root);root.resolve().relative_to(BASE.resolve())
    root.mkdir(parents=True,exist_ok=False)  # Never re-enter an earlier allocation.
    job_id=os.environ['SLURM_JOB_ID']
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',job_id],universal_newlines=True,timeout=20))
    execution=root/('execution-'+job_id);execution.mkdir()
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,
        authorization_sha256=g.sha(g.AUTH),job_id=job_id,node=job['NodeList'],source=source,
        start_unix=g.stamp(job['StartTime']),stop_unix=g.stamp(job['EndTime']),
        maximum_seconds=86400,GPU_count=2,prior_GPU_seconds=prior_usage())
    g.validate_job(job,lease,time.time());write(execution/'lease.json',lease)
    devices=visible_pair(os.environ['CUDA_VISIBLE_DEVICES'])
    watcher=subprocess.Popen([sys.executable,'-B','-u','-m','training.personaplex.matched_guard',
        '--lease',str(execution/'lease.json'),'--output',str(execution/'guard.json')],
        stdout=(execution/'guard.log').open('x'),stderr=subprocess.STDOUT)
    code=1
    try:
        until=time.time()+40
        while not (execution/'guard.json').exists():
            if watcher.poll() is not None or time.time()>until:raise RuntimeError('Independent two-GPU guard did not start')
            time.sleep(1)
        children=Children(Path.cwd(),source,execution,lease,watcher,
            dict(zip(('personaplex','context-spanning'),devices)),a.inputs,a.inputs_sha)
        entries=[];lock=threading.Lock()
        def admit(base):
            for capacity in ('narrow','broad'):
                lineage=root/base/capacity;lineage.mkdir(parents=True)
                at=children.run(base,capacity,'admit',lineage/'admit')
                checkpoint=lineage/'admit/init.pt';digest=g.sha(checkpoint)
                rt=children.run(base,capacity,'restore',lineage/'restore',[
                    '--checkpoint',str(checkpoint),'--checkpoint-sha',digest])
                entry=dict(base=base,capacity=capacity,admission=str(lineage/'admit/admission.json'),
                    admission_sha256=g.sha(lineage/'admit/admission.json'),admission_terminal=str(at),
                    restore=str(lineage/'restore/restore.json'),restore_sha256=g.sha(lineage/'restore/restore.json'),
                    restore_terminal=str(rt),training_terminal=str(lineage/'train/training-terminal.json'))
                with lock:entries.append(entry)
        children.parallel(admit)
        packet=root/'joint-admission.json'
        write(packet,dict(source=source,data_report_sha256=a.inputs_sha,entries=sorted(entries,key=lambda r:(r['base'],r['capacity']))))
        partial=[]
        def train(base):
            for capacity in ('narrow','broad'):
                lineage=root/base/capacity;init=lineage/'admit/init.pt'
                children.run(base,capacity,'train',lineage/'train',[
                    '--checkpoint',str(init),'--checkpoint-sha',g.sha(init),
                    '--restore-proof',str(lineage/'restore/restore.json'),
                    '--restore-proof-sha',g.sha(lineage/'restore/restore.json'),
                    '--joint-admission',str(packet),'--joint-admission-sha',g.sha(packet)])
                window=lineage/'train/training-window-terminal.json'
                terminal=json.loads((window if window.exists() else lineage/'train/training-terminal.json').read_text())
                saved=Path(terminal['checkpoint'] if window.exists() else terminal['final_checkpoint'])
                children.run(base,capacity,'restore',lineage/'final-restore',[
                    '--checkpoint',str(saved),'--checkpoint-sha',terminal['checkpoint_sha256']],label='final-restore')
                if window.exists():
                    with lock:partial.append(dict(base=base,capacity=capacity,**terminal))
                    return
        children.parallel(train)
        write(root/'phase-result.json',dict(source=source,all_four_complete=not partial,
            partial_lineages=partial,full_native_behavior_speech_backend_live_required=True,adopted=False))
        code=0
    finally:
        watcher.terminate()
        try:watcher.wait(timeout=10)
        except subprocess.TimeoutExpired:watcher.kill();watcher.wait()
        write(execution/'terminal.json',dict(source=source,actual_exit_code=code,job_id=job_id,
            ended_UTC=datetime.now(timezone.utc).isoformat(),adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','root','inputs','inputs-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
