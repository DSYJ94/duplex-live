"""Independent local scheduler heartbeat and bounded shared-storage admission.

Storage scans and durable status mirrors cannot block live scheduler renewal.
The existing 45-second scheduler gate, time/GPU caps and storage limits remain.
No model, optimizer, data selection, submission or restart code lives here.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from . import matched_guard as g
from .matched_job import write

STORAGE_TTL=120


def storage_valid(lease,path,now=None):
    now=time.time() if now is None else now
    value=json.loads(Path(path).read_text())
    if (value.get('job_id')!=lease['job_id'] or value.get('source')!=lease['source']
            or value.get('passed') is not True or not 0<=now-value['checked_unix']<STORAGE_TTL
            or type(value['home_bytes']) is not int or not 0<=value['home_bytes']<=80*1024**3
            or type(value['run_bytes']) is not int or not 0<=value['run_bytes']<=32*1024**3
            or value.get('separate_filesystem') is not True):
        raise RuntimeError('Independent storage admission failed, expired or changed identity')
    return value


def heartbeat_once(lease,output,storage,last=None):
    # This path reads only a tiny node-local receipt. It never walks NFS.
    storage_valid(lease,storage)
    checked=g.refresh(lease,output,last)
    # On timeout, retry within the ORIGINAL remaining freshness window.
    # Sleeping another ten seconds after a timeout can consume that window.
    delay=0 if checked==last else max(0,min(10,checked+35-time.time()))
    return checked,delay


def watch(lease,output,storage):
    last=None;initial_deadline=time.time()+40
    while True:
        try:last,delay=heartbeat_once(lease,output,storage,last)
        except subprocess.TimeoutExpired:
            if last is not None or time.time()>=initial_deadline:raise
            continue  # No guard exists and no model may load until a real successful check.
        time.sleep(delay)


def probe_storage(lease,output):
    from .matched_quota_recovery import storage_check
    if lease.get('storage_recovery_authorization_sha256'):
        from .matched_storage_quota import storage_check
    while True:
        began=time.monotonic()
        try:
            value=storage_check()
            value.update(job_id=lease['job_id'],source=lease['source'],passed=True,
                checked_unix=time.time(),probe_seconds=time.monotonic()-began)
        except BaseException as exc:
            write(output,dict(job_id=lease['job_id'],source=lease['source'],passed=False,
                checked_unix=time.time(),error=type(exc).__name__+': '+str(exc)));raise
        write(output,value);time.sleep(15)


def mirror(lease,local,output):
    while True:
        if Path(local).exists():
            value=json.loads(Path(local).read_text())
            if value.get('job_id')!=lease['job_id'] or value.get('source')!=lease['source']:
                raise ValueError('Wrong local heartbeat identity')
            # Monitoring copy only; workers read LOCAL, never this NFS mirror.
            write(output,value)
            storage=Path(local).parent/'storage.json'
            if storage.exists():write(Path(output).parent/'storage-last-observation.json',json.loads(storage.read_text()))
        time.sleep(10)


def stop(services,control,execution):
    for child in services:
        if child.poll() is None:child.terminate()
    for child in services:
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    path=Path(control)/'guard.json'
    if path.exists():write(Path(execution)/'guard.json',json.loads(path.read_text()))
    storage=Path(control)/'storage.json'
    if storage.exists():write(Path(execution)/'storage-last-observation.json',json.loads(storage.read_text()))


def start(lease,execution):
    """Create a private node-local heartbeat, keeping durable evidence on NFS."""
    from .matched_interrupted import DURABLE
    g.resource(lease);execution=Path(execution)
    control=Path('/tmp')/('duplex-matched-'+lease['job_id']+'-'+lease['source'][:12])
    control.mkdir(mode=0o700,exist_ok=False)
    if control.stat().st_dev==DURABLE.stat().st_dev:raise ValueError('Heartbeat must use independent node-local storage')
    write(control/'lease.json',lease);services=[]
    os.environ['APPTAINER_BINDPATH']=str(DURABLE)+':'+str(DURABLE)+','+str(control)+':'+str(control)
    def launch(mode,extra):
        command=[sys.executable,'-B','-u','-m','training.personaplex.matched_health','--mode',mode,
            '--lease',str(control/'lease.json')]+extra
        with (execution/('health-'+mode+'.log')).open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        services.append(child);return child
    def wait(path,child,seconds):
        deadline=time.time()+seconds
        while not path.exists():
            if child.poll() is not None or time.time()>=deadline:raise RuntimeError('Independent local health service failed: '+str(path))
            time.sleep(.2)
    try:
        probe=launch('storage',['--output',str(control/'storage.json')]);wait(control/'storage.json',probe,60)
        storage_valid(lease,control/'storage.json')
        watcher=launch('watch',['--output',str(control/'guard.json'),'--storage',str(control/'storage.json')])
        wait(control/'guard.json',watcher,45)
        g.ensure(lease,json.loads((control/'guard.json').read_text()))
        launch('mirror',['--local',str(control/'guard.json'),'--output',str(execution/'guard.json')])
        write(execution/'health-services.json',dict(job_id=lease['job_id'],source=lease['source'],
            local_control_dir=str(control),local_guard=str(control/'guard.json'),
            service_PIDs=[p.pid for p in services],NFS_mirror_is_not_native_authority=True))
        return watcher,services,control
    except BaseException:stop(services,control,execution);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('watch','storage','mirror'),required=True)
    for name in ('lease','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--storage');p.add_argument('--local');a=p.parse_args()
    lease=g.resource(json.loads(Path(a.lease).read_text()))
    if a.mode=='watch':watch(lease,a.output,a.storage)
    elif a.mode=='storage':probe_storage(lease,a.output)
    else:mirror(lease,a.local,a.output)
