"""Independent local scheduler heartbeat and bounded shared-storage admission.

Storage scans and durable status mirrors cannot block live scheduler renewal.
The existing 45-second scheduler gate, time/GPU caps and storage limits remain.
No model, optimizer, data selection, submission or restart code lives here.
"""
import argparse,json,subprocess,time
from pathlib import Path
from . import matched_guard as g
from .matched_job import write

STORAGE_TTL=120


def storage_valid(lease,path,now=None):
    now=time.time() if now is None else now
    value=json.loads(Path(path).read_text())
    if (value.get('job_id')!=lease['job_id'] or value.get('source')!=lease['source']
            or value.get('passed') is not True or not 0<=now-value['checked_unix']<STORAGE_TTL
            or value['home_bytes']>80*1024**3 or value['run_bytes']>32*1024**3
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
    last=None
    while True:
        last,delay=heartbeat_once(lease,output,storage,last);time.sleep(delay)


def probe_storage(lease,output):
    from .matched_quota_recovery import storage_check
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
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('watch','storage','mirror'),required=True)
    for name in ('lease','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--storage');p.add_argument('--local');a=p.parse_args()
    lease=g.resource(json.loads(Path(a.lease).read_text()))
    if a.mode=='watch':watch(lease,a.output,a.storage)
    elif a.mode=='storage':probe_storage(lease,a.output)
    else:mirror(lease,a.local,a.output)
