"""Recover preserved reports during the SAME approved GPU interval, before models.

Does not publish a model lease. Native admission still requires the complete
original hash proof. No additional machine, start, duration or payload is added.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import time

from . import british_speech_guard as g,results_recovery as r


def precheck():
    a=g.authority()
    if a.get('files_first_same_approved_GPU') is not True:raise ValueError('Exact files-first execution record required')
    base=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v75/resume12')
    for arm,digest in g.CHECKPOINTS.items():
        if r.sha(base/arm/'step-0060.pt')!=digest:raise ValueError('Critical final training checkpoint backup changed')
    return r.manifest()


def file_guard():
    from . import controller as c
    from .british_controller import R,NAME,ZONE,stop_exact
    lease=json.loads((R/'file-lease.json').read_text());handed=False
    g.authority()
    try:
        while True:
            vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            if vm['id']!=g.VM_ID:raise ValueError('Files-first GPU identity changed')
            if vm['status']=='TERMINATED':return
            start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
            if vm['status']!='RUNNING' or start!=lease['start_unix'] or lease['stop_unix']!=start+3270 or time.time()>lease['stop_unix']-100:raise ValueError('Files-first actual clock mismatch')
            if c.run(['/bin/ps','-p',str(lease['controller_pid']),'-o','lstart=']).strip()!=lease['controller_birth']:raise ValueError('Files-first controller gone')
            disk=c.cloud('compute','disks','describe','duplex-venus-large-data-20261104-eua-v67','--zone='+ZONE)
            if disk['id']!=g.DISK_ID or int(disk['sizeGb'])!=250:raise ValueError('Files-first disk changed')
            active=[x for x in c.cloud('compute','instances','list') if x['status'] not in ('STOPPED','TERMINATED') and x.get('guestAccelerators')]
            if len(active)!=1 or active[0]['id']!=g.VM_ID:raise ValueError('GPU overlap during recovery')
            (R/'file-guard.json').write_text(json.dumps(dict(vm_id=g.VM_ID,disk_id=g.DISK_ID,checked_unix=time.time(),expires_unix=time.time()+45,stop_unix=lease['stop_unix'],model_loading_allowed=False)))
            if (R/'both-guards-ready.json').exists():
                native=json.loads((R/'guard.json').read_text());guest=json.loads((R/'both-guards-ready.json').read_text())
                if native['vm_id']==g.VM_ID and native['disk_id']==g.DISK_ID and native['authorization_sha256']==g.sha(g.AUTH) and time.time()<native['expires_unix'] and native['stop_unix']==lease['stop_unix'] and guest['guest']['ActiveState']=='active':
                    g.complete_backup();handed=True;return
            time.sleep(10)
    finally:
        if not handed:stop_exact(c)


def recover(c,end,watch):
    from .british_controller import R
    inv,rows=r.manifest()
    for n,batch in enumerate(r.batches(rows)):
        if watch.poll() is not None or time.time()>end-180:raise ValueError('Files-first independent guard expired')
        filelist=R/('prior-results-%03d.files'%n);filelist.write_bytes(b''.join(x['path'].encode()+b'\0' for x in batch))
        ssh=[('ControlMaster=no' if x.startswith('ControlMaster=') else 'ControlPath=none' if x.startswith('ControlPath=') else x) for x in c.ssh_argv]
        args=['/opt/homebrew/bin/rsync','-ar','--no-owner','--no-group',*['--exclude='+x for x in sorted(r.EXCLUDED|{'core','core.*'})], '--from0','--files-from='+str(filelist),'-e',shlex.join(ssh[:-1]),ssh[-1]+':'+r.REMOTE+'/',str(r.DEST)+'/']
        c.run(args,min(90,int(end-time.time()-180)))
        for row in batch:
            p=r.checked_path(r.DEST,row['path'])
            if p.is_symlink() or p.stat().st_size!=row['bytes'] or r.sha(p)!=row['sha256']:raise ValueError('Recovered prior artifact hash differs')
        (R/('prior-results-%03d-verified.json'%n)).write_text(json.dumps(dict(files=len(batch),bytes=sum(x['bytes'] for x in batch))))
    for row in inv['files']:
        p=r.checked_path(r.DEST,row['path'])
        if p.is_symlink() or p.stat().st_size!=row['bytes'] or r.sha(p)!=row['sha256']:raise ValueError('Full original inventory mismatch')
    for row in inv['links']:
        p=r.checked_path(r.DEST,row['path'])
        if not p.is_symlink() or os.readlink(p)!=row['target']:raise ValueError('Original symlink changed')
    g.BACKUP.write_text(json.dumps(dict(all_hashes_verified=True,all_link_targets_verified=True,files=1856,bytes=17330768218,links=1,inventory_sha256=r.INVENTORY_SHA,recovered_via='same approved GPU interval, before any native model loading',prior_CPU_attempts_preserved_as_failed=True,vm_id=g.VM_ID,new_optimizer_updates=0),indent=2)+'\n')
    print('ORIGINAL1856_FILES_FULL_HASH_BACKUP_VERIFIED_BEFORE_MODEL',flush=True)
