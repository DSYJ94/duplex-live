"""Snapshot only this control40 run's outputs and verify incremental Mac backups.

No GPU jobs or provider calls. SSH keys stay local; remote snapshots contain only
the approved synthetic/derived audit and training output roots listed below.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import os
import re
import tempfile

ROOTS=('native-control40-preflight-v3/numerical','native-control40-supervisor-v4',
       'native-control40-training-followup-v1','native-control40-training-followup-v2',
       'native-control40-precise-pilot-v1','native-control40-attention-diagnostic-v1',
       'native-control40-precise-pilot-v2','native-control40-native20-v1',
       'native-control40-native20-speech-v1','native-control40-fresh-paired40-v1',
       'native-control40-precise-pilot-v3-adoption','native-control40-precise-pilot-v4-adoption',
       'native-control40-precise-pilot-v5-adoption','native-control40-precise-pilot-v6-adoption',
       'native-control40-precise-pilot-v7-adoption','native-control40-precise-pilot-v8-recovery')
FORBIDDEN={'session-token','.private','.env','keys','credentials','core','quarantine',
           'owner-conversations','owner_conversations'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def marker(base):
    values=[]
    for name in (ROOTS[0]+'/numerical.json',ROOTS[1]+'/gradient/gradient.json',
                 'native-control40-precise-pilot-v1/numerical/numerical.json',
                 'native-control40-precise-pilot-v1/gradient/gradient.json'):
        path=base/name
        if path.exists():
            value=json.loads(path.read_text());values.append([name,len(value['cases']),value.get('passed')])
    for root in ROOTS[2:]:
        for file in ('numerical/numerical.json','gradient/gradient.json'):
            path=base/root/file
            if path.exists():
                value=json.loads(path.read_text());values.append([str(path.relative_to(base)),len(value['cases']),value.get('passed')])
    for name in (root+'/'+file for root in ROOTS[1:] for file in
                 ('controller.json','execution.json','baseline520/execution.json','candidate-control40/execution.json')):
        path=base/name
        if path.exists():
            value=json.loads(path.read_text());values.append([name,value['phase'],len(value.get('cases',[]))])
    for name in (root+'/'+file for root in ROOTS[1:] for file in
                 ('baseline520/report.json','candidate-control40/report.json',
                  'independent-speech40/baseline520/report.json','independent-speech40/candidate-control40/report.json')):
        path=base/name
        if path.exists():
            value=json.loads(path.read_text());values.append([name,len(value['results']),sha(path)])
    for name in (root+'/'+file for root in ROOTS[2:] for file in
                 ('training/training-state.pt','training/report.json','training/reload-report.json',
                  'combined-readiness/control40-combined-readiness.json',
                  'combined-readiness/native-stream49-gradient-readiness.json',
                  'combined-readiness/complete-balanced-preflight.json',
                  'combined-readiness/audio49-bf16-preflight.json',
                  'combined-readiness/parent-adam-import.json')):
        path=base/name
        if path.exists():values.append([name,path.stat().st_size,path.stat().st_mtime_ns])
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


def safe_name(name):
    p=Path(name)
    if (p.is_absolute()or '..'in p.parts or not p.parts or FORBIDDEN.intersection(p.parts)
            or p.suffix in('.tmp','.partial','.lock','.env','.core')or p.name.startswith('core.')
            or not any(p.as_posix().startswith(root+'/')for root in ROOTS)):
        raise ValueError('Unsafe backup inventory path')
    return p


def inventory_rows(root,expected=None):
    file=root/'inventory.json'
    if expected is not None and sha(file)!=expected:raise ValueError('Changed backup inventory')
    value=json.loads(file.read_text());rows=value['files'];seen=set()
    if value['schema']!='dl-native-control40-progress-backup-inventory-v1' or not rows:
        raise ValueError('Nonempty source-pinned evidence inventory required')
    for row in rows:
        name=safe_name(row['path'])
        if (name.as_posix()in seen or type(row['bytes'])is not int or row['bytes']<0
                or not re.fullmatch('[0-9a-f]{64}',row['sha256'])):
            raise ValueError('Invalid or repeated backup inventory member')
        seen.add(name.as_posix())
    return rows


def previous_snapshot(base):
    root=base/'native-control40-backup-snapshots'
    if root.is_symlink():raise ValueError('Backup output root must not be linked elsewhere')
    if root.is_dir():
        for folder in sorted(root.iterdir(),reverse=True):
            if (folder.is_symlink()or not folder.is_dir()or not re.fullmatch(r'\d{8}T\d{6}\.\d{6}Z',folder.name)
                    or not(folder/'inventory.json').is_file()):continue
            try:return folder,inventory_rows(folder)
            except (OSError,ValueError,KeyError,TypeError):continue
    return None,[]


def verified_link(previous,source,target,row):
    """Link a verified closed-snapshot file, never the mutable live output."""
    candidate=previous/safe_name(row['path'])
    if (candidate.is_symlink()or not candidate.is_file()
            or not candidate.resolve().is_relative_to(previous.resolve())
            or candidate.samefile(source)or candidate.stat().st_size!=row['bytes']
            or sha(candidate)!=row['sha256']):return False
    try:os.link(candidate,target)
    except OSError:return False
    return True


def snapshot(base,last_marker=None):
    current=marker(base)
    if current==last_marker:return dict(changed=False,marker=current)
    previous,old_rows=previous_snapshot(base);old={r['path']:r for r in old_rows}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out=base/'native-control40-backup-snapshots'/stamp;out.mkdir(parents=True,exist_ok=False)
    files=[];reused=0;reused_bytes=0
    for name in ROOTS:
        folder=base/name
        if folder.is_symlink() or not folder.resolve().is_relative_to(base.resolve()):
            raise ValueError('Approved output root must stay inside migration storage')
        if not folder.is_dir():continue
        for source in sorted(folder.rglob('*')):
            relative=source.relative_to(base)
            if (source.is_symlink() or not source.is_file() or source.suffix in ('.tmp','.partial','.lock','.env')
                    or FORBIDDEN.intersection(relative.parts) or source.name.endswith('.core')
                    or source.name.startswith('core.')):continue
            if not source.resolve().is_relative_to(folder.resolve()):raise ValueError('Output escaped approved root')
            # Mutable logs/JSON are explicitly a progress snapshot. An atomic
            # training-state rename leaves the opened old/new inode intact.
            target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
            prior=old.get(relative.as_posix());linked=False
            if prior is not None and source.stat().st_size==prior['bytes'] and sha(source)==prior['sha256']:
                linked=verified_link(previous,source,target,prior)
            if not linked:
                with source.open('rb')as a,target.open('xb')as b:shutil.copyfileobj(a,b)
            fingerprint=sha(target)
            if linked and fingerprint!=prior['sha256']:
                target.unlink() # Never write through an old snapshot's hard link.
                with source.open('rb')as a,target.open('xb')as b:shutil.copyfileobj(a,b)
                fingerprint=sha(target);linked=False
            if linked:reused+=1;reused_bytes+=target.stat().st_size
            files.append(dict(path=str(relative),bytes=target.stat().st_size,sha256=fingerprint))
    manifest=dict(schema='dl-native-control40-progress-backup-inventory-v1',at_utc=stamp,
                  marker=current,files=files,source_roots=list(ROOTS),
                  live_progress_snapshot=True,not_a_training_or_behavior_approval=True,
                  reused_closed_snapshot_files=reused,reused_closed_snapshot_bytes=reused_bytes,
                  previous_inventory_sha256=sha(previous/'inventory.json')if previous is not None else None,
                  symlinks_credentials_owner_conversations_and_cores_excluded=True)
    inventory=out/'inventory.json';inventory.write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(changed=True,marker=current,snapshot=str(out),inventory_sha256=sha(inventory),files=len(files),reused_files=reused,reused_bytes=reused_bytes)


def verify(root,expected):
    inventory=root/'inventory.json'
    if sha(inventory)!=expected:raise ValueError('Changed backup inventory')
    value=json.loads(inventory.read_text())
    seen=set()
    for row in value['files']:
        name=Path(row['path']);p=root/name
        if (name.is_absolute() or '..' in name.parts or str(name) in seen or p.is_symlink()
                or FORBIDDEN.intersection(name.parts) or p.suffix in ('.env','.core') or p.name.startswith('core.')
                or not p.resolve().is_relative_to(root.resolve())
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):
            raise ValueError('Missing, changed or unsafe local evidence copy')
        seen.add(str(name))
    if not seen:raise ValueError('Empty evidence backup')
    return len(seen)


def reuse_local_files(reference,target,rows):
    """Prelink only matching verified snapshot contents; remaining paths transfer."""
    old={}
    if reference is not None:
        for row in inventory_rows(reference):old[(row['bytes'],row['sha256'])]=row
    missing=[];reused=[]
    for row in rows:
        name=safe_name(row['path']);dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True)
        previous=old.get((row['bytes'],row['sha256']))
        if previous is not None:
            source=reference/safe_name(previous['path'])
            if (not source.is_symlink()and source.is_file()and source.resolve().is_relative_to(reference.resolve())
                    and source.stat().st_size==row['bytes']and sha(source)==row['sha256']):
                try:os.link(source,dest);reused.append(name.as_posix());continue
                except OSError:pass
        missing.append(name.as_posix())
    return missing,reused


def receive_snapshot(info,target,ssh,host,reference=None):
    remote=Path(info['snapshot'])
    if remote.parent!=Path('/workspace/migration/native-control40-backup-snapshots'):
        raise ValueError('Unexpected remote snapshot root')
    target.mkdir(parents=True,exist_ok=False)
    prefix=['rsync','-ar','--exclude=session-token','--no-owner','--no-group','-e',shlex.join(ssh)]
    subprocess.run(prefix+[host+':'+str(remote/'inventory.json'),str(target)+'/'],
                   check=True,capture_output=True,text=True,timeout=60)
    rows=inventory_rows(target,info['inventory_sha256'])
    missing,reused=reuse_local_files(reference,target,rows)
    if missing:
        with tempfile.NamedTemporaryFile(prefix='duplex-approved-copy-',mode='wb')as names:
            names.write(('\0'.join(missing)+'\0').encode());names.flush()
            subprocess.run(prefix+['--from0','--files-from',names.name,host+':'+str(remote)+'/',str(target)+'/'],
                           check=True,capture_output=True,text=True,timeout=90)
    count=verify(target,info['inventory_sha256'])
    reused_set=set(reused);saved=sum(r['bytes']for r in rows if r['path']in reused_set)
    return count,dict(reused_verified_files=len(reused),reused_verified_bytes=saved,
                      transferred_file_bytes=sum(r['bytes']for r in rows if r['path']not in reused_set))


def ssh_command(key,known_hosts,host,port):
    if (not re.fullmatch(r'root@(?:[A-Za-z0-9][A-Za-z0-9.-]*|\[[0-9a-fA-F:]+\])',host)
            or type(port) is not int or not 1<=port<=65535):
        raise ValueError('Explicit actual root SSH host and TCP port required')
    filename=str(known_hosts)
    if any(c in filename for c in ('\n','\r','\0')):raise ValueError('Single known-hosts filename required')
    # OpenSSH parses -o through its config grammar even for a single argv item.
    filename='"'+filename.replace('\\','\\\\').replace('"','\\"')+'"'
    return ['ssh','-i',str(key),'-p',str(port),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
        '-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+filename]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote-snapshot',action='store_true');p.add_argument('--last-marker')
    p.add_argument('--source-commit',required=True)
    for n in ('key','known-hosts','remote-source','out'):p.add_argument('--'+n,type=Path)
    p.add_argument('--deadline-utc');p.add_argument('--host');p.add_argument('--port',type=int)
    p.add_argument('--pod-id');p.add_argument('--reference-snapshot',type=Path);a=p.parse_args()
    source=Path(__file__).resolve().parents[2]
    if a.remote_snapshot:
        if (subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=a.source_commit
                or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()):
            raise ValueError('Exact clean committed backup source required')
        print(json.dumps(snapshot(Path('/workspace/migration'),a.last_marker)));return
    if not all((a.key,a.known_hosts,a.remote_source,a.out,a.deadline_utc,a.host,a.port,a.pod_id)):
        p.error('Local copyback requires actual Pod/SSH host/port, exact SSH paths, output and finite deadline')
    deadline=datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00')).timestamp()
    if not time.time()+60<deadline<time.time()+12*3600:raise ValueError('Bounded future backup interval required')
    a.out.mkdir(parents=True,exist_ok=False)
    ssh=ssh_command(a.key,a.known_hosts,a.host,a.port)
    host=a.host;last=None;reference=a.reference_snapshot
    status=dict(schema='dl-native-control40-copyback-watch-v1',pid=os.getpid(),source_commit=a.source_commit,
                deadline_utc=a.deadline_utc,phase='starting',snapshots=[],changes_to_remote_jobs=False,
                pod_id=a.pod_id,ssh_host=a.host,ssh_port=a.port)
    def save():
        status['at_utc']=datetime.now(timezone.utc).isoformat()
        temp=a.out/'controller.partial';temp.write_text(json.dumps(status,indent=2)+'\n');temp.replace(a.out/'controller.json')
    save()
    while time.time()<deadline:
        try:
            command=['timeout','--signal=TERM','110s','env','PYTHONDONTWRITEBYTECODE=1','python3',str(a.remote_source/'portable/context/native_control_copyback.py'),
                     '--remote-snapshot','--source-commit',a.source_commit]
            if last:command+=['--last-marker',last]
            result=subprocess.run(ssh+[host,shlex.join(command)],check=True,capture_output=True,text=True,timeout=120)
            info=json.loads(result.stdout)
            if info['changed']:
                remote=Path(info['snapshot'])
                if remote.parent!=Path('/workspace/migration/native-control40-backup-snapshots'):
                    raise ValueError('Unexpected remote snapshot root')
                target=a.out/remote.name
                count,reuse=receive_snapshot(info,target,ssh,host,reference)
                status['snapshots'].append(dict(path=str(target),files=count,inventory_sha256=info['inventory_sha256'],**reuse))
                reference=target;last=info['marker']
            status.update(phase='watching',last_marker=last,last_check_succeeded=True);save()
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            status.update(phase='retrying',last_check_succeeded=False,error_type=type(error).__name__)
            save()
        time.sleep(min(60,max(0,deadline-time.time())))
    status['phase']='deadline_reached';save()


if __name__=='__main__':main()
