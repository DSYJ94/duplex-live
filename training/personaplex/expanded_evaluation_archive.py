"""Hash-complete static evaluation archive; no scratch/guard/model states."""
import json,os,subprocess,time
from pathlib import Path
from .evaluation_result_archive import inventory
from .matched_archive import independent_filesystems,sha


def static_inventory(root):
    root=Path(root);rows=[]
    for name in ('control32','dev64','dev-inputs'):
        for row in inventory(root/name):rows.append(dict(row,path=name+'/'+row['path']))
    for path in sorted(root.glob('*.json')):
        rows.append(dict(path=path.name,kind='file',bytes=path.stat().st_size,sha256=sha(path)))
    return sorted(rows,key=lambda row:row['path'])


def archive(root,destination,source,*,resume_stage=False):
    root=Path(root);destination=Path(destination);comparison=json.loads((root/'comparison.json').read_text())
    if not comparison['complete192']or comparison['source']!=source or comparison['optimizer_updates']!=0:raise ValueError('Full successful zero-update comparison required before backup')
    if destination.exists():raise ValueError('Never overwrite an immutable result archive')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not independent_filesystems(root,destination.parent):raise ValueError('Independent result filesystem required')
    rows=static_inventory(root)
    for row in rows:
        if row['kind']=='file':continue
        name=Path(row['path']);target=Path(row.get('target',''))
        if row['kind']!='symlink'or name.parent.name!='driver'or name.name not in ('libcuda.so','libcuda.so.1')or not target.is_absolute()or not target.name.startswith('libcuda.so'):
            raise ValueError('Only explicit runtime CUDA linker symlinks may be archived without following them')
    stage=destination.with_name('.'+destination.name+'-partial')
    if resume_stage:
        if not stage.is_dir()or stage.is_symlink():raise ValueError('Explicit existing partial archive required')
    else:stage.mkdir(exist_ok=False)
    for offset in range(0,len(rows),2000):
        names=b''.join(r['path'].encode()+b'\0'for r in rows[offset:offset+2000])
        subprocess.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--timeout=180','--from0','--files-from=-',str(root)+'/',str(stage)+'/'],input=names,check=True)
        print(json.dumps(dict(event='evaluation_archive_batch',through=min(offset+2000,len(rows)),total=len(rows))),flush=True)
    copied=sorted(inventory(stage),key=lambda r:r['path'])
    if rows!=copied:raise ValueError('Full independent result inventory/hash comparison failed')
    (stage/'archive-inventory.json').write_text(json.dumps(dict(source=str(root),producer=source,files=rows),indent=2)+'\n')
    receipt=dict(passed=True,source=str(root),destination=str(destination),producer=source,files=len(rows),bytes=sum(r['bytes']for r in rows),manifest_sha256=sha(stage/'archive-inventory.json'),independent_filesystem=True,every_file_sha256_verified=True,optimizer_updates=0,actual_unix=time.time())
    (stage/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');os.rename(stage,destination);return receipt
