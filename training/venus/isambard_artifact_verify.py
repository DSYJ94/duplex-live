"""Verify one sealed artifact inventory in Slurm; never execute a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def verify(root, manifest, digest):
    if sha(manifest)!=digest:raise ValueError('Sealed inventory changed')
    inv=json.loads(manifest.read_text());seen=set();total=0
    for row in inv['files']+inv['links']:
        rel=PurePosixPath(row['path'])
        if rel.is_absolute() or '..' in rel.parts or row['path'] in seen:raise ValueError('Unsafe or duplicate artifact path')
        seen.add(row['path']);p=root.joinpath(*rel.parts)
        try:p.parent.resolve().relative_to(root.resolve())
        except ValueError:raise ValueError('Artifact parent escapes root')
        if 'target' in row:
            if not p.is_symlink() or os.readlink(p)!=row['target']:raise ValueError('Original artifact link differs')
        else:
            if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Artifact bytes differ: '+row['path'])
            total+=row['bytes']
    if total!=inv['bytes']:raise ValueError('Artifact byte inventory differs')
    return dict(all_hashes_verified=True,files=len(inv['files']),links=len(inv['links']),bytes=total,manifest_sha256=digest,model_execution=False,new_optimizer_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','manifest','sha256','source','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_ACCOUNT')!='brics.u6ada':raise ValueError('Run artifact hashing through the project scheduler')
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:raise ValueError('Exact deployed Git source required')
    result=verify(Path(a.root),Path(a.manifest),a.sha256)
    result.update(job_id=os.environ['SLURM_JOB_ID'],source=a.source,completed_utc=datetime.now(timezone.utc).isoformat())
    with Path(a.output).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
