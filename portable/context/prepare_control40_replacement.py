"""Restore only verified runtime inputs on a fresh budget-guarded replacement.

The retained archive is read-only input. Models, original runs and the complete
control40 workset are copied to fresh local paths and independently rehashed.
Canonical aliases are activated only after verification. This never trains or
approves numerical/model quality. --metadata-only performs no staging or GPU work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from evals.moshi_model_download import sha256
from evals.native_retained_prepared import INVENTORY_SHA,VOCAB_SHA
from evals.native_control_snapshot import INVENTORY_SHA256 as WORKSET_SHA
from portable.context.native_development import atomic,utc
from portable.context.native_fresh_development import budget_guard,OWNER_LOCK
from portable.context.prepare_native_layout import checkout,link

ARCHIVE_SHA='4eac15590dc9def6dfb5e34b2cea92f30d9a067fabe0940e2876b6acdcbe6dce'
BASE='models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'
LEGACY='models/hf-cache/hub/models--kyutai--moshika-rag-pytorch-bf16/snapshots/7135a6e3c46abb66c2cd95cb04cbfcbe8376f83d/tokenizer_spm_32k_3.model'
VENDORS=(('context-spanning','79df302fa2820cae10c61bb2b7f302c52edb6ed2',
          'c992dc548838cb182e54e94101b6d4f8d7a0c6cb0d334b1fe37eeb381baa4264'),
         ('moshi-rag','8c6dfc101b7871baa428424bcdc583b74fb561d9',
          'de6faf6be1b8fb6ae926ca71b0cec5cd6676c077139a9dfa79c27e4c312f75d8'),
         ('moshi-finetune','2acc879fe7c48f885a18f6cc9548bccb2674d87b',
          '1859eb16d8f7f6e1d01815e8fb6da8b889cf391462caad7cb552af3e662695d8'))
CANONICAL=(Path('/projects/u6si/duplex-live'),Path('/lus/lfs1aip2/projects/u6si/duplex-live'))
FORBIDDEN={'session-token','.private','.env','keys','credentials','quarantine','core',
           'owner-conversations','owner_conversations'}


def safe_name(name):
    p=Path(name)
    if (p.is_absolute() or '..' in p.parts or not p.parts or FORBIDDEN.intersection(p.parts)
            or p.suffix in ('.env','.core') or p.name.startswith('core.')):
        raise ValueError('Unsafe or excluded runtime input: '+str(name))
    return p


def copy_verified(source,target,size,digest,source_root,bounded=lambda:None):
    source=Path(source);target=Path(target)
    if (not source.resolve().is_relative_to(Path(source_root).resolve())
            or not source.is_file() or source.stat().st_size!=size):
        raise ValueError('Runtime input is missing or outside its approved source')
    if target.exists() or target.is_symlink():raise ValueError('Refuse existing staging file: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_name(target.name+'.partial');h=hashlib.sha256();count=0
    with source.open('rb') as reader,temp.open('xb') as writer:
        for block in iter(lambda:reader.read(8*1024*1024),b''):
            bounded()
            count+=len(block);h.update(block);writer.write(block)
    if count!=size or h.hexdigest()!=digest:raise ValueError('Source copy hash/size differs: '+str(source))
    if temp.stat().st_size!=size or sha256(temp)!=digest:
        raise ValueError('Independent local read-back differs: '+str(target))
    os.link(temp,target);temp.unlink()  # Atomic no-overwrite publication.


def copy_rows(rows,source_root,target_root,bounded=lambda:None):
    if target_root.is_symlink():raise ValueError('Staging root must not be an alias')
    seen=set()
    for row in rows:
        name=safe_name(row['path']);target=target_root/name
        if str(name) in seen or target.exists() or target.is_symlink():
            raise ValueError('Duplicate or existing runtime input')
        if not target.parent.resolve().is_relative_to(target_root.resolve()):
            raise ValueError('Staging parent escaped through an existing alias')
        if any(p.is_symlink() for p in target.parents if p!=target_root and p.is_relative_to(target_root)):
            raise ValueError('Staging input parent must not be an existing alias')
        source=source_root/name
        if source.is_symlink():raise ValueError('Original/workset files must remain regular files')
        seen.add(str(name))
    for row in rows:
        bounded()
        name=safe_name(row['path'])
        copy_verified(source_root/name,target_root/name,row['bytes'],row['sha256'],source_root,bounded)
    return dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),all_independent_readback_hashes_match=True)


def metadata(archive,original_inventory,workset_inventory):
    from evals.personaplex_train import MODEL_HASHES
    from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256,VOICE_SHA256
    if sha256(original_inventory)!=INVENTORY_SHA or sha256(workset_inventory)!=WORKSET_SHA:
        raise ValueError('Exact original10390 and complete7797 inventories required')
    original=json.loads(original_inventory.read_text());workset=json.loads(workset_inventory.read_text())
    if (original['original_archive_inventory_sha256']!=ARCHIVE_SHA or original['file_count']!=10390
            or len(original['files'])!=10390 or original['bytes']!=24661548106
            or workset['file_count']!=7797 or len(workset['files'])!=7797
            or workset['uncompressed_bytes']!=8000842614):
        raise ValueError('Complete unchanged original/control input counts required')
    for value in (original,workset):
        names=[str(safe_name(row['path'])) for row in value['files']]
        if len(set(names))!=len(names) or any(Path(n).parts[0]!='runs' for n in names):
            raise ValueError('Distinct complete original/control run paths required')
    manifest=archive/'runs/context-spanning-assets-7454205/manifest.json'
    expected=next(r for r in original['files'] if r['path']==str(manifest.relative_to(archive)))
    if sha256(manifest)!=expected['sha256']:raise ValueError('Original Context Spanning manifest changed')
    assets=json.loads(manifest.read_text());weight_root='models/hf-cache/hub/models--mindlogicinc--context-spanning-7b/snapshots/'+MODEL_REVISION
    if (assets['source_commit']!=SOURCE_COMMIT or assets['revision']!=MODEL_REVISION
            or assets['files']['context_spanning_7b.pt']['sha256']!=WEIGHT_SHA256
            or assets['files']['voices/main.pt']['sha256']!=VOICE_SHA256):
        raise ValueError('Pinned native model selection or released voice changed')
    models=[]
    for name,digest in MODEL_HASHES.items():
        p=archive/BASE/name
        models.append(dict(path=str(Path(BASE)/name),bytes=p.stat().st_size,sha256=digest))
    for name,row in assets['files'].items():
        models.append(dict(path=str(Path(weight_root)/safe_name(name)),bytes=row['bytes'],sha256=row['sha256']))
    bundles=[]
    for name,commit,digest in VENDORS:
        p=archive/'runs/local-preservation-vendor-sources-20261019'/(name+'-'+commit+'.bundle')
        if sha256(p)!=digest:raise ValueError('Pinned vendor bundle differs: '+name)
        bundles.append(dict(name=name,commit=commit,bundle=str(p),sha256=digest))
    return original,workset,models,bundles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','original-inventory','workset-inventory','source','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('stage','workset','cost-guard-receipt'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--pod-id')
    parser.add_argument('--deadline-utc');parser.add_argument('--metadata-only',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Fresh versioned preparation receipt required')
    original,workset,models,bundles=metadata(args.archive,args.original_inventory,args.workset_inventory)
    state=dict(schema='dl-native-control40-replacement-preparation-v1',source_commit=args.source_commit,
        recorded_utc=utc(),phase='metadata_verified_only',original_inventory_sha256=INVENTORY_SHA,
        workset_inventory_sha256=WORKSET_SHA,original_files=10390,workset_files=7797,model_files=models,
        vendor_bundles=bundles,archive_mutated=False,optimizer_updates=0,GPU_work_executed=False,
        actual_staging_completed=False,training_or_behavior_approved=False,
        original_archive_not_fully_rehashed_by_this_command=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.metadata_only:
        atomic(args.out,state);print(json.dumps(state));return
    if not all((args.stage,args.workset,args.cost_guard_receipt,args.pod_id,args.deadline_utc)):
        parser.error('Actual staging requires explicit fresh stage/workset, same-Pod cost guard and bounded deadline')
    from datetime import datetime
    deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00')).timestamp()
    if deadline<=time.time()+120:raise ValueError('Bounded future staging deadline required')
    budget_guard(args.cost_guard_receipt,args.pod_id,deadline)
    if (args.stage.exists() or args.stage.is_symlink() or not args.stage.is_absolute()
            or not args.stage.resolve().is_relative_to(Path('/opt'))
            or not args.source.resolve().is_relative_to(Path('/opt'))
            or args.source.resolve().is_relative_to(args.archive.resolve())
            or args.stage.resolve().is_relative_to(args.archive.resolve())):
        raise ValueError('Fresh ephemeral staging root outside preserved archive required')
    if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=args.source_commit:
        raise ValueError('Exact deployment commit required')
    if subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean immutable execution source required')
    for path in CANONICAL:
        if path.exists() or path.is_symlink():raise ValueError('Refuse existing canonical native alias')
    import fcntl
    OWNER_LOCK.parent.mkdir(parents=True,exist_ok=True)
    lock=OWNER_LOCK.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    if any(v.strip().isdigit() for v in occupied.splitlines()):raise ValueError('Existing GPU work forbids runtime alias changes')
    required=original['bytes']+sum(r['bytes'] for r in models)+workset['uncompressed_bytes']+5_000_000_000
    if shutil.disk_usage(args.stage.parent).free<required:raise ValueError('Insufficient local staging disk')
    args.stage.mkdir();state.update(phase='copying_verified_inputs',pod_id=args.pod_id,stage=str(args.stage))
    atomic(args.out,state)
    def bounded():
        if time.time()>=deadline:raise TimeoutError('Replacement staging deadline reached')
    def expired(signum,frame):raise TimeoutError('Replacement staging wall-clock deadline reached')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    try:
        bounded();state['original_runs']=copy_rows(original['files'],args.archive,args.stage,bounded)
        bounded();state['workset']=copy_rows(workset['files'],args.workset,args.source,bounded)
        for row in models:
            bounded();name=safe_name(row['path'])
            copy_verified(args.archive/name,args.stage/name,row['bytes'],row['sha256'],args.archive/'models',bounded)
        for row in bundles:
            bounded();name=row['name'];commit=row['commit']
            target=(args.stage/'vendor'/(name+'-'+commit) if name=='context-spanning'
                    else Path('/workspace')/('control40-vendor-'+name))
            checkout(Path(row['bundle']),target,commit)
            if name!='context-spanning':link(target,args.stage/'vendor'/(name+'-'+commit))
        for name in ('tmp','logs'):link(args.archive/name,args.stage/name)
        worktrees=Path('/workspace/duplex-live/worktrees')
        if not worktrees.is_dir():raise ValueError('Retained historical Git checkouts must be restored before runtime activation')
        link(worktrees,args.stage/'worktrees')
        vocab=args.stage/BASE/'tokenizer_spm_32k_3.model'
        if sha256(vocab)!=VOCAB_SHA:raise ValueError('Exact original legacy vocabulary required')
        link(vocab,args.stage/LEGACY)
        bounded()
        for path in CANONICAL:link(args.stage,path)
        state.update(phase='complete_staging_only',actual_staging_completed=True,
            all_active_model_hashes_independently_rehashed=True,canonical_aliases=list(map(str,CANONICAL)),
            native_python='/workspace/runtime-x86-ngc2510/bin/python',
            required_pythonpath='/workspace/control40-vendor-moshi-rag/moshi',
            fresh_compiler_GPU_probe_and_full_audits_still_required=True)
        atomic(args.out,state)
    except BaseException as error:
        state.update(phase='failed',error_type=type(error).__name__,error=str(error));atomic(args.out,state);raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':main()
