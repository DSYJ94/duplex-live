"""Restore the original pinned control40 workset, never regenerate approvals.

CPU stdlib only. Every original archive member and a second destination read
must match; source-pinned old validators still independently recompute all40.
"""
import argparse,hashlib,json,os,shutil,subprocess,tarfile,tempfile,time
from pathlib import Path

INVENTORY_SHA='79401e2c1051361d150da644250e8acf87814bda1c48b167af8a34345214fa8f'
ARCHIVE_SHA='87717cdf3e80977dbe78b18b0f1e75367411eab535d327dbee4679dcd9483616'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def destination(root,name):
    relative=Path(name)
    if (relative.is_absolute() or '..' in relative.parts or len(relative.parts)<3
            or relative.parts[0]!='runs' or not relative.parts[1].startswith('native-control3-')
            or any(p in ('session-token','.private','owner-conversations') for p in relative.parts)):
        raise ValueError('Only original synthetic native-control3 artifact paths allowed')
    path=root/relative;path.resolve().relative_to(root.resolve())
    return path


def extract(archive,rows,root,*,resume=False):
    expected={r['path']:r for r in rows};seen=set()
    if len(expected)!=len(rows):raise ValueError('Duplicate original inventory paths')
    for name in expected:destination(root,name)
    if root.is_symlink():raise ValueError('No symlink restoration root')
    root.mkdir(parents=True,exist_ok=resume);reused=repaired=0
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            name=member.name;path=destination(root,name)
            if not member.isfile() or name not in expected or name in seen:
                raise ValueError('Unexpected archive member, link or duplicate')
            row=expected[name]
            if member.size!=row['bytes']:raise ValueError('Original archive file size differs')
            path.parent.mkdir(parents=True,exist_ok=True);value=hashlib.sha256()
            if path.is_symlink():raise ValueError('No symlink in partial restoration')
            good=path.is_file() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            # Missing/partial copies are assembled separately, then published
            # atomically. Existing valid bytes are never rewritten. Failed
            # partial destination bytes are preserved outside the source tree.
            temporary=None;target=None
            if not good:
                fd,temporary=tempfile.mkstemp(prefix='.control40-restore-',dir=str(root.parent))
                target=os.fdopen(fd,'wb')
            try:
                with stream.extractfile(member) as source:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        value.update(block)
                        if target is not None:target.write(block)
                if target is not None:target.close()
                if value.hexdigest()!=row['sha256']:raise ValueError('Original archive member hash differs')
                if good:reused+=1
                else:
                    if path.exists():
                        saved=root.parent/(root.name+'-partial-preserved-v1')/name
                        saved.parent.mkdir(parents=True,exist_ok=True)
                        if saved.exists():raise ValueError('Preserve prior partial-copy evidence')
                        shutil.copyfile(path,saved);repaired+=1
                    os.replace(temporary,str(path));temporary=None
            finally:
                if target is not None and not target.closed:target.close()
                if temporary is not None:Path(temporary).unlink()
            seen.add(name)
    if seen!=set(expected):raise ValueError('Complete original workset required, not a subset')
    for name,row in expected.items():
        path=destination(root,name)
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Independent destination hash differs')
    return dict(files=len(seen),bytes=sum(r['bytes'] for r in rows),all_archive_member_hashes_exact=True,
        all_destination_hashes_independently_exact=True,existing_valid_files_reused=reused,
        partial_files_preserved_and_repaired=repaired)


def main(a):
    began=time.time()
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=a.source:
        raise ValueError('Exact Git artifact-restoration producer required')
    spec=json.loads(a.inventory.read_text())
    if (sha(a.inventory)!=INVENTORY_SHA or sha(a.archive)!=ARCHIVE_SHA
            or spec['file_count']!=7797 or len(spec['files'])!=7797
            or spec['uncompressed_bytes']!=8000842614 or a.archive.stat().st_size!=1023720756
            or spec['owner_conversations_credentials_core_dumps_included'] is not False):
        raise ValueError('Original complete approved control40 snapshot required')
    report=extract(a.archive,spec['files'],a.out,resume=a.resume)
    shutil.copyfile(a.inventory,a.out/'inventory.json')
    if sha(a.out/'inventory.json')!=INVENTORY_SHA:raise ValueError('Original inventory bytes changed')
    report.update(source=a.source,passed=True,inventory_sha256=INVENTORY_SHA,archive_sha256=ARCHIVE_SHA,
        seconds=time.time()-began,original_audio_words_receipts_folds_unchanged=True,
        old_complete40_recomputation_still_required=True,GPU_allocated=False,optimizer_updates=0,
        native_training_or_behavior_approved=False)
    (a.out/'restore.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archive','inventory','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--source',required=True);main(p.parse_args())
