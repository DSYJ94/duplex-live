"""Copy new immutable epoch/window states to independent Lustre; no model work.

Never changes training, submits jobs, touches OneDrive or accepts a failed hash.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from .matched_interrupted import DURABLE
from .matched_paths import BASE
from .matched_job import write


def start_helper(root,job,source):
    import sys
    execution=Path(root)/('execution-'+job)
    with (execution/'independent-archive.log').open('x') as log:
        child=subprocess.Popen([sys.executable,'-B','-u','-m','training.personaplex.matched_archive',
            '--root',str(root),'--job',job,'--source',source],stdout=log,stderr=subprocess.STDOUT)
    write(execution/'independent-archive-process.json',dict(source=source,job_id=job,pid=child.pid,
        training_does_not_depend_on_archive_process=True,Mac_connection_required=False))
    return child


def finish_helper(child,root,job,source):
    if child is None:return
    timed_out=False
    try:child.wait(timeout=330)
    except subprocess.TimeoutExpired:
        timed_out=True;child.terminate()
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait()
    write(Path(root)/('execution-'+job)/'independent-archive-terminal.json',dict(source=source,job_id=job,
        actual_exit_code=child.returncode,timed_out=timed_out,training_interrupted=False))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def independent_filesystems(left,right):
    return Path(left).stat().st_dev!=Path(right).stat().st_dev


def archive(checkpoint,receipt,destination,source):
    checkpoint=Path(checkpoint);receipt=Path(receipt);destination=Path(destination)
    reference=Path(str(checkpoint)+'.native-reference.json')
    value=json.loads(receipt.read_text());native=json.loads(reference.read_text())
    digest=value['checkpoint_sha256']
    step=value.get('lineage_updates',value.get('total_lineage_updates'))
    if (checkpoint.is_symlink() or value['source']!=source or native['source']!=source
            or native['step']!=step or native['checkpoint_sha256']!=digest or sha(checkpoint)!=digest):
        raise ValueError('Exact complete source/reference/checkpoint required before backup')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not independent_filesystems(destination.parent,checkpoint):
        raise ValueError('Backup must be on a separate filesystem')
    if destination.exists():
        old=json.loads((destination/'archive-receipt.json').read_text())
        if (old['sha256']!=digest or old.get('source')!=source or old.get('step')!=step
                or sha(destination/checkpoint.name)!=digest
                or (destination/reference.name).read_bytes()!=reference.read_bytes()
                or (destination/receipt.name).read_bytes()!=receipt.read_bytes()):
            raise ValueError('Existing immutable backup differs; never overwrite it')
        return old,False
    stage=destination.with_name('.'+destination.name+'-stage-'+str(os.getpid()));stage.mkdir(exist_ok=False)
    result=subprocess.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--timeout=180',
        str(checkpoint),str(reference),str(receipt),str(stage)+'/'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        universal_newlines=True,timeout=300)
    if result.returncode:raise RuntimeError('Actual archive rsync failed: '+str(result.returncode)+' '+result.stderr[-2000:])
    if sha(stage/checkpoint.name)!=digest or (stage/reference.name).read_bytes()!=reference.read_bytes() or (stage/receipt.name).read_bytes()!=receipt.read_bytes():
        raise ValueError('Destination full hash/reference/receipt mismatch')
    record=dict(source=source,checkpoint=str(checkpoint),destination=str(destination/checkpoint.name),step=step,
        bytes=checkpoint.stat().st_size,sha256=digest,actual_rsync_exit_code=0,hash_verified=True,
        separate_filesystem=True,actual_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),native_restore_still_required=True,
        backup_program_source=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],universal_newlines=True).strip())
    record['files']={p.name:dict(bytes=(stage/p.name).stat().st_size,sha256=sha(stage/p.name)) for p in (checkpoint,reference,receipt)}
    write(stage/'archive-receipt.json',record);os.rename(str(stage),str(destination));return record,True


def watch(root,job,source):
    root=Path(root);root.resolve().relative_to((DURABLE/'runs').resolve())
    lease=json.loads((root/('execution-'+job)/'lease.json').read_text())
    if lease['job_id']!=job or lease['source']!=source:raise ValueError('Exact actual training run required')
    destination=BASE/'native-moshi-postupdate-backup-v1'/('job-'+job);records={}
    while True:
        receipts=list(root.glob('*/*/train/epoch-[0-9][0-9].json'))+list(root.glob('*/*/train/training-window-terminal.json'))
        for receipt in sorted(receipts):
            relative=receipt.relative_to(root);base,capacity=relative.parts[:2]
            if base not in ('personaplex','context-spanning') or capacity not in ('narrow','broad'):raise ValueError('Unknown checkpoint lineage')
            key=str(relative)
            if key in records:continue
            value=json.loads(receipt.read_text())
            checkpoint=receipt.with_suffix('.pt') if receipt.name.startswith('epoch-') else Path(value['checkpoint'])
            if checkpoint.parent!=receipt.parent:raise ValueError('Checkpoint outside exact training lineage')
            record,created=archive(checkpoint,receipt,destination/base/capacity/receipt.stem,source)
            records[key]=record
            write(root/'independent-archive-status.json',dict(job_id=job,source=source,copies=list(records.values()),passed=True,training_interrupted=False,OneDrive_writes=False))
            if created:print(json.dumps(dict(event='checkpoint_archived',job_id=job,**record)),flush=True)
        terminal=root/('execution-'+job)/'terminal.json'
        if terminal.exists():
            print(json.dumps(dict(event='archive_watch_finished',job_id=job,source=source,immutable_states=len(records),host_terminal=json.loads(terminal.read_text()),training_interrupted=False)),flush=True);return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','job','source'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:watch(a.root,a.job,a.source)
    except BaseException as exc:
        write(Path(a.root)/'independent-archive-failure.json',dict(job_id=a.job,source=a.source,
            error=type(exc).__name__+': '+str(exc),passed=False,training_interrupted=False,
            actual_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        raise
