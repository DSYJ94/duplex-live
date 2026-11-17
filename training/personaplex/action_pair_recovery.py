"""Exact failed-parent identity and immutable raw-output reuse for audit recovery."""
import json
import os
from pathlib import Path
from .expanded_guard import sha
from .evaluation_result_archive import inventory

BASE=Path('/projects/u6ada/duplex-live')
OLD=BASE/'native-ret-paired-20261117-v1'
BACKUP=Path('/home/u6ada/dsyj94.u6ada/duplex-native-ret-paired-backups-v1/job-8540802/readout/generated-before-audit')
ORIGIN='c66b72ba02211496285e061178d4e9c3669267f3'


def completed_parent(scope):
    fit=json.loads((OLD/'readout/fit/terminal.json').read_text())
    backed=json.loads((OLD/'job-8540802/readout-fit-backup.json').read_text())
    raw=json.loads((BACKUP/'archive-receipt.json').read_text())
    rows=json.loads((BACKUP/'archive-inventory.json').read_text())
    if (fit['actual_new_readout_updates']!=128 or fit['backbone_updates']!=0
            or fit['provenance']['source']!=ORIGIN or sha(fit['checkpoint'])!=scope['readout_checkpoint_sha256']
            or not backed['passed'] or not backed['every_file_sha256_verified']
            or sha(Path(backed['destination'])/'archive-inventory.json')!=backed['inventory_sha256']):
        raise ValueError('Complete immutable readout128 checkpoint and independent backup required')
    if (not raw['passed'] or not raw['every_file_sha256_verified']
            or sha(BACKUP/'archive-inventory.json')!=raw['inventory_sha256']):
        raise ValueError('Finish original independent raw-output backup before recovery')
    for cohort,count in (('control32',32),('dev64',64)):
        report=json.loads((OLD/'readout/evaluate'/cohort/'personaplex-expanded105M/results.json').read_text())
        if report['source']!=ORIGIN or report['required_cases']!=count or len(report['cases'])!=count or any(not r['complete']for r in report['cases']):
            raise ValueError('All96 completed generation outputs required, no partial reuse')
    restored=json.loads((OLD/'readout/evaluate/native-restore.json').read_text())
    if (not restored['actual_CUDA']or not restored['Adam_RNG_OPAL_exact']
            or restored['new_optimizer_updates']!=0 or restored['saved_readout_updates']!=128
            or restored['original_fit_sha256']!=scope['readout_checkpoint_sha256']
            or restored['provenance']['source']!=ORIGIN):
        raise ValueError('Preserve original actual native readout restore proof')
    if (OLD/'bias/fit/state').exists():
        raise ValueError('Unexpected bias state; never replay saved updates')
    return dict(fit=fit,fit_backup=backed,raw_backup=raw,native_restore=restored),rows


def clone_raw(source,target,rows):
    """Hardlink raw bytes on Lustre; later independent archive copies the bytes.

    Audit creates only NEW derived/report files. Reject an already audited or
    partially audited tree instead of overwriting its output or recognition.
    """
    source,target=Path(source),Path(target)
    selected=[r for r in rows if r['path'].startswith(('control32/','dev64/'))]
    forbidden={'report.json','audit-progress.json','joined-emitted-raw.wav','timed-emitted-raw.wav'}
    if not selected or any(Path(r['path']).name in forbidden for r in selected):
        raise ValueError('Pristine complete generation-only tree required')
    if target.exists() or source.resolve()==target.resolve():
        raise ValueError('Fresh separate audit namespace required')
    target.mkdir(parents=True)
    for row in selected:
        path=Path(row['path'])
        if path.is_absolute()or '..'in path.parts:raise ValueError('Contained original relative paths required')
        src,dst=source/path,target/path;dst.parent.mkdir(parents=True,exist_ok=True)
        if row['kind']=='symlink':os.symlink(row['target'],dst)
        elif row['kind']=='file':os.link(src,dst)
        else:raise ValueError('Only original regular files and preserved link text allowed')
    if inventory(target)!=selected:
        raise ValueError('Every cloned original byte must match the independently backed inventory')
    return dict(files=len(selected),bytes=sum(r['bytes']for r in selected),all_original_bytes_verified=True)


def execute_recovery(run,preserve):
    run('readout','audit-saved')
    receipts={'readout/audit':preserve('readout','evaluate',0)}
    run('bias','fit');receipts['bias/fit']=preserve('bias','fit',128)
    run('bias','evaluate');receipts['bias/evaluate']=preserve('bias','evaluate',0)
    return receipts
