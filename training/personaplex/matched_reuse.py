"""CPU-only reuse preflight; cannot authorize, submit or run a model.

Original admission and restore bytes keep their actual producer. An offline
copy maps only their original root into its verified copyback directory.
A future training producer still needs explicit source compatibility and a NEW
actual fresh-process native restore; this report grants neither approval.
"""
import argparse,hashlib,json,math
from pathlib import Path
from .matched_native import BASES,CAPACITIES,verified_admission

ORIGIN='8b822b0d85ccea4f291255204cbc358fa85469c4'
ORIGINAL_ROOT=Path('/projects/u6ada/duplex-live/native-moshi-matched-train-v1')
DATA_SHA='72e2b514b1c922f6591e18e1d30b198628170db2a2cc5e2efdb12d8d3d538e26'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actual_usage(text,expected):
    """Count only exact top-level actual allocations, never step duplicates."""
    rows={}
    for line in text.splitlines():
        fields=line.split('|')
        if fields[0] not in expected:
            if fields[0].isdigit():raise ValueError('Uncounted top-level allocation')
            continue
        if fields[0] in rows or len(fields)<5:raise ValueError('Duplicate/incomplete actual allocation')
        ident,account,state,elapsed,tres=fields[:5]
        wanted=expected[ident]
        resources=dict(p.split('=',1) for p in tres.split(',') if '=' in p)
        if (account!='brics.u6ada' or state!=wanted['state'] or not elapsed.isdigit()
                or int(elapsed)!=wanted['elapsed'] or resources.get('gres/gpu')!=str(wanted['GPUs'])):
            raise ValueError('Actual scheduler accounting differs from pinned history')
        rows[ident]=int(elapsed)*wanted['GPUs']
    if rows.keys()!=expected.keys():raise ValueError('All actual phase allocations required')
    return sum(rows.values())


def preflight(root,packet_sha):
    root=Path(root).resolve();packet=root/'joint-admission.json'
    if packet.is_symlink() or digest(packet)!=packet_sha:raise ValueError('Exact original joint packet required')
    value=json.loads(packet.read_text());entries=value['entries']
    if (value['source']!=ORIGIN or value['data_report_sha256']!=DATA_SHA or len(entries)!=4
            or {(r['base'],r['capacity']) for r in entries}!={(b,c) for b in BASES for c in CAPACITIES}):
        raise ValueError('Exact complete four-scope original comparison required')
    def read_path(original):
        relative=Path(original).relative_to(ORIGINAL_ROOT)
        physical=root/relative
        physical.resolve().relative_to(root)
        if '..' in relative.parts or physical.is_symlink():
            raise ValueError('Original proof namespace escape')
        return physical
    rows=[];seconds={b:0. for b in BASES}
    for row in entries:
        base,cap=row['base'],row['capacity']
        expected=ORIGINAL_ROOT/base/cap
        if (Path(row['admission'])!=expected/'admit/admission.json'
                or Path(row['restore'])!=expected/'restore/restore.json'):
            raise ValueError('Original lineage proof crossed another arm')
        admission,restore,estimate=verified_admission(row,ORIGIN,DATA_SHA,read_path)
        if not math.isfinite(estimate) or estimate<=0:raise ValueError('Actual measured forecast required')
        seconds[base]+=estimate
        rows.append(dict(base=base,capacity=cap,original_source=ORIGIN,
            original_admission=row['admission'],admission_sha256=row['admission_sha256'],
            original_restore=row['restore'],restore_sha256=row['restore_sha256'],
            original_checkpoint=str(expected/'admit/init.pt'),checkpoint_sha256=admission['checkpoint_sha256'],
            actual_restored_step=restore['restored_training_step'],estimated_complete196_seconds=estimate))
    return dict(original_source=ORIGIN,data_report_sha256=DATA_SHA,original_joint_packet_sha256=packet_sha,
        all_four_original_admissions_and_restores_verified=True,lineages=rows,
        serial_forecast_seconds_per_base=seconds,future_GPU_seconds_with6h_reserve=2*max(seconds.values())+21600,
        original_proofs_unchanged=True,optimizer_updates_performed=0,model_executed=False,
        new_source_compatibility_and_fresh_native_restore_still_required=True,
        training_or_runtime_extension_authorized=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--packet-sha',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=preflight(a.root,a.packet_sha)
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(all_four_original_admissions_and_restores_verified=True,training_authorized=False)))
