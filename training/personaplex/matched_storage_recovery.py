"""Exact owner-approved normal112/104 recovery after actual failed8274714."""
import json,math
from pathlib import Path
from . import matched_guard as g
from .matched_interrupted import DURABLE

AUTH=Path('docs/NATIVE-MOSHI-STORAGE-RECOVERY-AUTHORIZATION-2026-11-10.json')
PROPOSAL=Path('docs/NATIVE-MOSHI-STORAGE-RECOVERY-PROPOSAL-2026-11-10.json')
PROOF=Path('docs/NATIVE-MOSHI-STORAGE-STOP-2026-11-10.json')
PREDECESSOR='15a0420d901a6462da4a18e93a93a59904098b27'
FAILED_ROOT=DURABLE/'runs/normal88-recovery-8274714'
MARKER='owner-approved-normal112-104-storage-recovery-v1'
STATES={'personaplex':(112,'2d2fe9a5b6a4efc160543c444912a4a3e7a3c2f81029f06d511d186e4d50d80c'),
        'context-spanning':(104,'22cab2fbeccc42d1a361d6c45819e31a67a68b05e65e2b9439e053bb5ceafd3c')}


def authority():
    value=json.loads(AUTH.read_text());proposal=json.loads(PROPOSAL.read_text())
    expected=dict(status='OWNER_APPROVED_STORAGE_RECOVERY',owner_answer='Approve recovery; keep 112 GPU-hours',
        proposal_commit='203166171d3b64cc85333f2b1e365a382737539a',provider='Isambard',account='brics.u6ada',
        failed_job='8274714',actual_native_producer=PREDECESSOR,failed_allocated_seconds=5325,
        actual_prior_GPU_seconds=44709,maximum_aggregate_GPU_hours=112,additional_GPU_hours=0,
        simultaneous_GPUs=2,maximum_single_job_seconds=86400,next_interval=6,proposed_maximum_total_running_intervals=8,
        first_recovery_allowed_after_failed_window5=True,new_GCP_compute=False,new_purchases=False,
        new_external_model_API_calls=0,OneDrive_writes=False,Mac_checkpoint_downloads=False)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Exact owner storage-recovery grant required')
    changes={'status','owner_answer','proposal_commit','question_call'}
    if {k:v for k,v in value.items() if k not in changes}!={k:v for k,v in proposal.items() if k not in changes|{'no_submission_until_actual_owner_scope_amendment'}}:
        raise ValueError('Owner grant must preserve the exact approved proposal')
    return g.sha(AUTH)


def reference_allowed(a,producer):
    base=getattr(a,'base',None)
    if producer!=PREDECESSOR or base not in STATES or getattr(a,'capacity',None)!='narrow':return False
    if (getattr(a,'checkpoint_sha',None)!=STATES[base][1]
            or Path(a.checkpoint)!=FAILED_ROOT/base/'narrow/train/live.pt'):return False
    authority();return True


def validate(row,reference,step):
    authority();base=row['base']
    if base not in STATES:raise ValueError('Unknown selected normal-state lineage')
    expected,digest=STATES[base];checkpoint=FAILED_ROOT/base/'narrow/train/live.pt'
    if (row.get('storage_recovery')!=MARKER or row['capacity']!='narrow' or step!=expected
            or Path(row['checkpoint'])!=checkpoint or row['checkpoint_sha256']!=digest
            or reference['checkpoint_sha256']!=digest or reference['step']!=expected or reference['source']!=PREDECESSOR):
        raise ValueError('Only exact approved normal112/104 states may recover')
    proof=json.loads(PROOF.read_text())
    if proof!=json.loads((FAILED_ROOT/'failed-preservation-inventory.json').read_text()):
        raise ValueError('Actual storage failure and preservation evidence changed')
    if (proof['job']!='8274714' or proof['total_GPU_seconds']!=44709
            or proof['host_terminal']['actual_exit_code']!=1 or '|FAILED|1:0|' not in proof['actual_sacct']):
        raise ValueError('Failed window must remain failed and fully counted')
    record=next(r for r in proof['backup_records'] if r['checkpoint']==str(checkpoint))
    if (record['step']!=expected or record['sha256']!=digest or record['hash_verified'] is not True
            or record['separate_filesystem'] is not True or record['actual_rsync_exit_code']!=0):
        raise ValueError('Actual independent normal-state preservation required')
    backup=Path(record['destination'])
    if (reference!=json.loads(Path(str(backup)+'.native-reference.json').read_text())
            or g.sha(Path(str(checkpoint)+'.native-reference.json'))!=record['files']['live.pt.native-reference.json']['sha256']):
        raise ValueError('Original native reference must stay unchanged')
    logs=[json.loads(s) for s in (checkpoint.parent/'training.jsonl').read_text().splitlines()]
    first,last=(89,117) if base=='personaplex' else (81,109)
    if ([r['step'] for r in logs]!=list(range(first,last+1))
            or any(r['source']!=PREDECESSOR or not math.isfinite(r['objective']) or r['LR']<=0 for r in logs)):
        raise ValueError('Actual failed-window training log must remain original')
    return True
