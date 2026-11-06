"""Explicitly scoped CPU replacement; shares recovery math, never old grants.

Only offline --plan is available without the consolidated owner authorization.
Run in its own process so the independent guard applies the exact same settings.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

from . import results_recovery as recovery

AUTH = Path('docs/NATIVE-BRITISH-SPEECH-RECOVERY-AUTHORIZATION-2026-11-06.json')
PROPOSAL = Path('docs/NATIVE-BRITISH-SPEECH-RECOVERY-PROPOSAL-2026-11-06.json')
PRIOR = Path('runs/venus-large-capacity-20261104/results-recovery-v82')


def prior_usage():
    vm = json.loads((PRIOR/'provider-final.json').read_text())
    if vm['id'] != '1327566272434576329' or vm['status'] != 'TERMINATED':
        raise ValueError('Exact prior CPU terminal required')
    elapsed = datetime.fromisoformat(vm['lastStopTimestamp']).timestamp()-datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    if not 137.69 <= elapsed <= 137.71: raise ValueError('Prior CPU usage changed')
    binding = json.loads((PRIOR/'original-binding-restored.json').read_text())
    if binding != dict(original_vm_id=recovery.GPU_ID,disk_id=recovery.DISK_ID,status='TERMINATED',auto_delete=False):
        raise ValueError('Original disk restoration receipt required')
    if not json.loads((PRIOR/'temporary-cpu-removed.json').read_text())['original_disk_preserved']:
        raise ValueError('Prior CPU cleanup required')
    return elapsed


def require_authorization():
    a=json.loads(AUTH.read_text())
    p=json.loads(PROPOSAL.read_text())
    if a.get('status')!='APPROVED' or not a.get('human_reply_verbatim') or a.get('proposal_path')!=str(PROPOSAL) or a.get('proposal_sha256')!=recovery.sha(PROPOSAL):
        raise ValueError('Exact consolidated owner approval required')
    if a.get('maximum_new_CPU_intervals')!=1 or a.get('maximum_new_CPU_seconds')!=600 or a.get('CPU_recovery_total_USD_maximum')!=.10:
        raise ValueError('Exact remaining CPU allowance required')
    if a.get('phase_ceiling_USD')!=175 or a.get('project_ceiling_USD')!=350 or a.get('same_disk_id')!=recovery.DISK_ID or a.get('same_GPU_id')!=recovery.GPU_ID:
        raise ValueError('Existing resource/cost limits required')
    c=p['CPU']
    if c['maximum_new_intervals']!=1 or c['maximum_new_running_seconds']!=600 or c['recovery_total_USD_maximum']!=.10 or c['name']!='duplex-venus-results-cpu-20261106-v2':
        raise ValueError('Consolidated proposal CPU limits changed')
    elapsed=prior_usage()
    if elapsed+600+100>900 or time.time()>=recovery.DEADLINE-1200:
        raise ValueError('Whole CPU recovery must fit old aggregate time/deadline')
    return a


def configure():
    recovery.R=Path('runs/venus-large-capacity-20261104/results-recovery-v83')
    recovery.CPU='duplex-venus-results-cpu-20261106-v2'
    recovery.PROPOSAL=PROPOSAL
    recovery.AUTH=AUTH
    recovery.RUN_SECONDS=600
    # Carry prior actual usage at the ORIGINAL maximum quote, conservatively.
    recovery.PRIOR_CPU_USD=prior_usage()*.12/3600
    recovery.ENTRYPOINT='training.venus.british_recovery'
    recovery.require_authorization=require_authorization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ('plan','quote','execute','guard'):mode.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    configure()
    if args.plan:
        _,rows=recovery.manifest()
        print(json.dumps(dict(missing_files=len(rows),missing_bytes=sum(x['bytes'] for x in rows),batches=len(list(recovery.batches(rows))),prior_CPU_actual_seconds=prior_usage(),next_CPU_maximum_seconds=600,authorized=AUTH.exists(),cloud_mutations=0)))
    elif args.quote:
        price=recovery.quote();print(json.dumps(dict(hourly_USD=price['hourly_usd'],cloud_mutations=0)))
    elif args.guard:recovery.guard()
    else:recovery.execute()


if __name__=='__main__':main()
