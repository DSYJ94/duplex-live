"""One owner-approved rollback; old89 is retained as a rejected checkpoint."""
import json
from pathlib import Path
from . import matched_guard as g
from .matched_interrupted import DURABLE
from .matched_failed_state import PREDECESSOR

AUTH=Path('docs/NATIVE-MOSHI-NORMAL88-RECOVERY-AUTHORIZATION-2026-11-10.json')
PROOF=Path('docs/NATIVE-MOSHI-RESTORE89-FAILURE-2026-11-10.json')
CHECKPOINT=DURABLE/'preserved/personaplex-normal-step88-v1/live.pt'
SHA='287d2f3159a340b1f17ae38e511119187a727a3e80e2bfc9c05c6485fdd796c9'
MARKER='owner-approved-raw88-rollback-v1'


def authority():
    value=json.loads(AUTH.read_text())
    expected=dict(status='OWNER_APPROVED_NORMAL88_ROLLBACK_RECOVERY',owner_answer='Approve rollback; keep existing caps',
        proposal_commit='a03453fe7ad8e168def8c93966b1ffef9b050492',provider='Isambard',account='brics.u6ada',
        failed_job='8273944',failed_allocated_seconds=119,actual_prior_GPU_seconds=34059,
        maximum_aggregate_GPU_hours=112,maximum_total_running_intervals=6,next_interval=5,
        maximum_single_job_seconds=86400,simultaneous_GPUs=2,personaplex_step=88,context_spanning_step=80,
        deliberate_saved_update_replay=[89],stop_if_either_narrow_restore_fails=True,
        all_four_fresh_native_restores_before_learning=True,new_purchases=False,new_GCP_compute=False,
        new_external_model_API_calls=0,OneDrive_writes=False,Mac_checkpoint_downloads=False,
        all_old_checkpoints_and_failed_artifacts_preserved=True)
    if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Exact owner one-update rollback grant required')
    return g.sha(AUTH)


def reference_allowed(a,producer):
    if (producer!=PREDECESSOR or getattr(a,'base',None)!='personaplex' or getattr(a,'capacity',None)!='narrow'
            or getattr(a,'checkpoint_sha',None)!=SHA or Path(a.checkpoint)!=CHECKPOINT):return False
    authority();return True


def validate(row,reference,step):
    authority();proof=json.loads(PROOF.read_text())
    if proof!=json.loads((DURABLE/'failure8273944-normal88-preservation.json').read_text()):
        raise ValueError('Actual failure/normal88 preservation evidence changed')
    if (proof['failed_job']!='8273944' or proof['actual_exit']!='1:0' or proof['allocated_seconds']!=119
            or proof['GPUs']!=2 or proof['actual_updates']!=0 or proof['all_prior_GPU_seconds']!=34059):
        raise ValueError('Actual failed start4 must remain failed and counted')
    original=proof['normal88']
    if (row.get('normal88_recovery')!=MARKER or row['base']!='personaplex' or row['capacity']!='narrow'
            or step!=88 or Path(row['checkpoint'])!=CHECKPOINT or row['checkpoint_sha256']!=SHA
            or reference['checkpoint_sha256']!=SHA or reference['step']!=88 or reference['source']!=PREDECESSOR
            or reference['execution_id']!=original['execution_id'] or original['sha256']!=SHA
            or original['step']!=88 or Path(original['path'])!=CHECKPOINT or not original['both_hashes_verified']):
        raise ValueError('Only unchanged pinned normal raw88 may take this rollback exception')
    old=DURABLE/'runs/quota-recovery-8260542/personaplex/narrow/train/live.pt.native-reference.json'
    if reference!=json.loads(old.read_text()):raise ValueError('Original native reference must not be rewritten')
    return True
