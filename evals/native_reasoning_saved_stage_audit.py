"""Read-only audit of the exact V18 eight-update export; no recovery or approval.

This verifies preserved bytes and complete logged objective exposure. It never
loads Torch, writes a fit report, changes a process, or authorizes GPU work.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

SOURCE = 'da389a7031a5504495397d1575952194ffa209f1'
CONTROLLER_SOURCE = '0d75986fca9f1ef943dfece77c7859235163c0ad'
FIT_COMMAND_SHA256 = 'e4ad94d23096960c0c81800d6983adcaa304f842638218cd840afb69e36bfc9f'
STEPS_SHA256 = '2eef7695db0d090a95b8cc77a0bac090acae8ceb734209249db9ca7f0bb3384d'
PILOT = 'native-reasoning32-retained-pilot-v18'
READY = 'native-reasoning32-retained-pilot-v17/combined-readiness/reasoning32-combined-readiness.json'
PINNED = {
    PILOT+'/training/training-state.pt': '75fd39b1dc9a7f66b8125a1804cab87cfad384446d592df4534c3e1cff5f0367',
    PILOT+'/training/adapters.safetensors': '34edb69c696fef1ac02e498bb1351a7655442d6c7c041562ca145abfb3a77176',
    PILOT+'/training/data.json': 'de92099057ded15a1af0e387c96b457cdf4c1731c26594dcdca211d4c442fc1e',
    PILOT+'/training/voice.pt': '2085b3d58d2f2b20042e2e57a60206dd43c3028173cd9140e83d6c7b5136404d',
    PILOT+'/training/approved-complete-balanced-preflight.json': '2c695e6006cc82af426903be3d0a5a813bfd4fafcae72f010acf47b53722c086',
    READY: '46753a71712712250aff747661bf3de0d9e8d165f12977f03c90ac6d04df20eb',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def confined(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Relative confined artifact path required')
    resolved = (root/relative).resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError('Artifact escaped the verified snapshot')
    return resolved


def optimizer_rows(text):
    rows = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # Framework warnings are retained in the original log.
        if isinstance(row, dict) and 'reasoning32_microbatch' in row and 'step' in row:
            rows.append(row)
    if [r['step'] for r in rows] != list(range(1, 9)):
        raise ValueError('Exactly eight ordered optimizer receipts, without duplicates, required')
    if any(not math.isfinite(r['gradient_norm_before_clip']) or r['gradient_norm_before_clip']<=0
           for r in rows):
        raise ValueError('Every actual optimizer receipt must have a finite positive norm')
    return rows


def terminal_timeout(state, controller_handle, fit_handle):
    """Check supplied original-host observations; this is never a launch permit."""
    if (state['pid']!=28726 or state['source_commit']!=SOURCE
            or state['controller_commit']!=CONTROLLER_SOURCE or state['pod_id']!='8fsk2kouknoiac'
            or state['phase']!='failed_preserved_no_retry' or state['passed'] is not False
            or state.get('error_type')!='TimeoutExpired' or state['optimizer_update_limit']!=8
            or [r['name'] for r in state['stages']]!=[
                'whole_workset','gradient32','combined_readiness','bounded8_training']
            or any(r.get('actual_exit')!=0 for r in state['stages'][:3])):
        raise ValueError('Exact original post-fit timeout and complete pre-fit approvals required')
    fit = state['stages'][-1]
    if (fit['pid']!=28807 or fit['birth']!=40829227
            or type(fit.get('actual_exit')) is not int or fit['actual_exit']==0):
        raise ValueError('Original failed fit exit must remain recorded')
    command_hash = hashlib.sha256(json.dumps(state['planned_commands']['bounded8_training'],
        separators=(',', ':')).encode()).hexdigest()
    if command_hash!=FIT_COMMAND_SHA256:
        raise ValueError('Original fit command changed')
    for handle, birth in ((controller_handle,40829059),(fit_handle,40829227)):
        if handle is not None and (handle['birth']!=birth or handle['state']!='Z'):
            raise ValueError('Original-host controller and fit must actually be terminal')
    return True


def audit(snapshot):
    from evals.context_native_complete_batch import validate_exposure
    from evals.context_native_stream_training import exposure as stream_exposure
    from evals.native_control_training import exposure as control_exposure
    from evals.native_reasoning_training import exposure as reasoning_exposure
    root = Path(snapshot).resolve(strict=True)
    inventory_path = root/'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    entries = {r['path']: r for r in inventory['files']}
    if len(entries)!=len(inventory['files']):
        raise ValueError('Duplicate inventory paths')
    names = [*PINNED, PILOT+'/bounded8_training.log', PILOT+'/controller.json']
    verified = {}
    for name in names:
        path = confined(root, name); entry = entries[name]; actual = digest(path)
        if (path.stat().st_size!=entry['bytes'] or actual!=entry['sha256']
                or name in PINNED and actual!=PINNED[name]):
            raise ValueError('Changed saved-stage artifact: '+name)
        verified[name] = dict(bytes=entry['bytes'], sha256=actual)
    state = json.loads((root/PILOT/'controller.json').read_text())
    if (state['source_commit']!=SOURCE or state['controller_commit']!=CONTROLLER_SOURCE
            or state['pid']!=28726 or state['pod_id']!='8fsk2kouknoiac'):
        raise ValueError('Saved export belongs to a different execution')
    steps = optimizer_rows((root/PILOT/'bounded8_training.log').read_text())
    steps_hash = hashlib.sha256(json.dumps(steps,sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()
    if steps_hash!=STEPS_SHA256:
        raise ValueError('Original eight optimizer receipts changed')
    ready = json.loads((root/READY).read_text())
    original = json.loads((root/PILOT/'training/approved-complete-balanced-preflight.json').read_text())
    policy = original['policy']
    acoustic = policy['assistant_audio_loss_targets_per8']
    if acoustic%8:
        raise ValueError('Complete49 per-epoch target count changed')
    validate_exposure(steps,policy['teacher_ids'],acoustic//8,policy['complete_batch_membership'])
    stream_exposure(steps,ready['native49_policy'])
    control_exposure(steps,ready['control40_policy'])
    reasoning_exposure(steps,ready['policy'])
    for step in steps:
        if any(not math.isfinite(row['loss']) for rows in step['complete_balanced_microbatches'].values()
               for row in rows):
            raise ValueError('Nonfinite retained objective')
    return dict(schema='dl-native-reasoning32-saved-eight-audit-v1',
        at_utc=datetime.now(timezone.utc).isoformat(), inventory_sha256=digest(inventory_path),
        audit_implementation_sha256=digest(Path(__file__)), optimizer_receipts_sha256=steps_hash,
        source_commit=SOURCE, snapshot=str(root), verified_files=verified,
        complete_eight_optimizer_rows=True, retained154_native49_control40_reasoning32_exposure_verified=True,
        observed_original_phase=state['phase'], optimizer_updates_executed=0,
        original_fit_exit_zero_inferred=False, training_state_loaded_or_restored=False,
        fit_finalization_approved=False, independent_restore_approved=False,
        gpu_or_training_authorized=False, behavior_qualified=False, model_adopted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.snapshot)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
