"""Read verified completed histories from the controlled v50 data-only stop.

Inherited numerical results stay explicitly inherited; they are never described
as a new replay or a successful original whole job. No optimizer/model runs here.
"""
import hashlib,json
from pathlib import Path

PRODUCER='bd7ea9aee6b1a9ac36278c5fda9a9e2df3ed31ac'
WITNESS=Path('docs/NATIVE-VENUS-CONTEXT-PARTIAL14-REUSE-WITNESS-2026-11-02.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admitted_rows(root,witness=WITNESS):
    if Path(witness).resolve()!=WITNESS.resolve():raise ValueError('Exact recorded partial14 witness required')
    w=json.loads(WITNESS.read_text())
    if w['producer_source']!=PRODUCER or w['original_job_terminal_success'] or w['verified_completed']!=14 or w['required_new_scenes']!=36 or not w['controlled_stop']:
        raise ValueError('Preserve actual partial/controlled status and exact producer')
    rows={r['id']:r for r in w['rows']}
    if len(rows)!=14:raise ValueError('All14 previously verified histories required')
    for r in rows.values():
        path=Path(root)/(r['id']+'.pt');p=r['proof']
        if path.is_symlink() or path.stat().st_size!=r['bytes'] or sha(path)!=r['sha256']:
            raise ValueError('Original saved history hash/bytes changed')
        if not p['passed'] or p['logit_error']!=0 or p['hidden_error']!=0:
            raise ValueError('Actual original exact prefix parity required')
    return rows


def load_verified(path,row,timeline,duplex,expected_layers=35):
    import torch
    from .native_schedule import schedule
    if Path(path).is_symlink() or Path(path).stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Saved history changed before loading')
    ex=torch.load(path,map_location='cpu',weights_only=False)
    expected_hash=hashlib.sha256(json.dumps(timeline,sort_keys=True).encode()).hexdigest()
    if (ex['timeline_sha256']!=expected_hash or ex['id']!=row['id'] or ex['fold']!=timeline['fold'] or ex['family']!=timeline['family']
            or ex['forward_path']!='native-frozen-prefix-v1' or ex['prefix_layers']!=expected_layers or not ex['native_teacher_forced'] or ex['student_behavior_claimed']):
        raise ValueError('Exact original source/fold/native labels required')
    scheduled=schedule(duplex,timeline)
    targets=[token for frame in range(timeline['frames']) for token in scheduled.get(frame,[duplex.listen_token_id])]
    if ex['targets'].tolist()!=targets or len(ex['positions'])!=row['proof']['positions']:
        raise ValueError('Every original native target must equal the current exact schedule')
    if len(ex['prefix_inputs'])!=len(ex['segments']):raise ValueError('Complete original prefix segments required')
    for entry,segment in zip(ex['prefix_inputs'],ex['segments']):
        if entry['hidden'].shape!=(1,segment['length'],ex['embeddings'].shape[-1]):raise ValueError('Invalid original prefix shape')
    proof=dict(row['proof'],kind='inherited_pinned_producer_pre_serialization_exact_check',producer_source=PRODUCER,original_file_sha256=row['sha256'],independently_recomputed_this_process=False,original_whole_job_terminal_success=False)
    return ex,proof
