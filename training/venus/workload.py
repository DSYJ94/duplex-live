"""Whole-corpus runtime accounting; never selects by loss or changes examples."""
from .large_inputs import MANIFEST_SHA

FEATURES=('tokens','segments','prefix_tokens','attention_cells','head_tokens')


def geometry(row,example):
    wanted={int(p) for p in example['positions']}
    segments=example['segments']
    return dict(id=row['id'],fold=row['fold'],family=row['family'],tokens=len(example['embeddings']),
                segments=len(segments),prefix_tokens=sum(s['start']+s['length'] for s in segments),
                attention_cells=sum(s['length']*(s['start']+s['length']) for s in segments),
                head_tokens=sum(s['length'] for s in segments if s['start']+s['length']-1 in wanted))


def whole_corpus(rows,calibration_ids):
    if (len(rows)!=304 or len({r['id'] for r in rows})!=304
            or sum(r['fold']=='TRAIN' for r in rows)!=240 or sum(r['fold']=='DEV' for r in rows)!=64
            or len(calibration_ids)!=4 or len(set(calibration_ids))!=4):
        raise ValueError('Whole304/240TRAIN64DEV and allfour calibration rows required')
    train={r['id']:r for r in rows if r['fold']=='TRAIN'}
    if any(i not in train for i in calibration_ids):raise ValueError('Calibration requires TRAIN only')
    if any(any(type(r[k]) is not int or r[k]<=0 for k in FEATURES) for r in rows):
        raise ValueError('Positive exact native geometry required')
    normalizers={k:max(train[i][k] for i in calibration_ids) for k in FEATURES}
    # Ordinary scenes perform the additional frozen-base retention forward.
    # A factor2 for all work is conservative even though backward runs once.
    weights={i:max(r[k]/normalizers[k] for k in FEATURES)*(2 if r['family']=='ordinary' else 1) for i,r in train.items()}
    return dict(cache_manifest_sha256=MANIFEST_SHA,train_ids=sorted(train),train_count=240,dev_count=64,
                calibration_ids=list(calibration_ids),normalizers=normalizers,weights=weights,
                train_weight=sum(weights.values()),calibration_weight=sum(weights[i] for i in calibration_ids),
                uses_losses_or_answers=False,rows=rows)


def epoch_seconds(main_step,control_step,work):
    if (work['cache_manifest_sha256']!=MANIFEST_SHA or work['train_count']!=240 or work['dev_count']!=64
            or len(work['train_ids'])!=240 or len(set(work['train_ids']))!=240
            or set(work['weights'])!=set(work['train_ids']) or min(main_step,control_step)<=0):
        raise ValueError('Complete original workload and actual both-arm timings required')
    rebuilt=whole_corpus(work['rows'],work['calibration_ids'])
    if rebuilt!=work:raise ValueError('Runtime weights must derive from complete frozen geometry')
    main=main_step*work['train_weight']/work['calibration_weight']+240
    # Both arms cover the identical240scenes. Their actual runtimes differ;
    # use the same complete geometry with each separately measured baseline.
    control=control_step*work['train_weight']/work['calibration_weight']+240
    return dict(main=main,control=control,scenes_each=240,
                main_basis='complete240nativegeometry/all4measuredlongrows',control_basis='complete240geometry/actual4prefixrows',
                fixed_scene_overhead_seconds=1,selection_uses_losses_or_heldout=False)
