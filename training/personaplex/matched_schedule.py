"""Complete coverage and explicit objective mass for the approved four-arm study.

This is a new stratified schedule, not the old full154 batch at every update.
It grants no data, native-gradient, restore or behavioral approval.
"""
from collections import Counter
import math

SIZES = dict(actual=28, reference=4, old25=25, new24=24, word=10,
             clock=14, audio49=49, stream49=49, control40=40,
             reasoning32=32, ordinary48=48)
STEPS = 49
EPOCHS = 4
BALANCED = frozenset(('actual', 'old25', 'new24', 'stream49'))


def validate(groups):
    if set(groups) != set(SIZES):
        raise ValueError('Every retained objective and all48 new ordinary scenes required')
    for name, size in SIZES.items():
        rows = groups[name]
        if len(rows) != size or len({r['id'] for r in rows}) != size:
            raise ValueError('Changed complete membership: ' + name)
        if any(r.get('fold') != 'TRAIN' for r in rows):
            raise ValueError('No DEV or held-out target may enter training')
        if name in BALANCED and any(not isinstance(r.get('kind'), str) or not r['kind'] for r in rows):
            raise ValueError('Original balanced-kind labels required: ' + name)
    return True


def weights(name, rows):
    mass = 64. if name in ('word', 'clock') else 1.
    if name in BALANCED:
        counts = Counter(r['kind'] for r in rows)
        return [mass / (len(counts) * counts[r['kind']]) for r in rows]
    return [mass / len(rows)] * len(rows)


def epoch(groups, number):
    """Return all49 updates, with inverse-visit correction per component.

    If the old full-batch component weight is w and it appears v times here,
    each visit gets49*w/v. Thus the cycle MEAN gives exactly w to that
    component. Smaller groups rotate across epochs without dropping members.
    """
    validate(groups)
    if type(number) is not int or not 0 <= number < EPOCHS:
        raise ValueError('Only the four approved complete epochs')
    schedule = [[] for _ in range(STEPS)]
    for group, size in SIZES.items():
        order = [(number * STEPS + step) % size for step in range(STEPS)]
        visits = Counter(order)
        original = weights(group, groups[group])
        for step, index in enumerate(order):
            schedule[step].append(dict(group=group, index=index,
                id=groups[group][index]['id'], weight=STEPS * original[index] / visits[index],
                visits_this_epoch=visits[index], original_component_weight=original[index]))
    audit(groups, schedule)
    return schedule


def audit(groups, schedule):
    """Check actual exposure records, including order, membership and mass."""
    validate(groups)
    if len(schedule) != STEPS:
        raise ValueError('A passing subset is not a complete coverage epoch')
    for step in schedule:
        if len(step) != len(SIZES) or {x['group'] for x in step} != set(SIZES):
            raise ValueError('Every update must retain every objective group')
    for group, rows in groups.items():
        actual = {r['id']: 0. for r in rows}
        visits = Counter()
        for step in schedule:
            row = next(x for x in step if x['group'] == group)
            index = row['index']
            if type(index) is not int or not 0 <= index < len(rows) or rows[index]['id'] != row['id']:
                raise ValueError('Changed component identity')
            if type(row['weight']) not in (int, float) or not math.isfinite(row['weight']) or row['weight'] <= 0:
                raise ValueError('Invalid component weight')
            actual[row['id']] += row['weight'] / STEPS
            visits[row['id']] += 1
        if set(visits) != set(actual):
            raise ValueError('A retained member was never visited')
        for row, expected in zip(rows, weights(group, rows)):
            if not math.isclose(actual[row['id']], expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Changed complete-cycle objective mass: ' + group)
    return True
