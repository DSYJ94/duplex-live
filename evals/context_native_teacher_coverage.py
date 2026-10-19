"""Declared full24 TRAIN coverage in eight updates, without changing kind weights."""
from evals.context_native_teacher_objective import teacher_index


def complete_indices(component_kinds, step, reference_index, teacher_kinds):
    """Average one half of the matching teacher kind per visit; no subset omission."""
    names = ('private', 'simple', 'clarify', 'cancel')
    groups = [[i for i,kind in enumerate(teacher_kinds) if kind == name] for name in names]
    if len(teacher_kinds) != 24 or [len(group) for group in groups] != [8,8,4,4]:
        raise ValueError('Complete24 coverage requires every approved native teacher in all four kinds')
    anchor = teacher_index(component_kinds, step, reference_index, teacher_kinds)
    group = groups[reference_index]; half = len(group)//2
    start = (group.index(anchor)%2)*half
    return group[start:start+half]
