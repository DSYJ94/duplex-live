"""Resolve an explicitly mixed component without bypassing its own validator."""


def manifest_name(folder, early=False, mixed=False):
    if mixed:
        names=[name for name in ('early-future.json','actual-future.json') if (folder/name).is_file()]
        if len(names)!=1:
            raise RuntimeError('Mixed component needs exactly one unambiguous capture manifest')
        return names[0]
    return 'early-future.json' if early else 'actual-future.json'


def balanced_index(kinds, step):
    """Cycle equally over kinds, then over examples within each kind."""
    if step<0 or not kinds or any(not isinstance(k,str) or not k for k in kinds):
        raise ValueError('Balanced sampling needs nonempty verified kinds and a nonnegative step')
    groups={kind:[i for i,k in enumerate(kinds) if k==kind] for kind in sorted(set(kinds))}
    group=groups[sorted(groups)[step%len(groups)]]
    return group[(step//len(groups))%len(group)]
