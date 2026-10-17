"""Resolve an explicitly mixed component without bypassing its own validator."""


def manifest_name(folder, early=False, mixed=False):
    if mixed:
        names=[name for name in ('early-future.json','actual-future.json') if (folder/name).is_file()]
        if len(names)!=1:
            raise RuntimeError('Mixed component needs exactly one unambiguous capture manifest')
        return names[0]
    return 'early-future.json' if early else 'actual-future.json'
