"""Select different approved TRAIN worlds for the same original user question.

This returns contrast candidates only. The caller must check native first-token
positions and actual reference clocks before using either conditioning stream.
"""
import re


def world_key(identity, sidecar):
    match = re.fullmatch(r'teacher-cf-(recovery-train-\d+)-[0-3]', identity)
    meta = sidecar['timeline']['meta']
    if not match or meta['split'] != 'train' or meta['kind'] != 'private':
        return None
    user = tuple((e['start_ms'], e['text']) for e in sidecar['timeline']['events']
                 if e['ch'] == 'user' and e['kind'] == 'utterance')
    if not user or not meta['phrase'] or len(sidecar['reference_arrivals']) != 1:
        raise ValueError('Paired world lacks an approved user question or actual receipt')
    return match[1], meta['project'], meta['field'], user


def contrast_candidates(sidecars):
    keys = {identity:world_key(identity, sidecar) for identity,sidecar in sidecars.items()}
    result = []
    for positive in sorted(keys):
        if keys[positive] is None:
            continue
        for negative in sorted(keys):
            if negative == positive or keys[negative] != keys[positive]:
                continue
            a = sidecars[positive]['timeline']['meta']['phrase'].split()[0].casefold()
            b = sidecars[negative]['timeline']['meta']['phrase'].split()[0].casefold()
            if a != b:
                result.append((positive,negative))
    return result
