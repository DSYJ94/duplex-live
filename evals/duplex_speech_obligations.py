"""Actual ordinary replies still owed by the hybrid client.

IDs refer to responses, not cancelable private tools. Content comes from actual
controller results, never reference answers. Heard-prefix alignment is unknown.
"""


def snapshot(queue,active=None):
    by_id={}
    for item in list(queue)+([active] if active else []):
        if item['result']['route']!='answer':continue
        row=dict(id=item['task_id'],request_text=item['request_text'],text=item['result']['content'])
        if row['id'] in by_id and row!=by_id[row['id']]:raise RuntimeError('Conflicting actual owed-response state')
        by_id[row['id']]=row
    return list(by_id.values())


def consume(queue,identities,approved_snapshot):
    if (not isinstance(identities,list) or any(not isinstance(i,str) for i in identities)
            or len(set(identities))!=len(identities) or not set(identities)<={p['id'] for p in approved_snapshot}):
        raise ValueError('Cannot consume an unknown or duplicate speech obligation')
    selected=set(identities)
    return [item for item in queue if item['task_id'] not in selected]
