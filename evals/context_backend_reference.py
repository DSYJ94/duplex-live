"""Keep actual tool facts in a context span even if worker prose omits them."""


def reference(result, world):
    content = result['content']
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('Cannot inject empty backend text')
    # A failed request remains a failure notice, even if it executed a read.
    if result['route'] == 'error':
        return content, 0
    facts = []
    with world.lock:
        for receipt in result['receipts']:
            if receipt['tool'] != 'lookup_record':
                continue
            if (receipt.get('executed') is not True or receipt.get('read_only') is not True
                    or receipt not in world.receipts or not world.deliverable(receipt)):
                raise RuntimeError('Unverified or stale record cannot enter a context span')
            if any(not isinstance(receipt.get(k), str) or not receipt[k].strip()
                    for k in ('project', 'field', 'value')):
                raise RuntimeError('Incomplete actual record receipt')
            facts.append('The {field} for Project {project} is {value}.'.format(**receipt))
    if not facts:
        return content, 0
    # The receipt and the worker's narrative are independent outputs. Retain
    # both; this supplies context, never forces a spoken token or response start.
    return 'Verified tool results:\n' + '\n'.join(facts) + '\nBackend response:\n' + content, len(facts)
