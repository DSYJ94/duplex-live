"""Verify the existing public key's real lifetime; never read a private key."""
import json
from pathlib import Path
import time

PUBLIC_KEY=Path('runs/local-migration-20261024/.private/gcp-native-ssh.pub')


def ensure(stop_unix,receipts,*,force=False):
    from . import controller as c
    from .sustained import authority
    authority()
    expected=PUBLIC_KEY.read_text().split()[:2]
    if len(expected)!=2:raise ValueError('Existing approved public SSH key required')
    def read():
        rows=c.cloud('compute','os-login','ssh-keys','list')
        found=[]
        for row in rows:
            value=row.get('value',row)
            if str(value.get('key','')).split()[:2]==expected:found.append(value)
        if len(found)>1:raise ValueError('Ambiguous public-key registration')
        return found[0] if found else None
    value=read();changed=force or not value or int(value.get('expirationTimeUsec',0))/1e6<stop_unix+1800
    if changed:
        c.cloud('compute','os-login','ssh-keys','update' if value else 'add','--key-file='+str(PUBLIC_KEY),'--ttl=24h')
        value=read()
    if not value or int(value.get('expirationTimeUsec',0))/1e6<stop_unix+1800:
        raise ValueError('Read-back key expiry cannot cover the approved run and recovery margin')
    result=dict(checked_unix=time.time(),expiration_unix=int(value['expirationTimeUsec'])/1e6,
                fingerprint=value.get('fingerprint'),registration_changed=changed,private_key_read=False)
    with (Path(receipts)/'ssh-key-lifetime.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result
