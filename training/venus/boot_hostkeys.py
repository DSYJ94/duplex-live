"""Retain authenticated host trust across the approved boot-disk move.

GPU guest attributes retain its old keys. Its boot disk now has keys published
during authorized CPU recovery. Never learn keys from an unverified handshake.
"""
import base64
import hashlib
import json
from pathlib import Path

ROOT=Path('runs/venus-large-capacity-20261104/results-recovery-v83')
PROOFS={
    'known-hosts':'bd2f4cb307e3b0dd8fa684c388f38f676db4a30995e45410b9e68d85ceb72dde',
    'provider-start.json':'3cb29f0cff386ed08f07847dd539ccf99d63c29230d7e31fa157d757666a199b',
    'original-binding-restored.json':'b7ce9dc6c8fe05b27a57cc498b16fb653a017bf446ac5bdd12d4a4bdabd948c6',
}
FINGERPRINT='SHA256:L18HjVDGmDWYQ2gIsxDr2JyfYvCMd44j7rvFquRRkDs'


def verified():
    for name,digest in PROOFS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Authenticated CPU host-key provenance changed: '+name)
    vm=json.loads((ROOT/'provider-start.json').read_text())
    restored=json.loads((ROOT/'original-binding-restored.json').read_text())
    boot=[x for x in vm['disks'] if x.get('boot')]
    if vm['id']!='2296018948858061861' or len(boot)!=1 or boot[0]['autoDelete'] or not boot[0]['source'].endswith('/disks/duplex-venus-large-data-20261104-eua-v67'):
        raise ValueError('Exact authorized CPU and same retained boot disk required')
    if restored!=dict(original_vm_id='7521667153519853425',disk_id='2297341858915206463',status='TERMINATED',auto_delete=False):
        raise ValueError('Exact preserved disk return receipt required')
    keys=[line.split()[1:3] for line in (ROOT/'known-hosts').read_text().splitlines() if len(line.split())==3 and line.split()[1]=='ssh-ed25519']
    if len(keys)!=1:raise ValueError('One previously authenticated ED25519 key required')
    algorithm,value=keys[0]
    fingerprint='SHA256:'+base64.b64encode(hashlib.sha256(base64.b64decode(value,validate=True)).digest()).decode().rstrip('=')
    if fingerprint!=FINGERPRINT:raise ValueError('Verified disk key fingerprint changed')
    return [dict(key=algorithm,value=value)]
