"""Stage immutable public reviewer weights on Isambard, without GPU or tokens."""
import argparse
import hashlib
import json
from pathlib import Path

PIN=Path('docs/NATIVE-QWEN32-REVIEWER-PUBLIC-FILES-2026-11-07.json')
REPO='Qwen/Qwen3-32B'
REVISION='9216db5781bf21249d130ec9da846c4624c16137'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def check_pin(pin):
    if pin['repo']!=REPO or pin['revision']!=REVISION or pin['license']!='apache-2.0':
        raise ValueError('Exact public reviewer and license required')
    names=[r['path'] for r in pin['files']]
    if len(set(names))!=len(names) or not names or any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
        raise ValueError('Bounded distinct public model paths required')
    if sum(r['bytes'] for r in pin['files'])!=pin['total_bytes'] or pin['total_bytes']>70_000_000_000:
        raise ValueError('Public download exceeds pinned size')
    if any(not r['sha256'] for r in pin['files'] if r['path'].endswith('.safetensors')):
        raise ValueError('Every public weight shard needs an upstream SHA256')
    return pin


def verify_ready(directory,receipt):
    """Download already hashed all bytes; loading checks exact paths and sizes."""
    directory=Path(directory);pin=check_pin(json.loads(PIN.read_text()))
    r=json.loads(Path(receipt).read_text())
    if (r['pin_sha256']!=sha(PIN) or r['revision']!=REVISION or not r['all_hashes']
            or r['download_directory']!=str(directory) or len(r['files'])!=len(pin['files'])):
        raise ValueError('Complete immutable download receipt required')
    actual={x['path']:x for x in r['files']}
    for expected in pin['files']:
        row=actual[expected['path']];path=directory/row['path']
        if row['bytes']!=expected['bytes'] or path.stat().st_size!=expected['bytes']:
            raise ValueError('Public reviewer size differs')
        if expected['sha256'] and row['sha256']!=expected['sha256']:
            raise ValueError('Upstream public weight hash differs')
        if not expected['path'].endswith('.safetensors') and sha(path)!=row['sha256']:
            raise ValueError('Reviewer config/tokenizer changed')
    return r


def main(a):
    from huggingface_hub import snapshot_download
    pin=check_pin(json.loads(PIN.read_text()))
    if a.receipt.exists():raise ValueError('Preserve existing download receipts')
    directory=snapshot_download(repo_id=REPO,revision=REVISION,token=False,
        cache_dir=str(a.cache),allow_patterns=[r['path'] for r in pin['files']],max_workers=4)
    rows=[]
    for row in pin['files']:
        path=Path(directory)/row['path'];digest=sha(path)
        if path.stat().st_size!=row['bytes'] or row['sha256'] and digest!=row['sha256']:
            raise ValueError('Actual public model hash/size differs')
        rows.append(dict(path=row['path'],bytes=path.stat().st_size,sha256=digest))
    a.receipt.write_text(json.dumps(dict(repo=REPO,revision=REVISION,pin_sha256=sha(PIN),
        download_directory=directory,files=rows,all_hashes=True,public_weights_only=True,
        authenticated=False,GPU_used=False,external_model_API_calls=0),indent=2)+'\n')
    print(json.dumps(dict(revision=REVISION,files=len(rows),bytes=sum(r['bytes'] for r in rows),all_hashes=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    main(p.parse_args())
