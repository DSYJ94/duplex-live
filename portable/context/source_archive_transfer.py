"""Python3.6-compatible source-side packing and payload-bound signed PUTs.

No reusable account key is accepted. Request-specific authorization headers arrive
through SSH stdin, remain in memory, and are never logged or saved.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.parse
import urllib.request


def safe(name):
    p = PurePosixPath(name)
    forbidden = {'keys','credentials','owner-conversations','session-token',
                 '.aws','.ssh','.env','.hf_token','.typesafe_key'}
    assert p.parts and not p.is_absolute() and '..' not in p.parts
    assert not any(x in forbidden or x.startswith('core-') for x in p.parts)
    assert p.name != 'core' and '\n' not in name and '\r' not in name
    return name


def pack(plan):
    root = Path('/projects/u6si/duplex-live')
    out = root/'runs/runpod-migration-20261021/s3-archive-source-v1'
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in plan['archives']:
        name = safe(spec['name'])
        assert '/' not in name and name.startswith('native-v3-') and name.endswith('.tar')
        target = out/name
        if target.exists():
            raise RuntimeError('Archive already exists; inspect before restarting')
        try:
            h = hashlib.sha256()
            with target.with_suffix('.partial').open('xb') as writer:
                for row in spec['members']:
                    header = base64.b64decode(row['header'])
                    writer.write(header); h.update(header)
                    if row['bytes']:
                        source = root/safe(row['path'])
                        assert source.is_file() and source.stat().st_size == row['bytes']
                        member_hash = hashlib.sha256()
                        with source.open('rb') as reader:
                            for block in iter(lambda: reader.read(8*1024*1024), b''):
                                writer.write(block);h.update(block);member_hash.update(block)
                        assert member_hash.hexdigest() == row['sha256']
                tail = base64.b64decode(spec['tail'])
                writer.write(tail);h.update(tail)
            assert h.hexdigest() == spec['sha256']
            assert target.with_suffix('.partial').stat().st_size == spec['bytes']
            target.with_suffix('.partial').replace(target)
            result = {'name':name,'passed':True,'sha256':h.hexdigest(),'bytes':spec['bytes']}
        except (OSError, AssertionError) as e:
            result = {'name':name,'passed':False,'error_type':type(e).__name__}
        results.append(result)
        print(json.dumps(result),flush=True)
    (out/'packing-receipt.json').write_text(json.dumps(results,indent=2)+'\n')


def upload(plan):
    name = safe(plan['name']);assert '/' not in name
    source = Path('/projects/u6si/duplex-live/runs/runpod-migration-20261021/s3-archive-source-v1')/name
    assert source.is_file() and source.stat().st_size == plan['bytes']
    bucket = plan['bucket'];assert bucket == '49yytzyoge'
    def one(part):
        parsed = urllib.parse.urlparse(part['url'])
        assert parsed.scheme == 'https' and parsed.netloc == 's3api-eur-is-1.runpod.io'
        assert parsed.path == '/'+bucket+'/native-archive-v3/'+name
        with source.open('rb') as reader:
            reader.seek(part['offset']);data=reader.read(part['bytes'])
        assert len(data) == part['bytes']
        for attempt in range(5):
            try:
                assert part['headers']['X-Amz-Content-SHA256'] == hashlib.sha256(data).hexdigest()
                request=urllib.request.Request(part['url'],data=data,headers=part['headers'],method='PUT')
                with urllib.request.urlopen(request,timeout=50) as response:
                    assert response.status == 200
                    etag=response.headers['ETag']
                return {'PartNumber':part['number'],'ETag':etag}
            except Exception as e:
                if getattr(e,'code',500) < 500:
                    raise RuntimeError('Signed upload HTTP status '+str(e.code))
                if attempt == 4:
                    raise RuntimeError('Signed upload part failed: '+type(e).__name__+
                                       ' status '+str(getattr(e,'code',None)))
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(one,plan['parts']))
    print(json.dumps({'name':name,'parts':parts,'all_signed_parts_uploaded':True}),flush=True)


if __name__ == '__main__':
    command=sys.argv[1];plan=json.load(sys.stdin)
    if command == 'pack':pack(plan)
    elif command == 'upload':upload(plan)
    else:raise ValueError('Unknown operation')
