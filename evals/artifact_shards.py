"""Checksummed pieces for resumable model transfers; atomic verified restore."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from evals.moshi_model_download import sha256


def split(source, out, expected, chunk_bytes=512*1024*1024):
    if not re.fullmatch('[a-f0-9]{64}',expected) or type(chunk_bytes) is not int or chunk_bytes<1:
        raise ValueError('Expected fixed SHA256 and positive chunk size')
    out.mkdir(parents=True,exist_ok=True)
    whole=hashlib.sha256(); pieces=[]
    with source.open('rb') as reader:
        while True:
            block=reader.read(min(chunk_bytes,1024*1024))
            if not block:break
            name=f'chunk-{len(pieces):04d}.bin'; temporary=out/(name+'.tmp')
            digest=hashlib.sha256(); count=0
            with temporary.open('wb') as writer:
                while block:
                    writer.write(block);digest.update(block);whole.update(block);count+=len(block)
                    if count==chunk_bytes:break
                    block=reader.read(min(chunk_bytes-count,1024*1024))
            temporary.replace(out/name)
            pieces.append({'file':name,'bytes':count,'sha256':digest.hexdigest()})
    if not pieces or whole.hexdigest()!=expected:
        raise RuntimeError('Source differs from the externally pinned model hash')
    manifest={'schema':'dl-artifact-shards-v1','source_filename':source.name,
        'bytes':sum(p['bytes'] for p in pieces),'sha256':expected,
        'chunk_bytes':chunk_bytes,'pieces':pieces}
    temporary=out/'manifest.json.tmp';temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(out/'manifest.json')
    return manifest


def restore(manifest_path, target, expected):
    manifest=json.loads(manifest_path.read_text()); pieces=manifest['pieces']
    if (manifest['schema']!='dl-artifact-shards-v1'
            or not re.fullmatch('[a-f0-9]{64}',expected) or manifest['sha256']!=expected
            or not pieces or type(manifest['bytes']) is not int
            or type(manifest['chunk_bytes']) is not int or manifest['chunk_bytes']<1
            or sum(p['bytes'] for p in pieces)!=manifest['bytes']):
        raise RuntimeError('Invalid or unpinned shard manifest')
    for i,p in enumerate(pieces):
        if (p['file']!=f'chunk-{i:04d}.bin' or not re.fullmatch('[a-f0-9]{64}',p['sha256'])
                or type(p['bytes']) is not int or not 0<p['bytes']<=manifest['chunk_bytes']
                or (i<len(pieces)-1 and p['bytes']!=manifest['chunk_bytes'])):
            raise RuntimeError('Invalid shard identity, order or size')
    if target.exists():
        if sha256(target)!=expected:raise RuntimeError('Refusing to overwrite a different existing target')
        return manifest
    target.parent.mkdir(parents=True,exist_ok=True); temporary=target.with_name(target.name+'.assembling')
    whole=hashlib.sha256()
    with temporary.open('wb') as writer:
        for p in pieces:
            digest=hashlib.sha256();count=0
            with (manifest_path.parent/p['file']).open('rb') as reader:
                for block in iter(lambda:reader.read(1024*1024),b''):
                    digest.update(block);whole.update(block);writer.write(block);count+=len(block)
            if count!=p['bytes'] or digest.hexdigest()!=p['sha256']:
                raise RuntimeError('Incomplete or corrupted transfer piece: '+p['file'])
    if temporary.stat().st_size!=manifest['bytes'] or whole.hexdigest()!=expected:
        raise RuntimeError('Reassembled model differs from pinned hash')
    temporary.replace(target)
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('split','restore'))
    ap.add_argument('--source',type=Path);ap.add_argument('--out',type=Path)
    ap.add_argument('--manifest',type=Path);ap.add_argument('--target',type=Path)
    ap.add_argument('--sha256',required=True)
    args=ap.parse_args()
    if args.mode=='split':
        if not args.source or not args.out:ap.error('Split requires source and output directory')
        result=split(args.source,args.out,args.sha256)
    else:
        if not args.manifest or not args.target:ap.error('Restore requires manifest and explicit target')
        result=restore(args.manifest,args.target,args.sha256)
    print(json.dumps({'mode':args.mode,'sha256':result['sha256'],
        'bytes':result['bytes'],'pieces':len(result['pieces']),'verified':True}))


if __name__=='__main__':main()
