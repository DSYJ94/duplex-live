"""Fetch only the three pinned public Context Spanning assets, without credentials."""
import argparse,hashlib,json,os,urllib.request
from pathlib import Path
from evals.context_spanning_download import SOURCE_COMMIT,MODEL_REVISION,WEIGHT_SHA256,VOICE_SHA256
from data.personaplex.source_export import sha


def run(root):
    root=Path(root);manifest=json.loads((root/'original-manifest.json').read_text())
    expected={'context_spanning_7b.pt':(16743028866,WEIGHT_SHA256),'voices/main.pt':(3198,VOICE_SHA256),'README.md':(9091,'385380cf10f4b9ae06ee56e49ffad126307d79c43d3552da3ed7b27cc3a0283c')}
    if manifest['source_commit']!=SOURCE_COMMIT or manifest['revision']!=MODEL_REVISION or {k:(v['bytes'],v['sha256']) for k,v in manifest['files'].items()}!=expected:raise ValueError('Exact original public manifest required')
    for name,(size,digest) in expected.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if p.is_file() and p.stat().st_size==size and sha(p)==digest:continue
        temp=p.with_name(p.name+'.public-download-v2')
        h=hashlib.sha256();written=0
        with urllib.request.urlopen('https://huggingface.co/mindlogicinc/context-spanning-7b/resolve/'+MODEL_REVISION+'/'+name,timeout=60) as response,temp.open('xb') as f:
            for chunk in iter(lambda:response.read(8*1024*1024),b''):
                f.write(chunk);h.update(chunk);written+=len(chunk)
                if written%(256*1024*1024)<len(chunk):print(json.dumps(dict(asset=name,bytes=written,total=size)),flush=True)
        if written!=size or h.hexdigest()!=digest:raise ValueError('Pinned public download hash differs: '+name)
        if p.exists():p.rename(p.with_name(p.name+'.rsync-partial-preserved-v1'))
        os.replace(temp,p)
    result=dict(files=3,bytes=sum(v[0] for v in expected.values()),all_hashes=True,method='direct unauthenticated pinned public HTTPS',credentials_used=False,optimizer_updates=0)
    (root/'direct-staging-complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);run(p.parse_args().root)
