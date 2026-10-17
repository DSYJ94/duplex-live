"""Pin the released BayLing baseline and codecs for an isolated offline audition."""
import argparse
import json
from pathlib import Path
import subprocess
from evals.moshi_model_download import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vendor',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    from huggingface_hub import snapshot_download
    pins_file=Path(__file__).with_name('assets')/'bayling-audition-pins.json'
    pins=json.loads(pins_file.read_text())
    args.out.mkdir(exist_ok=False)
    if not args.vendor.exists():
        subprocess.run(['git','clone',pins['source_repo'],str(args.vendor)],check=True)
        subprocess.run(['git','-C',str(args.vendor),'checkout','--detach',pins['source_commit']],check=True)
    if (subprocess.check_output(['git','-C',str(args.vendor),'rev-parse','HEAD'],text=True).strip()!=pins['source_commit']
            or subprocess.check_output(['git','-C',str(args.vendor),'status','--porcelain'],text=True)):
        raise RuntimeError('BayLing source must be clean and pinned')
    models=[]
    for item in pins['models']:
        path=Path(snapshot_download(item['repo'],revision=item['revision'],
            allow_patterns=list(item['files']),max_workers=2))
        files={}
        for name,expected in item['files'].items():
            file=path/name
            if not file.is_file() or file.stat().st_size!=expected['bytes']:
                raise RuntimeError('Missing or changed public asset: '+item['repo']+'/'+name)
            digest=sha256(file)
            if expected['sha256'] and digest!=expected['sha256']:
                raise RuntimeError('Asset differs from upstream LFS SHA256: '+name)
            files[name]={'sha256':digest,'bytes':file.stat().st_size}
        models.append({'repo':item['repo'],'revision':item['revision'],'path':str(path),'files':files})
    result={'schema':'dl-bayling-assets-v1','pins_sha256':sha256(pins_file),
        'source':str(args.vendor),'source_commit':pins['source_commit'],'models':models,
        'training_performed':False,'inference_tested':False,'offline_audition_only':True,
        'live_streaming_qualified':False,'delegation_qualified':False,'quality_qualified':False}
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
