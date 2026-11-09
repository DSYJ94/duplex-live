"""Fresh CPU input revalidation in the original clean Git verifier revision.

Run this file by absolute path in a separate process. The main trainer keeps
its current source; October receipts keep their exact original capture-code
pins. No monkeypatch, changed receipt, threshold, model output or GPU approval.
"""
import argparse,hashlib,inspect,json,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace

LEGACY_SOURCE='37579ab3da8ca5307164e9d308f8277bd386ef7f'
LEGACY_TEACHER_SHA='4a41a9890dd24b7a68d37bcc41140016a0ace4797ebe185ee7563208bee9a0f0'
BASE=Path('/projects/u6ada/duplex-live')
LEGACY=BASE/'source-original-input-verifier-v1'
OLD=Path('/projects/u6si/duplex-live');MIG=Path('/workspace/migration')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(a):
    current=Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','-C',str(current),'rev-parse','HEAD'],text=True).strip()!=a.source:
        raise ValueError('Exact current wrapper producer required')
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    if (subprocess.check_output(['git','-C',str(LEGACY),'rev-parse','HEAD'],text=True).strip()!=LEGACY_SOURCE
            or subprocess.check_output(['git','-C',str(LEGACY),'status','--porcelain','--untracked-files=no'],text=True).strip()
            or sha(LEGACY/'evals/native_control_teacher.py')!=LEGACY_TEACHER_SHA):
        raise ValueError('Exact clean original input-verifier source required')
    os.chdir(LEGACY);sys.path.insert(0,str(LEGACY))
    import torch
    from evals.context_native_audio49_data import digest
    from evals.native_control_engine import verify as control
    from evals.native_reasoning_engine import verify as reasoning
    if Path(inspect.getfile(control)).resolve()!=(LEGACY/'evals/native_control_engine.py').resolve():
        raise ValueError('Original verifier import isolation failed')
    assets=OLD/'runs/context-spanning-assets-7454205/manifest.json'
    spec=json.loads(assets.read_text());os.environ.update(CS_BASE_DIR=str(BASE/'public-models/personaplex-fdaf-v1'),
        CS_WEIGHTS_DIR=spec['path'],CS_VOICES_DIR=str(Path(spec['path'])/'voices'))
    sys.path.insert(0,spec['source'])
    if a.out.exists():raise ValueError('Preserve existing legacy-input artifact')
    started=time.time();groups={};records={}
    for name,cache,engine in [('control40',MIG/'native-control40-preflight-v2/codes',BASE/'legacy-control40-engine-v1'),
        ('reasoning32',MIG/'native-reasoning32-codes-v11',MIG/'native-reasoning32-engine-v11')]:
        if name=='control40':report=control(cache,engine)
        else:
            original=json.loads((cache/'cache.json').read_text())
            args=SimpleNamespace(inputs=MIG/'native-reasoning32-timing-v11-inputs-v1.json',
                version=MIG/'native-reasoning32-timing-v11-reapproval-v1',
                semantic=MIG/'native-reasoning32-timing-v11-semantic-v1/semantic.json',data=MIG/'native-reasoning32-converted-v11',
                model=Path(original['model']),codec_source=Path(original['codec_source']),
                interleaver_source=Path(original['interleaver_source']),cache=cache,assets=assets,
                proof=OLD/'runs/context-training-proof-7510331',out=engine)
            ingestion=json.loads((args.data/'ingestion.json').read_text())
            if not args.semantic.is_file() or sha(args.semantic)!=ingestion['semantic_approval_sha256']:
                raise ValueError('Exact original PASSED v11 semantic file required; never the failed v10 ancestor')
            report=reasoning(args)
        rows=[]
        for row in report['cases']:
            path=engine/row['id']/row['file'];saved=torch.load(path,weights_only=True,map_location='cpu')
            rows.append(dict(id=row['id'],kind=row['family'],fold='TRAIN',inputs=saved['inputs'],
                mask=saved['mask'],targets=saved['targets'],codes=saved['codes'][None],
                audio_mask=(saved['audio_mask'] if name=='reasoning32' else saved['mask'])[None],
                calls=row['calls'],source_sha256=sha(path)))
        if len(rows)!=(40 if name=='control40' else 32):raise ValueError('Complete original input cohort required')
        groups[name]=rows
        records[name]=dict(engine=str(engine),engine_report_sha256=sha(engine/'engine-inputs.json'),
            cache=str(cache),cache_report_sha256=sha(cache/'cache.json'),complete=True,
            tensor_fingerprints=[dict(id=r['id'],tensors={k:digest(v) for k,v in r.items() if isinstance(v,torch.Tensor)}) for r in rows])
        print(json.dumps(dict(complete=name,cases=len(rows),GPU_used=False)),flush=True)
    a.out.mkdir(parents=True,exist_ok=False)
    path=a.out/'inputs.pt';torch.save(groups,path)
    result=dict(schema='dl-moshi-original-input-verifier-v1',source=a.source,legacy_source=LEGACY_SOURCE,
        legacy_teacher_sha256=LEGACY_TEACHER_SHA,wrapper_sha256=sha(Path(__file__)),pid=os.getpid(),
        actual_CPU_only=True,complete40_and32=True,passed=True,cases=records,inputs_sha256=sha(path),
        old_thresholds_audio_words_receipts_and_folds_unchanged=True,optimizer_updates=0,
        native_gradient_restore_or_behavior_approved=False,seconds=time.time()-started)
    (a.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True)
    p.add_argument('--out',type=Path,required=True);main(p.parse_args())
