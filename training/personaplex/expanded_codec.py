"""Complete native1072 Mimi/text cache, with unchanged TRAIN and DEV ownership.

Shards are distinct writers. No old cache, source waveform or fold is modified.
This codec stage is not native Engine capture or training admission.
"""
import argparse,json,os,subprocess
from pathlib import Path
from evals.moshi_model_download import sha256 as sha


def inputs(data):
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows']
    if not manifest['complete1072'] or manifest['TRAIN']!=1008 or manifest['DEV']!=64 or len(rows)!=1072 or len({r['id'] for r in rows})!=1072:raise ValueError('Complete original1072 folds required')
    if manifest['optimizer_updates']!=0 or manifest['native_training_approved'] is not False or manifest['source_trimming']:raise ValueError('Untrained unchanged source conversion required')
    if sum(r['fold']=='TRAIN' for r in rows)!=1008 or sum(r['fold']=='DEV' for r in rows)!=64 or len({r['path'] for r in rows})!=1072:raise ValueError('No DEV relabeling, repeated waveform path or omitted sources')
    for row in rows:
        path=data/row['path'];side=path.with_suffix('.json')
        if not path.resolve().is_relative_to(data.resolve()) or sha(path)!=row['wav_sha256'] or sha(side)!=row['sidecar_sha256'] or json.loads(side.read_text())['fold']!=row['fold']:raise ValueError('Source PCM/sidecar/fold changed')
    if sum(r['fold']=='TRAIN' for r in rows)!=1008 or sum(r['fold']=='DEV' for r in rows)!=64:raise ValueError('No DEV relabeling or omitted sources')
    return manifest,rows


def encode(audio,side,mimi,builder,tokenizer):
    import numpy as np
    import torch
    from evals.native_control_cache import check_codes,frame_duration
    if audio.ndim!=2 or audio.shape[1]!=2 or not len(audio) or len(audio)%1920 or not np.isfinite(audio).all() or np.abs(audio).max()>.995 or side['fold'] not in ('TRAIN','DEV'):raise ValueError('Whole unclipped native channels and explicit fold required')
    frames=len(audio)//1920
    with torch.no_grad():
        acoustics=mimi.encode(torch.as_tensor(audio.T.copy(),dtype=torch.float32,device='cuda')[:,None])
        if acoustics.shape!=(2,8,frames):raise ValueError('Every original frame and all16 acoustic heads required')
        text=builder.prepare_item(side['alignments'],frame_duration(frames))
        if text.shape!=(1,1,frames):raise ValueError('Whole native word timeline required')
        codes=torch.cat([text,acoustics.reshape(1,16,frames)],dim=1).cpu()
    return codes,check_codes(codes,frames,side,tokenizer)


def build(a):
    import soundfile as sf
    import torch
    from evals.native_control_cache import runtime,source_pins
    from evals.moshi_training_smoke import tensor_hashes
    from .expanded_guard import check as guarded
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source:raise ValueError('Exact committed producer required')
    if not 0<=a.shard<a.shards<=16:raise ValueError('Explicit bounded distinct codec shards required')
    guarded(a.lease,120)
    _,rows=inputs(a.data);pins=source_pins(a.codec_source,a.interleaver_source,a.model)
    out=a.root/('shard-%02d'%a.shard);out.mkdir(parents=True,exist_ok=False)
    mimi,tokenizer,builder=runtime(a.codec_source,a.interleaver_source,a.model);before=tensor_hashes(mimi.named_parameters())
    report=dict(source=a.source,code_sha256=sha(Path(__file__)),source_manifest_sha256=sha(a.data/'manifest.json'),source_pins=pins,shard=a.shard,shards=a.shards,pid=os.getpid(),actual_CUDA=True,gpu=torch.cuda.get_device_name(),rows=[],complete_shard=False,optimizer_updates=0,native_training_approved=False)
    for row in rows[a.shard::a.shards]:
        guarded(a.lease,60)
        path=a.data/row['path'];audio,rate=sf.read(path,dtype='float32',always_2d=True)
        if rate!=24000 or len(audio)!=row['frames']*1920:raise ValueError('Complete original24k native source required')
        side=json.loads(path.with_suffix('.json').read_text());codes,timing=encode(audio,side,mimi,builder,tokenizer)
        name=row['id']+'.pt';torch.save(dict(codes=codes,source_record=row),out/name)
        report['rows'].append(dict(id=row['id'],fold=row['fold'],file=name,sha256=sha(out/name),frames=row['frames'],native_word_timing=timing))
        (out/'cache.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(shard=a.shard,completed=len(report['rows']),required=len(rows[a.shard::a.shards]))),flush=True)
    if before!=tensor_hashes(mimi.named_parameters()):raise ValueError('Codec parameters changed')
    report.update(complete_shard=True,all_codec_parameters_unchanged=True)
    (out/'cache.json').write_text(json.dumps(report,indent=2)+'\n')


def verify(a):
    import torch
    from evals.native_control_cache import text_builder,source_pins,check_codes
    _,rows=inputs(a.data);pins=source_pins(a.codec_source,a.interleaver_source,a.model);_,tokenizer,builder=text_builder(a.codec_source,a.interleaver_source,a.model,'cpu');entries={};pids=set()
    for rank in range(a.shards):
        out=a.root/('shard-%02d'%rank);report=json.loads((out/'cache.json').read_text())
        if not report['complete_shard'] or report['shards']!=a.shards or report['shard']!=rank or report['source']!=a.source or report['code_sha256']!=sha(Path(__file__)) or report['source_manifest_sha256']!=sha(a.data/'manifest.json') or report['source_pins']!=pins or report['optimizer_updates']!=0 or not report['actual_CUDA'] or not report['all_codec_parameters_unchanged'] or report['pid']==os.getpid():raise ValueError('Actual complete separate-process pinned shard required')
        if [r['id'] for r in report['rows']]!=[r['id'] for r in rows[rank::a.shards]]:raise ValueError('Whole deterministic shard coverage required')
        pids.add(report['pid'])
        for entry,row in zip(report['rows'],rows[rank::a.shards]):
            path=out/entry['file']
            if path.name!=row['id']+'.pt' or sha(path)!=entry['sha256'] or entry['fold']!=row['fold']:raise ValueError('Native shard artifact/fold changed')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['source_record']!=row:raise ValueError('Exact original source ownership required')
            side=json.loads((a.data/row['path']).with_suffix('.json').read_text());timing=check_codes(saved['codes'],row['frames'],side,tokenizer)
            if timing!=entry['native_word_timing']:raise ValueError('Fresh complete native word timing differs')
            entries[row['id']]=dict(entry,shard=rank)
    if len(entries)!=1072:raise ValueError('All1072 native examples required')
    result=dict(source=a.source,code_sha256=sha(Path(__file__)),source_manifest_sha256=sha(a.data/'manifest.json'),complete1072=True,TRAIN=1008,DEV=64,rows=[entries[r['id']]for r in rows],actual_independent_CPU_verifier_pid=os.getpid(),capture_pids=sorted(pids),optimizer_updates=0,native_training_approved=False,native_engine_capture_and_restore_still_required=True)
    with (a.root/'cache.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','verify'])
    for n in ('data','root','model','codec-source','interleaver-source','lease'):p.add_argument('--'+n,type=Path,required=n!='lease')
    p.add_argument('--source',required=True);p.add_argument('--shards',type=int,required=True);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    (build if a.stage=='build' else verify)(a)


if __name__=='__main__':main()
