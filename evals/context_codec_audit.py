"""Actual raw-PCM native streaming codec IDs versus approved offline cache IDs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ('assets','base','cache','out'): ap.add_argument('--'+n,type=Path,required=True)
    args = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from evals.context_spanning_download import SOURCE_COMMIT
    from evals.context_spanning_probe import audited_loads
    from evals.personaplex_train import MODEL_HASHES
    from evals.recovery_data import validate_cache, validate_conversion
    assets = json.loads(args.assets.read_text()); source = Path(assets['source'])
    if (assets['source_commit'] != SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() != SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)):
        raise RuntimeError('Changed native codec source')
    name = 'tokenizer-e351c8d8-checkpoint125.safetensors'
    if sha256(args.base/name) != MODEL_HASHES[name]: raise RuntimeError('Changed Mimi checkpoint')
    cache = validate_cache(args.cache); data = validate_conversion(Path(cache['conversion']))
    row = next(r for r in cache['examples'] if r['split']=='train' and r['kind']=='private')
    origin = next(r for r in data['accepted'] if r['scenario_id']==row['scenario_id'])
    audio, rate = sf.read(origin['path'],dtype='float32',always_2d=True)
    if rate!=24000 or audio.shape[1]!=2 or len(audio)%1920 or not np.isfinite(audio).all():
        raise RuntimeError('Expected complete approved assistant-left/user-right native frames')
    expected = torch.load(args.cache/row['file'],map_location='cpu',weights_only=True)['codes'][0]
    os.environ['CS_BASE_DIR']=str(args.base)
    sys.path.insert(0,str(source))
    from contextspan.model.weights import load_mimi
    args.out.mkdir(exist_ok=False)
    records=[]
    with audited_loads(records): mimi=load_mimi('cuda').eval().requires_grad_(False)
    with torch.no_grad():
        batch=mimi.encode(torch.tensor(audio.T.copy(),device='cuda')[:,None]).cpu()
        offline=mimi.encode(torch.tensor(audio[:,1].copy(),device='cuda')[None,None]).cpu()
        live=[]
        with mimi.streaming(1):
            for f in range(0,len(audio),1920):
                live.append(mimi.encode(torch.tensor(audio[f:f+1920,1].copy(),device='cuda')[None,None]).cpu())
        online=torch.cat(live,dim=2)
    side=json.loads(Path(origin['path']).with_suffix('.json').read_text())
    active=torch.zeros(expected.shape[1],dtype=torch.bool)
    for e in side['timeline']['events']:
        if e['ch']=='user' and e['kind']=='utterance':
            for f in range(len(active)):
                active[f] |= f*.08 < e['actual_end_ms']/1000 and (f+1)*.08 > e['actual_start_ms']/1000
    def compare(actual, target):
        if actual.shape!=target.shape: return {'shape_match':False,'actual':list(actual.shape),'expected':list(target.shape)}
        valid=target>=0; different=(actual!=target)&valid
        return {'shape_match':True,'exact':not bool(different.any()),
            'valid_per_head':valid.sum(1).tolist(),'mismatches_per_head':different.sum(1).tolist(),
            'speech_mismatches_per_head':different[:,active].sum(1).tolist(),
            'silence_mismatches_per_head':different[:,~active].sum(1).tolist(),
            'first_mismatches_head_frame':different.nonzero().tolist()[:24]}
    target=expected[9:17]
    shifts={}
    if online[0].shape==target.shape:
        for shift in range(-2,3):
            a=online[0,:,max(0,shift):online.shape[-1]+min(0,shift)]
            b=target[:,max(0,-shift):target.shape[-1]-max(0,shift)]
            valid=b>=0
            shifts[str(shift)]=(((a==b)&valid).sum(1)/valid.sum(1).clamp_min(1)).tolist()
    report={'schema':'dl-context-codec-stream-audit-v1','scenario':row['scenario_id'],'split':'train',
        'source_commit':SOURCE_COMMIT,'codec_sha256':MODEL_HASHES[name], 'cache_sha256':sha256(args.cache/'cache.json'),
        'wav_sha256':origin['wav_sha256'],'loads':records,'rate':rate,'input_samples':len(audio),
        'cudnn_enabled':torch.backends.cudnn.enabled,'torch':torch.__version__,'cuda':torch.version.cuda,
        'old_cache_vs_context_batch2_user':compare(batch[1],target),
        'old_cache_vs_context_batch2_assistant':compare(batch[0],expected[1:9]),
        'old_cache_vs_context_offline_user':compare(offline[0],target),
        'old_cache_vs_context_streaming_user':compare(online[0],target),
        'context_offline_vs_streaming_user':compare(online[0],offline[0]),
        'diagnostic_shift_agreement_per_head':shifts,'targets_shifted':False,'optimizer_steps':0}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
