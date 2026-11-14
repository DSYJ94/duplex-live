"""Export full1072 qualified native stereo pairs; preserve original source hashes.

The only signal conversion is declared16k->24k polyphase resampling. There is
no gain reduction, trimming or loss of any user/assistant source duration.
"""
import argparse,hashlib,json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def admission(plan,quality,semantics,alignment,*,expected_plan_sha=None):
    if not plan.get('owner_approved') or len(plan['exact_semantic_records'])!=1072 or len(plan['exact_speech_payloads'])!=2144:raise ValueError('Exact owner-approved full1072 source required')
    if not quality.get('complete') or len(quality['rows'])!=2144 or not quality.get('all_passed'):raise ValueError('Complete2144 independent speech qualification required')
    if not expected_plan_sha or semantics.get('source_plan_sha256')!=expected_plan_sha:raise ValueError('Semantic verdicts must bind the exact current source plan bytes')
    if not semantics.get('complete1072') or len(semantics['rows'])!=1072 or not semantics.get('all_passed'):raise ValueError('Complete1072 independent semantic approval required')
    if not alignment.get('complete1072') or len(alignment['rows'])!=1072 or not alignment.get('all_authored_words_measured'):raise ValueError('Every assistant authored word needs actual measured alignment')
    payloads={r['id']:r for r in plan['exact_speech_payloads']};q={r['id']:r for r in quality['rows']};a={r['id']:r for r in alignment['rows']};s={r['id']:r for r in semantics['rows']}
    ids={r['id'] for r in plan['exact_semantic_records']}
    if len(q)!=2144 or set(q)!=set(payloads) or len(ids)!=1072 or set(a)!=ids or set(s)!=ids:raise ValueError('No omitted/duplicated source or cross-cohort substitution')
    if any(not x['correct'] or not x['explanation_consistent'] for x in s.values()):raise ValueError('A failing semantic row cannot be hidden by summary')
    for source in plan['exact_semantic_records']:
        ident=source['id'];ar=a[ident]
        if source['question']!=payloads[ident+'-user']['text'] or source['response']!=payloads[ident+'-assistant']['text']:raise ValueError('Semantic and speech target words must agree')
        if ar['text']!=source['response'] or ar['source_sha256']!=q[ident+'-assistant']['source_sha256'] or ar['fold']!=source['fold']:raise ValueError('Exact source/fold/assistant alignment ownership required')
        for role in ('user','assistant'):
            key=ident+'-'+role
            if q[key]['fold']!=source['fold'] or payloads[key]['fold']!=source['fold'] or q[key]['source_words']!=payloads[key]['text'] or not q[key]['word_pass'] or not q[key]['peak_pass']:raise ValueError('Unqualified source target')
    return q,a


def resample_native(samples,rate):
    import numpy as np
    from scipy.signal import resample_poly
    if rate!=16000 or samples.ndim!=1 or samples.dtype!=np.float32 or not len(samples) or not np.isfinite(samples).all() or np.abs(samples).max()>.995:raise ValueError('Finite original16k source within unchanged peak gate required')
    out=resample_poly(samples,3,2).astype(np.float32,copy=False)
    if len(out)!=(len(samples)*3+1)//2 or not np.isfinite(out).all() or np.abs(out).max()>.995:raise ValueError('Derived24k geometry/peak failed; never attenuate to conceal it')
    return out


def main():
    p=argparse.ArgumentParser()
    for n in ('plan','quality','semantics','alignment','audio','out'):p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();plan,quality,semantics,alignment=[json.loads(x.read_text()) for x in (args.plan,args.quality,args.semantics,args.alignment)];q,aligned=admission(plan,quality,semantics,alignment,expected_plan_sha=sha(args.plan))
    import numpy as np
    import soundfile as sf
    from .source_export import pack
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'stereo').mkdir();records=[]
    for source in plan['exact_semantic_records']:
        ident=source['id'];waves=[];parents=[]
        for role in ('user','assistant'):
            key=ident+'-'+role;path=args.audio/(key+'.wav')
            if sha(path)!=q[key]['source_sha256']:raise ValueError('Original source waveform changed')
            raw,rate=sf.read(path,dtype='float32');waves.append(resample_native(raw,rate));parents.append(dict(id=key,sha256=q[key]['source_sha256'],original_frames=len(raw),original_rate=rate,resampled_frames=len(waves[-1])))
        audio,labels,start=pack(*waves,aligned[ident]['measured_words']);path=args.out/'stereo'/(ident+'.wav');sf.write(path,audio,24000,subtype='FLOAT')
        restored,rate=sf.read(path,dtype='float32')
        if rate!=24000 or not np.array_equal(audio,restored):raise ValueError('Complete derived channels must survive native export exactly')
        side=path.with_suffix('.json');side.write_text(json.dumps(dict(alignments=labels,fold=source['fold']))+'\n')
        records.append(dict(id=ident,fold=source['fold'],path=str(path.relative_to(args.out)),wav_sha256=sha(path),sidecar_sha256=sha(side),frames=len(audio)//1920,parents=parents,assistant_start_seconds=start/24000,native_training_approved=False))
    for fold in ('TRAIN','DEV'):
        (args.out/(fold.lower()+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in records if r['fold']==fold))
    result=dict(schema='expanded-native1072-source-export-v1',rows=records,complete1072=len(records)==1072,TRAIN=sum(r['fold']=='TRAIN' for r in records),DEV=sum(r['fold']=='DEV' for r in records),source_plan_sha256=sha(args.plan),quality_sha256=sha(args.quality),semantic_sha256=sha(args.semantics),alignment_sha256=sha(args.alignment),assistant_channel=0,user_channel=1,source_resampling='scipy.signal.resample_poly;3/2;full16kto24k;no gain',all_original_hashes_preserved=True,source_trimming=False,optimizer_updates=0,native_training_approved=False,native_engine_capture_and_independent_restore_pending=True)
    if (result['TRAIN'],result['DEV'])!=(1008,64):raise ValueError('Original complete folds required')
    (args.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
