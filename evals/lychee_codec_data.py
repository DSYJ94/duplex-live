"""Re-encode approved TRAIN speech into Lychee's native S3 units.

This creates codec round-trip diagnostics, not approved training examples. Each
source word interval is retained. Only independently checked sources are read;
validation and test audio are excluded. No synthetic word times are introduced.
"""
import argparse
import io
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def word_groups(alignments, duration):
    groups=[];current=[];prior=-1.
    for text,interval,speaker in alignments:
        if speaker!='SPEAKER_MAIN':continue
        start,end=interval
        if not (text.strip() and 0<=start<end<=duration+.001 and start>=prior-.001):
            raise ValueError('Invalid independent source word intervals')
        if current and start-current[-1][1][1]>=.6:
            groups.append(current);current=[]
        current.append((text.strip(),[start,end]));prior=end
    if current:groups.append(current)
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,nargs='+',required=True)
    for name in ('assets','voice','voice-provenance','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import s3tokenizer
    from scipy.signal import resample_poly
    from evals.recovery_data import validate_conversion
    from evals.lychee_live_assets import load
    rows=[];sources=[];seen=set()
    for data in args.data:
        receipt=validate_conversion(data)
        sources.append({'path':str(data),'ingestion_sha256':sha256(data/'ingestion.json')})
        for row in receipt['accepted']:
            if row['split']!='train':continue
            if row['scenario_id'] in seen:raise RuntimeError('Duplicate TRAIN scenario')
            seen.add(row['scenario_id']);rows.append(row)
    if args.limit<0:raise ValueError('Negative limit')
    if args.limit:rows=rows[:args.limit]
    if not rows:raise RuntimeError('No approved TRAIN speech')
    args.out.mkdir(exist_ok=False)
    vocoder,_,provenance=load(args.assets,args.voice,args.voice_provenance)
    accepted=[];rejected=[]
    manifest={'schema':'dl-lychee-s3-source-diagnostic-v1','sources':sources,
        'provenance':provenance,'source_train_scenarios':len(rows),'examples':accepted,'rejected':rejected,
        'independent_roundtrip_audit':'pending','training_approved':False,'native_interleaver_tested':False}
    torch.manual_seed(424242)
    for row in rows:
        sidecar=json.loads(Path(row['path']).with_suffix('.json').read_text())
        pcm,rate=sf.read(row['path'],dtype='float32',always_2d=True)
        if rate!=24000 or pcm.shape[1]!=2 or not np.isfinite(pcm).all():
            raise RuntimeError('Approved source is not finite assistant-left/user-right stereo')
        groups=word_groups(sidecar['alignments'],len(pcm)/rate)
        for index,words in enumerate(groups):
            # Slice by actual independently measured bounds, retaining silence.
            first=max(0,int(np.floor((words[0][1][0]-.12)*rate)))
            last=min(len(pcm),int(np.ceil((words[-1][1][1]+.16)*rate)))
            name=f"{row['scenario_id']}-speech-{index:03d}"
            if (last-first)/rate>28:
                rejected.append({'name':name,'reason':'Measured continuous segment exceeds 28s codec diagnostic bound'})
                continue
            folder=args.out/name;folder.mkdir()
            source=pcm[first:last,0];sf.write(folder/'source.wav',source,rate,subtype='FLOAT')
            text=' '.join(w[0] for w in words)
            audio=torch.from_numpy(resample_poly(source,2,3).astype(np.float32))
            with torch.inference_mode():
                mel=s3tokenizer.log_mel_spectrogram(audio)
                features,lengths=s3tokenizer.padding([mel])
                codes,code_lengths=vocoder.audio_tokenizer.quantize(features.cuda(),lengths.cuda())
                values=codes[0,:int(code_lengths[0])].long().tolist()
                if not values or any(not 0<=v<6561 for v in values):raise RuntimeError('Invalid native S3 units')
                blob=vocoder(values,str(args.voice))
            output,out_rate=sf.read(io.BytesIO(blob),dtype='float32')
            if out_rate!=24000 or output.ndim!=1 or not np.isfinite(output).all() or not np.any(np.abs(output)>.001):
                raise RuntimeError('Native codec round trip produced invalid or silent PCM')
            sf.write(folder/'model.wav',output,out_rate)
            torch.save({'s3_tokens':torch.tensor(values,dtype=torch.int64)},folder/'codes.pt')
            report={'schema':'dl-lychee-s3-roundtrip-v1','text':text,'source_scenario':row['scenario_id'],
                'source_split':'train','source_conversation_sha256':row['wav_sha256'],
                'source_sidecar_sha256':row['sidecar_sha256'],
                'source_audio_bounds_s':[first/rate,last/rate],
                'source_measured_words':[{'text':w,'start_s':a-first/rate,'end_s':b-first/rate} for w,(a,b) in words],
                'source_reference_arrivals':sidecar['reference_arrivals'],
                'source_kind':sidecar['timeline']['meta']['kind'],
                'codec_units':len(values),'native_codec_hz':25,
                'source_audio_sha256':sha256(folder/'source.wav'),'codes_sha256':sha256(folder/'codes.pt'),
                'audio_sha256':sha256(folder/'model.wav'),'training_approved':False,
                'independent_roundtrip_audit':'pending','microphone_acceptance':False}
            (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            accepted.append({'name':name,'path':str(folder),'report_sha256':sha256(folder/'report.json'),
                'source_scenario':row['scenario_id'],'source_split':'train'})
            (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps({'name':name,'units':len(values),'source_seconds':len(source)/rate}),flush=True)
    if not accepted:raise RuntimeError('No bounded native S3 diagnostics produced')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
