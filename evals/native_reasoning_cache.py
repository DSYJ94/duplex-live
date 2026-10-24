"""Complete32 Mimi/text cache with actual native requests and delivered contexts.

Reuses the pinned codec, vocabulary and interleaver. Retains every original PCM
frame and all40 observed requests/24 actual context deliveries. No ARC/provider
generation, forced answers, numerical approval or optimizer work occurs here.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from evals import native_reasoning_timing_convert as conversion
from evals import native_reasoning_timing_reapproval as timing
from evals import native_control_cache as native
from evals.native_reasoning_source import sha
from evals.recovery_encode import marker_frame,native_word_timing

SCHEMA='dl-native-reasoning32-codec-cache-v1'
FAMILIES={'direct-basic','delegate-reasoning','cancel-reasoning','replace-reasoning'}


def inventory(value):
    rows=value['accepted']
    if (value['schema']!=conversion.SCHEMA or value['complete32']is not True
            or value['training_approved']is not False or value['channel_order']!=['assistant','user']
            or [r['scenario_id']for r in rows]!=timing.mixed.IDS
            or Counter(r['family']for r in rows)!=Counter({k:8 for k in FAMILIES})
            or any(r['split']!='train'or r['scenario_class']!='native_reasoning4_teacher'
                or r['dir']!=r['scenario_id']or r['path']!=r['scenario_id']+'/conversation.wav'
                or type(r['native_frames'])is not int or r['native_frames']<1
                or r['samples']!=r['native_frames']*1920 for r in rows)):
        raise ValueError('Complete exact32 TRAIN inventory and all four families required')
    return rows


def controls(side,frames,family):
    markers=[marker_frame(e,12.5)for e in side['timeline']['events']if e['kind']=='marker']
    arrivals=side['reference_arrivals']
    if (markers!=[r['marker_ms']//80 for r in side['request_markers']]
            or len(set(markers))!=len(markers)or any(not 0<=f<frames for f in markers)
            or len(markers)!={'direct-basic':0,'delegate-reasoning':1,'cancel-reasoning':2,'replace-reasoning':2}[family]
            or len(arrivals)!=(0 if family=='direct-basic'else 1)
            or any(not 0<a['ready_at_s']<frames*.08 or not a['text'].strip()for a in arrivals)):
        raise ValueError('Preserve every actual native request and delivered context in the complete episode')
    return markers,arrivals


def text_lane(side,frames,family,tokenizer,builder):
    text=builder.prepare_item(side['alignments'],native.frame_duration(frames))
    if tuple(text.shape)!=(1,1,frames):raise ValueError('Exact integer interleaver frame coverage required')
    markers,arrivals=controls(side,frames,family)
    for frame in markers:
        if int(text[0,0,frame])not in (0,3):
            raise ValueError('Actual native request collides with independently measured spoken word')
        text[0,0,frame]=4
    if (text[0,0]==4).nonzero().flatten().tolist()!=markers:
        raise ValueError('No lost or spurious native request target permitted')
    full=232+frames+sum(2+len(tokenizer.encode(a['text']))for a in arrivals)
    if full!=side['native_frame_budget']['full_measured_prefix_live_and_span_frames']or full>1100:
        raise ValueError('Unchanged complete prefix/live/all-context geometry required')
    return text,markers,arrivals,native_word_timing(text[0,0].tolist(),side['alignments'],tokenizer.encode,12.5)


def checked_data(a):
    conversion.run(timing.load(a.inputs),a.version,a.semantic,a.data,verify_only=True)
    v=json.loads((a.data/'ingestion.json').read_text());return v,inventory(v)


def verify_files(cache,data,value,rows,tokenizer,builder):
    import torch
    if ([r['id']for r in value['examples']]!=[r['scenario_id']for r in rows]
            or {p.name for p in cache.iterdir()}!={'cache.json'}|{r['scenario_id']+'.pt'for r in rows}):
        raise ValueError('Complete32 exact cache inventory required; no passing subset')
    for entry,row in zip(value['examples'],rows):
        p=cache/entry['file'];folder=data/row['dir'];side=json.loads((folder/'conversation.json').read_text())
        if (p.is_symlink()or p.name!=row['scenario_id']+'.pt'or sha(p)!=entry['sha256']
                or row['artifacts']!={n:sha(folder/n)for n in conversion.converter.FILES}):
            raise ValueError('Original conversion or encoded cache bytes changed')
        payload=torch.load(p,map_location='cpu',weights_only=True);codes=payload['codes'];frames=row['native_frames']
        text,markers,arrivals,word_timing=text_lane(side,frames,row['family'],tokenizer,builder)
        if (set(payload)!={'codes','markers','arrivals'}or payload['markers']!=markers or payload['arrivals']!=arrivals
                or codes.dtype!=torch.int64 or tuple(codes.shape)!=(1,17,frames)
                or not torch.equal(codes[:,:1],text)or (codes[:,1:]<0).any()or(codes[:,1:]>=2048).any()
                or entry!=dict(id=row['scenario_id'],family=row['family'],split='train',file=p.name,
                    sha256=sha(p),frames=frames,markers=markers,delivered_contexts=len(arrivals),
                    source_artifacts=row['artifacts'],native_word_timing=word_timing)):
            raise ValueError('Exact text/PAD/request lane, all16 acoustic lanes and context ownership required')
    if (sum(len(r['markers'])for r in value['examples'])!=40
            or sum(r['delivered_contexts']for r in value['examples'])!=24):
        raise ValueError('All40 native requests and24 actual deliveries must survive')
    return value


def build(a):
    _,rows=checked_data(a)
    pins=native.source_pins(a.codec_source,a.interleaver_source,a.model)
    import soundfile as sf
    import torch
    _,tokenizer,builder=native.text_builder(a.codec_source,a.interleaver_source,a.model,'cpu')
    # Check ALL32 exact text/marker/word/context geometry before allocating Mimi.
    lanes=[]
    for row in rows:
        side=json.loads((a.data/row['dir']/'conversation.json').read_text())
        lanes.append(text_lane(side,row['native_frames'],row['family'],tokenizer,builder))
    mimi,_,_=native.runtime(a.codec_source,a.interleaver_source,a.model)
    a.out.mkdir(parents=True,exist_ok=False)
    v=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256=sha(Path(__file__)),conversion=str(a.data),ingestion_sha256=sha(a.data/'ingestion.json'),
        source_pins=pins,model=str(a.model),codec_source=str(a.codec_source),interleaver_source=str(a.interleaver_source),
        examples=[],complete32=False,frame_rate=12.5,sample_rate=24000,channel_order=['assistant','user'],
        assistant_codebooks=list(range(1,9)),user_codebooks=list(range(9,17)),
        source_words_audio_receipts_folds_changed=False,acoustic_codes_cropped_or_padded=False,
        native_engine_input_numerical_gradient_and_behavior_approval_required=True,
        optimizer_updates=0,provider_requests=0,training_approved=False,behavior_qualified=False)
    for row,(text,markers,arrivals,word_timing)in zip(rows,lanes):
        audio,rate=sf.read(a.data/row['path'],dtype='float32',always_2d=True)
        if rate!=24000 or audio.shape!=(row['samples'],2):
            raise ValueError('All original PCM frames/channels required')
        with torch.no_grad():
            acoustics=mimi.encode(torch.as_tensor(audio.T.copy(),dtype=torch.float32,device='cuda')[:,None])
            if tuple(acoustics.shape)!=(2,8,row['native_frames']):
                raise ValueError('Codec cannot crop or pad a source frame')
            codes=torch.cat([text,acoustics.reshape(1,16,row['native_frames']).cpu()],dim=1)
        name=row['scenario_id']+'.pt';torch.save(dict(codes=codes,markers=markers,arrivals=arrivals),a.out/name)
        v['examples'].append(dict(id=row['scenario_id'],family=row['family'],split='train',file=name,
            sha256=sha(a.out/name),frames=row['native_frames'],markers=markers,delivered_contexts=len(arrivals),
            source_artifacts=row['artifacts'],native_word_timing=word_timing))
        (a.out/'cache.json').write_text(json.dumps(v,indent=2)+'\n')
        print(json.dumps(dict(id=row['scenario_id'],complete=len(v['examples']),expected=32)),flush=True)
    v['complete32']=True;(a.out/'cache.json').write_text(json.dumps(v,indent=2)+'\n')
    verify_files(a.out,a.data,v,rows,tokenizer,builder)
    return v


def verify(a):
    _,rows=checked_data(a);v=json.loads((a.out/'cache.json').read_text())
    pins=native.source_pins(a.codec_source,a.interleaver_source,a.model)
    if (v['schema']!=SCHEMA or v['complete32']is not True or v['code_sha256']!=sha(Path(__file__))
            or v['ingestion_sha256']!=sha(a.data/'ingestion.json')or v['source_pins']!=pins
            or v['optimizer_updates']!=0 or v['provider_requests']!=0
            or v['training_approved']is not False or v['behavior_qualified']is not False
            or v['source_words_audio_receipts_folds_changed']is not False
            or v['acoustic_codes_cropped_or_padded']is not False
            or v['native_engine_input_numerical_gradient_and_behavior_approval_required']is not True
            or v['channel_order']!=['assistant','user']or v['frame_rate']!=12.5 or v['sample_rate']!=24000
            or v['assistant_codebooks']!=list(range(1,9))or v['user_codebooks']!=list(range(9,17))):
        raise ValueError('Complete unchanged native cache provenance required')
    _,tokenizer,builder=native.text_builder(a.codec_source,a.interleaver_source,a.model,'cpu')
    return verify_files(a.out,a.data,v,rows,tokenizer,builder)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','version','semantic','data','model','codec-source','interleaver-source','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    v=(verify if a.verify_only else build)(a)
    print(json.dumps(dict(complete32=v['complete32'],examples=len(v['examples']),training_approved=False)))
