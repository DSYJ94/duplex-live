"""Export ALL48 pre-existing ordinary TRAIN scenes, without synthesizing audio.

This is source-format preparation, not native training approval. Original96,
all folds, source bytes, failed audits and existing histories stay untouched.
"""
import argparse,hashlib,json,math,subprocess
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def pack(user,assistant,alignments,rate=24000):
    if rate!=24000:raise ValueError('Original24kHz sources required; no implicit resampling')
    for a in (user,assistant):
        if a.ndim!=1 or a.dtype!=np.float32 or not len(a) or not np.isfinite(a).all() or float(np.abs(a).max())>.995:
            raise ValueError('Finite original mono float32 within unchanged peak gate required')
    # A declared160ms gap after the COMPLETE original user recording. There is
    # no VAD guess, early teacher reply, trimming, gain change or per-word splice.
    onset_frame=math.ceil((len(user)/rate+.16)/.08)
    start=round(onset_frame*.08*rate)
    length=round(math.ceil((start+len(assistant)+.48*rate)/(.08*rate))*.08*rate)
    audio=np.zeros((length,2),dtype=np.float32)
    audio[:len(user),1]=user;audio[start:start+len(assistant),0]=assistant
    shifted=[]
    for word in alignments:
        text=word['text'];lo,hi=word['start'],word['end']
        if not isinstance(text,str) or not math.isfinite(lo+hi) or not 0<=lo<=hi<=len(assistant)/rate:
            raise ValueError('Original measured alignment must fit source')
        if text.strip():shifted.append([text,[start/rate+lo,start/rate+hi],'SPEAKER_MAIN'])
    if not shifted or any(b[1][0]<a[1][0] for a,b in zip(shifted,shifted[1:])):
        raise ValueError('Complete monotonic measured assistant labels required')
    return audio,shifted,start


def main(a):
    import soundfile as sf
    from data.venus.retention_quality import inventory
    work=Path.cwd()
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source:raise ValueError('Exact committed producer required')
    q=a.sources/'teacher-inputs-v1/quality.json';alignment=a.sources/'teacher-inputs-v1/alignment/report.json'
    qr=json.loads(q.read_text());ar=json.loads(alignment.read_text())
    qpin=json.loads((work/'docs/NATIVE-RETENTION96-DERIVED182-REVIEW-2026-11-07.json').read_text())['derived_report_sha256']
    apin=json.loads((work/'docs/NATIVE-RETENTION96-ALIGNMENT73-COMPLETE-2026-11-07.json').read_text())['actual_report_sha256']
    if sha(q)!=qpin or sha(alignment)!=apin or not qr['all_sources_pass'] or qr['passed']!=182 or len(ar['rows'])!=73 or not ar['all73_exact_authored_word_alignments']:
        raise ValueError('Full original182 source and73 alignment approvals required')
    payload=work/'docs/NATIVE-RETENTION96-SPEECH-PAYLOAD-2026-11-07.json'
    sources=inventory(a.sources/'original',a.sources/'resumed',payload)
    users={r['text']:r for r in sources if r['profile']=='user-british'}
    assistants={r['text']:r for r in sources if r['profile']=='assistant-british'}
    align={r['id']:r for r in ar['rows']}
    plan=work/'evals/fixtures/venus-retention-request-source-proposal-v2-20261107.json'
    scenes=json.loads(plan.read_text())['scenes']
    selected=[s for s in scenes if s['family']=='ordinary']
    if len(scenes)!=96 or len(selected)!=48 or any(s['fold']!='TRAIN' or len(s['user_turns'])!=1 for s in selected):
        raise ValueError('Exactly ALL48 original ordinary TRAIN scenes, selected before rendering')
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'stereo').mkdir();records=[]
    for scene in selected:
        u=users[scene['user_turns'][0]];b=assistants[scene['teacher_program']['answer']];aligned=align[b['id']]
        if aligned['source_sha256']!=b['source_sha256'] or aligned['text']!=b['text']:raise ValueError('Original word/audio binding changed')
        for s in (u,b):
            if sha(s['wav'])!=s['source_sha256']:raise ValueError('Original source bytes changed')
        user,sr=sf.read(u['wav'],dtype='float32');assistant,br=sf.read(b['wav'],dtype='float32')
        if sr!=br:raise ValueError('Original channel sample rates differ')
        audio,labels,onset=pack(user,assistant,aligned['measured_words'],sr)
        path=a.output/'stereo'/(scene['id']+'.wav');sf.write(path,audio,sr,subtype='FLOAT')
        restored,rr=sf.read(path,dtype='float32')
        if rr!=sr or not np.array_equal(audio,restored) or not np.array_equal(restored[:len(user),1],user) or not np.array_equal(restored[onset:onset+len(assistant),0],assistant):raise ValueError('Original channel samples must survive export exactly')
        sidecar=path.with_suffix('.json');sidecar.write_text(json.dumps(dict(alignments=labels),indent=2)+'\n')
        records.append(dict(id=scene['id'],fold='TRAIN',path=str(path.relative_to(a.output)),duration=len(audio)/sr,
            user_source_sha256=u['source_sha256'],assistant_source_sha256=b['source_sha256'],alignment_source_sha256=sha(alignment),
            wav_sha256=sha(path),labels_sha256=sha(sidecar),assistant_start_seconds=onset/sr,raw_peak=float(np.abs(audio).max()),original_channel_samples_exact=True))
    (a.output/'train.jsonl').write_text(''.join(json.dumps(dict(path=r['path'],duration=r['duration']))+'\n' for r in records))
    manifest=dict(schema='dl-personaplex-ordinary48-source-export-v1',source=a.source,source_plan_sha256=sha(plan),
        source_quality_sha256=sha(q),alignment_sha256=sha(alignment),complete_all48=len(records)==48,TRAIN=48,DEV=0,HOLDOUT=0,
        original96_and_folds_unchanged=True,all182_original_sources_verified=True,assistant_channel=0,user_channel=1,
        source_resampling=False,gain_change=False,source_trimming=False,external_API_calls=0,optimizer_updates=0,
        prior25_and24_teacher_histories_and28_components_not_replaced=True,training_approved=False,
        native_tokenization_loss_masks_gradient_and_restore_pending=True,rows=records)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source',required=True);main(p.parse_args())
