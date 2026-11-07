"""Compile all96 new teachers from original audio, measured words and real work.

No native cache or training approval is issued. Source recordings and failed
audits remain intact. Worker monotonic clocks and authored dialogue clocks are
explicitly distinct; teacher-forced requests are not student dispatch success.
"""
import argparse
import hashlib
import json
import wave
from pathlib import Path
from .retention_speech import prepare, PLAN, sha
from .retention_quality import inventory
from .paced_timeline import paced_scene
from .receipts import lookup_key


def inputs(original,resumed,payload,quality,alignment):
    sources=inventory(original,resumed,payload);q=json.loads(Path(quality).read_text());a=json.loads(Path(alignment).read_text())
    word_receipt=json.loads(Path('docs/NATIVE-RETENTION96-DERIVED182-REVIEW-2026-11-07.json').read_text())
    alignment_receipt=json.loads(Path('docs/NATIVE-RETENTION96-ALIGNMENT73-COMPLETE-2026-11-07.json').read_text())
    if sha(quality)!=word_receipt['derived_report_sha256'] or sha(alignment)!=alignment_receipt['actual_report_sha256']:
        raise ValueError('Exact preserved complete word/alignment reviews required')
    if (not q['complete'] or not q['all_sources_pass'] or q['passed']!=182 or len(q['rows'])!=182
            or not q['all_lookup_keys_exact'] or not a['complete']
            or not a['all73_exact_authored_word_alignments'] or len(a['rows'])!=73
            or a['independent_source_review_sha256']!=sha(quality)):
        raise ValueError('Complete182 word/peak/exact-key and73 measured alignment required')
    qr={r['id']:r for r in q['rows']};ar={r['id']:r for r in a['rows']};users={};assistant={}
    for s in sources:
        row=qr[s['id']]
        if not row['passed'] or row['source_sha256']!=s['source_sha256'] or row['text']!=s['text']:
            raise ValueError('Original recognized source differs')
        if s['profile']=='assistant-british':
            aligned=ar[s['id']]
            if aligned['text']!=s['text'] or aligned['source_sha256']!=s['source_sha256']:
                raise ValueError('Measured assistant source differs')
            assistant[s['text']]=dict(authored=s['text'],passed=True,word_error_rate=row['word_error_rate'],
                peak=s['original_peak'],source_sha256=s['source_sha256'],
                words=[dict(text=w['text'],timestamp=[w['start'],w['end']]) for w in aligned['measured_words']])
        else:
            users[s['text']]=dict(id=s['id'],sha256=s['source_sha256'],seconds=s['seconds'],wav=s['wav'])
    if len(users)!=109 or len(assistant)!=73:raise ValueError('All original user/assistant bindings required')
    return users,assistant


def verify_receipt(scene,receipt):
    p=scene['teacher_program'];events=receipt['events'];dispatch=[e for e in events if e['type']=='native_delegate']
    if (receipt['scene_id']!=scene['id'] or receipt['world']!=scene['synthetic_private_world']
            or not receipt['teacher_forced_request'] or receipt['native_student_dispatch_claimed']
            or not dispatch or dispatch[0]['query']!=p['query']
            or lookup_key(p['query'])!=next(iter(receipt['world']))):
        raise ValueError('Exact real synthetic worker request/scene required')
    ident=dispatch[0]['work_id'];done=[e for e in events if e['type']=='worker_completed']
    if scene['family']=='cancel':
        cancelled={e['work_id'] for e in events if e['type']=='worker_cancelled'}
        confirms=[e for e in events if e['type']=='cancellation_confirmed']
        if done or len(confirms)!=1 or confirms[0]['targets']!=[ident] or ident not in cancelled:
            raise ValueError('Cancel a genuinely pending matching task, never an empty target')
    elif len(done)!=1 or done[0]['work_id']!=ident or done[0]['result']!='Lookup '+lookup_key(p['query'])+': '+next(iter(receipt['world'].values())):
        raise ValueError('Actual matching result required')


def main(a):
    import numpy as np
    from scipy.signal import resample_poly
    from tokenizers import Tokenizer
    users,assistant=inputs(a.original,a.resumed,a.payload,a.quality,a.alignment)
    tokenizer=Tokenizer.from_file(str(a.tokenizer));plan=json.loads(PLAN.read_text())
    work_review=Path('docs/NATIVE-RETENTION96-WORKER-RECEIPTS-2026-11-07.json')
    work=json.loads(work_review.read_text());work_hashes={r['id']:r['receipt_sha256'] for r in work['rows']}
    if not work['complete48'] or len(work_hashes)!=48:raise ValueError('Complete actual48 worker review required')
    if len(plan['scenes'])!=96 or any(s['fold']!='TRAIN' for s in plan['scenes']):
        raise ValueError('Exact new96 TRAIN only; old folds stay unchanged')
    if a.output.exists():raise ValueError('Preserve prior teachers')
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for scene in plan['scenes']:
        path=a.receipts/(scene['id']+'.json');receipt=json.loads(path.read_text()) if scene['synthetic_private_world'] else None
        if receipt:
            if sha(path)!=work_hashes[scene['id']]:raise ValueError('Actual worker receipt changed')
            verify_receipt(scene,receipt)
        timeline=paced_scene(scene,users,receipt,assistant)
        for action in timeline['actions']:
            if action['kind']=='speak':
                if any(len(tokenizer.encode(u['text'],add_special_tokens=False).ids)>16 for u in action['native_speech_units']):
                    raise ValueError('Measured speech exceeds native16-token teacher-unit capacity')
        pcm=np.zeros(timeline['frames']*16000,dtype='<i2');last_end=0
        for turn in timeline['user_turns']:
            source=users[turn['text']]
            if sha(source['wav'])!=source['sha256']:raise ValueError('Original source audio changed')
            with wave.open(source['wav']) as stream:raw=stream.readframes(stream.getnframes())
            converted=np.rint(resample_poly(np.frombuffer(raw,dtype='<i2').astype(np.float64),2,3))
            if not np.isfinite(converted).all() or float(np.abs(converted).max())/32768>.995:
                raise ValueError('Source resampling clips; never attenuate or clamp it')
            values=converted.astype('<i2');start=round(turn['start_seconds']*16000);end=start+len(values)
            if start<last_end or end>len(pcm):raise ValueError('Unplanned user overlap or truncation')
            pcm[start:end]=values;last_end=end
        folder=a.output/scene['id'];folder.mkdir()
        with wave.open(str(folder/'user.wav'),'wb') as stream:
            stream.setparams((1,2,16000,0,'NONE','not compressed'));stream.writeframes(pcm.tobytes())
        timeline.update(user_wav_sha256=sha(folder/'user.wav'),dialogue_clock='authored_synthetic_native_frames',
            worker_clock='actual_monotonic_separate_not_relabelled',independent_semantic_review='PENDING')
        (folder/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
        if receipt:(folder/'real-worker.json').write_bytes(path.read_bytes())
        rows.append(dict(id=scene['id'],fold='TRAIN',family=scene['family'],frames=timeline['frames'],
            timeline_sha256=sha(folder/'timeline.json'),user_wav_sha256=timeline['user_wav_sha256']))
    report=dict(schema='dl-retention96-measured-timelines-v1',complete96=len(rows)==96,train=96,dev=0,
        scenes=rows,source_plan_sha256=sha(PLAN),payload_sha256=sha(a.payload),
        independent_word_quality_sha256=sha(a.quality),alignment73_sha256=sha(a.alignment),
        tokenizer_sha256=sha(a.tokenizer),native_speech_timing_schema='word-aligned-native-v1',
        actual_worker_review_sha256=sha(work_review),
        teacher_forced_requests=True,native_student_dispatch=False,original_recordings_preserved=True,
        no_per_word_timestamp_interpolation=True,independent_semantic_review='PENDING',
        native_cache_and_replay_approval='PENDING',training_approved=False)
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='scenes'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','resumed','payload','quality','alignment','receipts','tokenizer','output'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
