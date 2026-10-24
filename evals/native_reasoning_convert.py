"""Measured complete32 native data, gated by full speech and semantic approvals.

No provider requests, codec/model loading or optimizer work. Preserve every
original PCM sample, independently measured word and actual request/delivery receipt. Only
observed-end/grid-tail silence is added. Numerical/native approval stays required.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import subprocess

from evals import native_reasoning_teacher_mixed as mixed
from evals import native_reasoning_teacher_mixed_speech as speech
from evals import native_reasoning_semantic_prepared as prepared_review
from evals import native_reasoning_semantic as semantic
from evals.native_reasoning_source import sha
from evals.native_reasoning_verification_session import source_readbacks
from evals.native_control_convert import padded_stems
from evals.native_control_teacher_artifacts import input_availability

SCHEMA='dl-native-reasoning32-measured-conversion-v1'
FAMILIES={'direct-basic','delegate-reasoning','cancel-reasoning','replace-reasoning'}
FILES=('stem_user.wav','stem_assistant.wav','conversation.wav','conversation.json','timeline.actual.json')
CODE=('evals/native_reasoning_convert.py','evals/native_control_convert.py',
      'data/synthetic/timeline.py','data/synthetic/timing_audit.py','tokenizer/grid.py')


def code_pins():return {n:sha(Path(n))for n in CODE}


def approvals(c,batch,speech_approval,prepared,semantic_approval):
    if not semantic_approval.is_file():raise ValueError('Complete32 independent semantic approval is required before conversion')
    sent=prepared_review.validate(c,batch,speech_approval,prepared)
    semantic.verify_responses(sent,c.source_quality,batch,speech_approval,semantic_approval,prepared_inputs=prepared)
    checked=speech.verify(c,batch,speech_approval)
    manifest=json.loads((c.source/'source-audio.json').read_text());plan=json.loads((c.source/'source-plan.json').read_text())
    if ([r['id']for r in plan['cases']]!=mixed.IDS or [r['id']for r in manifest['scenes']]!=mixed.IDS
            or any(r['split']!='train'for r in plan['cases'])
            or Counter(r['family']for r in plan['cases'])!=Counter({f:8 for f in FAMILIES})):
        raise ValueError('All32 exact TRAIN scenes in all four families required')
    return manifest,plan,checked


def task_clocks(ledger,events,observed_s):
    """Markers follow native arrivals; spans follow completed actual wire writes.

    A server acknowledgment is retained separately, never substituted for a
    response computation/commit. Undelivered obsolete results remain in the ledger
    and acquire no fictitious reference span.
    """
    tasks=ledger['tasks'];native=[e for e in events if e['type']=='session.delegation.created']
    if list(tasks)!=[e['delegation']['id']for e in native]:raise ValueError('Every request requires one actual native event')
    markers=[];arrivals=[]
    for event in native:
        identity=event['delegation']['id'];task=tasks[identity];request_s=task['native_request_observed_at_s']
        if not math.isfinite(request_s)or not 0<=event['received_at_s']<=request_s<observed_s:
            raise ValueError('Invalid actual native request clock')
        frame=math.ceil(request_s/.08)
        if not 0<=frame<math.ceil(observed_s/.08):raise ValueError('Actual request falls outside representable native frames')
        markers.append(dict(task_id=identity,observed_at_s=request_s,received_event_at_s=event['received_at_s'],marker_ms=frame*80))
        if 'wire_write_completed_at_s'not in task:continue
        at=task['wire_write_completed_at_s'];receipt=task['receipt'];content=receipt['content']
        writes=[e for e in events if e['type']=='local.commentary.write'and e['delegation_id']==identity]
        if len(writes)!=1:raise ValueError('Exactly one actual context write required')
        write=writes[0]
        acks=[e for e in events if e['type']=='session.commentary.appended'and e.get('client_event_id')==write['event_id']]
        if (len(acks)!=1 or not isinstance(content,str)or not content.strip()
                or write['content']!=content or not all(math.isfinite(x)for x in
                   (at,task['provider_finished_at_s'],receipt['actual_status_committed_at_s'],acks[0]['received_at_s']))
                or not request_s<=task['provider_finished_at_s']<=receipt['actual_status_committed_at_s']
                   <=write['received_at_s']<=at<=acks[0]['received_at_s']<=observed_s):
            raise ValueError('Delivered context needs exact content and causal computation/commit/write/ack clocks')
        arrivals.append(dict(ready_at_s=at,text=content,task_id=identity,
                             operation=receipt['operation'],revision=receipt['revision'],
                             server_ack_received_at_s=acks[0]['received_at_s']))
    arrivals.sort(key=lambda a:a['ready_at_s'])
    if len({m['marker_ms']for m in markers})!=len(markers):raise ValueError('Native requests collide on one control frame')
    if any(not 1<=math.ceil(a['ready_at_s']/.08)<math.ceil(observed_s/.08)for a in arrivals):
        raise ValueError('Delivered context is outside representable live frames')
    if len({math.ceil(a['ready_at_s']/.08)for a in arrivals})!=len(arrivals):
        raise ValueError('Two contexts share a native frame; explicit ordering approval required')
    return markers,arrivals


def measured_timeline(identity,family,scene,report,source_checks,objects,assistant_words,ledger,events):
    from data.synthetic.timeline import Timeline,Event
    from data.synthetic.timing_audit import grid_words_by_event
    if (len(scene['clips'])!=len(report['source_starts'])or len(scene['clips'])!=len(report['source_lengths'])
            or not assistant_words or any(len(w['text'].strip().split())!=1 or not math.isfinite(w['start']+w['end'])
                or not 0<=w['start']<w['end']<=report['observed_until_s']for w in assistant_words)):
        raise ValueError('Complete measured source clips and individual assistant words required')
    ev=[];user_words=[]
    for clip,start,samples in zip(scene['clips'],report['source_starts'],report['source_lengths']):
        obj=objects[clip['object_id']];audit=source_checks[clip['object_id']]
        if type(start)is not int or start<0 or type(samples)is not int or samples<1 or audit['passed']is not True:
            raise ValueError('Exact measured original source sample intervals required')
        if (start+samples)/24000>report['observed_until_s']:raise ValueError('Source clip extends beyond the complete observation')
        measured=audit['measured_words']
        if not measured or any(not 0<=w['start']<w['end']<=samples/24000 for w in measured):
            raise ValueError('Independent complete source word clocks required')
        user_words.extend(dict(text=w['text'].strip(),start_s=start/24000+w['start'],end_s=start/24000+w['end'],
                               object_id=clip['object_id'],source_wav_sha256=obj['wav_sha256'])for w in measured)
        ev.append(Event('user',round(start/24),clip['text'],actual_start_ms=round(start/24),
                        actual_end_ms=round((start+samples)/24),note='Original actually sent source sample clock'))
    start=round(assistant_words[0]['start']*1000);end=round(assistant_words[-1]['end']*1000)
    ev.append(Event('assistant',start,' '.join(w['text'].strip()for w in assistant_words),
        actual_start_ms=start,actual_end_ms=end,word_onsets_ms=[round(w['start']*1000)-start for w in assistant_words],
        note='All independently measured words, including every inter-response pause'))
    markers,arrivals=task_clocks(ledger,events,report['observed_until_s'])
    for m in markers:ev.append(Event('assistant',m['marker_ms'],kind='marker',
        actual_start_ms=m['marker_ms'],actual_end_ms=m['marker_ms'],note='delegate actual native request '+m['task_id']))
    timeline=Timeline(identity,'native_reasoning4_teacher',sorted(ev,key=lambda e:e.actual_start_ms),
        dict(split='train',family=family,source_model='gpt-live-1',teacher_audio=True,verified=False,
             reference_arrivals=arrivals,request_markers=markers,
             context_clock='Actual completed client wire write; server acknowledgment separately retained',
             all_native_requests_and_delivered_contexts_preserved=True))
    _,grid=grid_words_by_event(timeline)
    if len(grid)!=len(assistant_words)or [w for _,_,w in grid]!=[w['text'].strip()for w in assistant_words]:
        raise ValueError('A spoken word was lost, duplicated or reordered on the native grid')
    offset=max(abs(tick*80-word['start']*1000)for (_,tick,_),word in zip(grid,assistant_words))
    if offset>200:raise ValueError('Word placement exceeds unchanged200ms timing bar')
    return timeline,user_words,markers,arrivals,offset


def derive(c,batch,scene,case,audio,source_checks,objects,tokenizer):
    folder=batch/scene['id'];report=json.loads((folder/'report.json').read_text())
    ledger=json.loads((folder/'lifecycle.json').read_text());events=[json.loads(l)for l in(folder/'events.jsonl').read_text().splitlines()]
    timeline,user_words,markers,arrivals,offset=measured_timeline(scene['id'],case['family'],scene,report,
        source_checks,objects,audio['measured_words'],ledger,events)
    user,assistant,clock=padded_stems(folder/'user_in.wav',folder/'model_out.wav',report['observed_until_s'])
    actual=input_availability(events,report['sent_input_samples'],(folder/'input-stream.pcm').read_bytes())
    budget=report['native_frame_budget'];span_frames=sum(2+len(tokenizer.encode(a['text']))for a in arrivals)
    full=232+clock['frames']//1920+span_frames
    if (clock['frames']//1920!=budget['full_measured_real_frames']
            or full!=budget['full_measured_prefix_live_and_span_frames']or full>1100
            or budget['unchanged_frame_limit']!=1100 or budget['frame_count_diagnostic_pass']is not True):
        raise ValueError('Complete prefix/live/all delivered spans exceed or differ from unchanged1100-frame geometry')
    side=dict(alignments=[[w['text'].strip(),[w['start'],w['end']],'SPEAKER_MAIN']for w in audio['measured_words']],
        user_words=user_words,reference_arrivals=arrivals,request_markers=markers,timeline=timeline.to_dict(),
        actual_input_packet_availability=actual,original_lifecycle=ledger,pcm_clock=clock,
        native_frame_budget=budget,actual_provider_transcript=audio['actual_provider_transcript'],
        independent_asr_text=audio['asr_text'])
    return timeline,side,user,assistant,offset


def convert(c,batch,speech_approval,prepared,semantic_approval,out):
    import numpy as np
    import soundfile as sf
    if not semantic_approval.is_file():raise ValueError('Complete32 independent semantic approval is required before conversion')
    roots=[*mixed.input_roots(c),batch,speech_approval.parent,prepared,semantic_approval.parent]
    if any(out.resolve().is_relative_to(p.resolve())for p in roots if p.is_dir()):
        raise ValueError('New conversion output must be outside every immutable input root')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Conversion requires a clean committed checkout; run heavy source checks remotely')
    pins=code_pins()
    with source_readbacks(roots,include_old_speech=True):
        manifest,plan,checked=approvals(c,batch,speech_approval,prepared,semantic_approval)
        tokenizer=mixed.geometry(c.engine,c.tokenizer);objects={x['object_id']:x for x in manifest['objects']}
        source_checks={x['object_id']:x for x in json.loads(c.source_quality.read_text())['results']}
        # Derive ALL32 before writing a directory; no passing subset is emitted.
        derived=[derive(c,batch,s,case,a,source_checks,objects,tokenizer)
                 for s,case,a in zip(manifest['scenes'],plan['cases'],checked['results'])]
        out.mkdir(parents=True,exist_ok=False);rows=[]
        for scene,case,(timeline,side,user,assistant,offset)in zip(manifest['scenes'],plan['cases'],derived):
            folder=out/scene['id'];folder.mkdir()
            for name,pcm in [('stem_user.wav',user),('stem_assistant.wav',assistant),
                             ('conversation.wav',np.stack([assistant,user],axis=1))]:sf.write(folder/name,pcm,24000,subtype='FLOAT')
            timeline.save(folder/'timeline.actual.json');(folder/'conversation.json').write_text(json.dumps(side,indent=2)+'\n')
            rows.append(dict(scenario_id=scene['id'],dir=scene['id'],split='train',family=case['family'],
                scenario_class='native_reasoning4_teacher',path=(Path(scene['id'])/'conversation.wav').as_posix(),
                samples=len(user),native_frames=len(user)//1920,
                duration=len(user)/24000,max_grid_offset_ms=offset,artifacts={n:sha(folder/n)for n in FILES}))
        for fold in('train','validation','test'):(out/(fold+'.jsonl')).write_text(''.join(json.dumps(r)+'\n'for r in rows if r['split']==fold))
        value=dict(schema=SCHEMA,at_utc=datetime.now(timezone.utc).isoformat(),code_pins=pins,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            teacher_collection_sha256=sha(batch/'teacher-collection.json'),source_quality_sha256=sha(c.source_quality),
            speech_approval_sha256=sha(speech_approval),prepared_inputs_sha256=sha(prepared),semantic_approval_sha256=sha(semantic_approval),
            complete32=True,accepted=rows,channel_order=['assistant','user'],optimizer_updates=0,training_approved=False,
            word_label_source='Complete independent measured ASR; actual provider deltas retained separately',
            nonzero_word_agreement_cases=[dict(id=r['id'],raw_wer=r['raw_wer'])for r in checked['results']if r['raw_wer']],
            human_listening='pending',
            native_codec_numerical_gradient_and_behavior_approval_required=True,original_words_audio_receipts_folds_unchanged=True)
        (out/'ingestion.json').write_text(json.dumps(value,indent=2)+'\n')
        verify_derived(out,value,derived)
        if code_pins()!=pins:raise ValueError('Conversion implementation changed during readback')
    return value


def verify_derived(out,value,derived):
    import numpy as np
    import soundfile as sf
    if ([r['scenario_id']for r in value['accepted']]!=mixed.IDS or len(derived)!=32
            or value['schema']!=SCHEMA or value['code_pins']!=code_pins()or value['complete32']is not True
            or value['channel_order']!=['assistant','user']or value['optimizer_updates']!=0 or value['training_approved']is not False
            or value['native_codec_numerical_gradient_and_behavior_approval_required']is not True
            or value['original_words_audio_receipts_folds_unchanged']is not True):
        raise ValueError('Exact complete32 derived conversion required')
    if out.is_symlink()or {p.name for p in out.iterdir()}!=set(mixed.IDS)|{'ingestion.json','train.jsonl','validation.jsonl','test.jsonl'}:
        raise ValueError('Only the complete32 approved conversion inventory is allowed')
    for row,(timeline,side,user,assistant,offset)in zip(value['accepted'],derived):
        folder=out/row['dir']
        if folder.is_symlink()or {p.name for p in folder.iterdir()}!=set(FILES)or any(p.is_symlink()for p in folder.iterdir()):
            raise ValueError('Unexpected or linked derived artifact')
        if (row['dir']!=row['scenario_id']or row['split']!='train'or row['samples']!=len(user)
                or row['native_frames']!=len(user)//1920 or row['family']!=timeline.meta['family']
                or row['scenario_class']!='native_reasoning4_teacher'
                or row['path']!=(Path(row['dir'])/'conversation.wav').as_posix()
                or row['duration']!=len(user)/24000 or row['max_grid_offset_ms']!=offset
                or json.loads((folder/'conversation.json').read_text())!=side
                or json.loads((folder/'timeline.actual.json').read_text())!=timeline.to_dict()
                or row['artifacts']!={n:sha(folder/n)for n in FILES}):raise ValueError('Measured conversion/receipt changed')
        for name,wanted in [('stem_user.wav',user),('stem_assistant.wav',assistant),('conversation.wav',np.stack([assistant,user],axis=1))]:
            pcm,rate=sf.read(folder/name,dtype='float32')
            if rate!=24000 or not np.array_equal(pcm,wanted):raise ValueError('Original PCM changed, cropped, swapped or rescaled')
    for fold in('train','validation','test'):
        if [json.loads(l)for l in(out/(fold+'.jsonl')).read_text().splitlines()]!=[r for r in value['accepted']if r['split']==fold]:
            raise ValueError('TRAIN membership or held-out folds changed')
    return value


def verify(c,batch,speech_approval,prepared,semantic_approval,out):
    """Rebuild every target from the approved originals and compare all32 files."""
    roots=[*mixed.input_roots(c),batch,speech_approval.parent,prepared,semantic_approval.parent,out]
    with source_readbacks(roots,include_old_speech=True):
        manifest,plan,checked=approvals(c,batch,speech_approval,prepared,semantic_approval)
        tokenizer=mixed.geometry(c.engine,c.tokenizer);objects={x['object_id']:x for x in manifest['objects']}
        source_checks={x['object_id']:x for x in json.loads(c.source_quality.read_text())['results']}
        derived=[derive(c,batch,s,case,a,source_checks,objects,tokenizer)
                 for s,case,a in zip(manifest['scenes'],plan['cases'],checked['results'])]
        value=json.loads((out/'ingestion.json').read_text())
        if (value['nonzero_word_agreement_cases']!=[dict(id=r['id'],raw_wer=r['raw_wer'])for r in checked['results']if r['raw_wer']]
                or value['word_label_source']!='Complete independent measured ASR; actual provider deltas retained separately'
                or value['human_listening']!='pending'):
            raise ValueError('Independent label provenance or residual transcript disagreements changed')
        for key,path in [('teacher_collection',batch/'teacher-collection.json'),('source_quality',c.source_quality),
                         ('speech_approval',speech_approval),('prepared_inputs',prepared),('semantic_approval',semantic_approval)]:
            if value[key+'_sha256']!=sha(path):raise ValueError('Conversion approval/source byte binding changed')
        return verify_derived(out,value,derived)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true')
    for n in('config','batch','speech-approval','prepared','semantic-approval','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run=verify if a.verify_only else convert;v=run(mixed.load(a.config),a.batch,a.speech_approval,a.prepared,a.semantic_approval,a.out)
    print(json.dumps(dict(complete32=v['complete32'],accepted=len(v['accepted']),training_approved=False)))
