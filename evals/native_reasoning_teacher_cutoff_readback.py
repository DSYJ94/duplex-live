"""Independent V7 capture readback; preserve original NameError verdict and bytes.

Only readback obtains the already-recorded transcript list before using it.
The frozen original recorder's exact source hashes/configuration remain required.
No provider request, waveform repair or whole-corpus approval occurs here.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

from evals.native_reasoning_teacher_v7 import code_hashes,configuration,source_manifest
from evals.native_reasoning_stream_v6 import RATE,last_active_sample,quiet_deadline
from evals.native_control_teacher_artifacts import input_availability
from evals.native_reasoning_teacher import geometry,frame_budget
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_source import sha
from evals.native_reasoning_teacher_lifecycle_review import case_review
from evals import native_reasoning_teacher_cutoff_probe as original_probe

def verify_record(source,scene,folder,record,tokenizer):
    """Independent input/output readback; never a semantic or teacher approval."""
    import numpy as np
    import soundfile as sf
    from evals.audio_playout import render_packets
    for name,row in record['artifacts'].items():
        p=folder/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink()
                or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']):raise ValueError('Changed raw teacher artifact')
    r=json.loads((folder/'report.json').read_text());env=json.loads((folder/'environment.json').read_text())
    if (r.get('transport_complete') is not True or env['actual_provider_transport'] is not True
            or env['session_config']!=configuration() or env['code_hashes']!=code_hashes()
            or env['source_id']!=scene['id'] or env['source_manifest_sha256']!=sha(source/'source-audio.json')
            or env['operator_annotations_or_oracles_in_model_inputs'] is not False):raise ValueError('Complete actual unchanged raw transport required')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    stream=(folder/'input-stream.pcm').read_bytes();input_availability(events,r['sent_input_samples'],stream)
    manifest,_=source_manifest(source);objects={v['object_id']:v for v in manifest['objects']}
    for clip,start in zip(scene['clips'],r['source_starts']):
        pcm,rate=sf.read(source/objects[clip['object_id']]['wav'],dtype='int16')
        if start is None or rate!=RATE or stream[start*2:(start+len(pcm))*2]!=pcm.astype('<i2').tobytes():raise ValueError('Source utterance changed or cropped in actual stream')
    audio=(folder/'output-stream.pcm').read_bytes();packets=json.loads((folder/'audio-packets.json').read_text());cursor=0;chunks=[]
    for p in packets:
        block=audio[p['byte_offset']:p['byte_offset']+p['bytes']]
        if p['byte_offset']!=cursor or len(block)!=p['bytes'] or hashlib.sha256(block).hexdigest()!=p['pcm_sha256']:raise ValueError('Changed or omitted output packet')
        cursor+=len(block);chunks.append((p['arrival_s'],np.frombuffer(block,dtype='<i2')))
    if cursor!=len(audio):raise ValueError('Unaccounted raw audio samples')
    expected,_=render_packets(chunks,RATE,0);saved,rate=sf.read(folder/'model_out.wav',dtype='int16')
    if rate!=RATE or not np.array_equal(expected,saved) or r['observed_until_s']<len(saved)/RATE:raise ValueError('Output changed, cropped or not fully observed')
    ledger=json.loads((folder/'lifecycle.json').read_text());native=[e for e in events if e['type']=='session.delegation.created']
    if list(ledger['tasks'])!=[e['delegation']['id'] for e in native]:raise ValueError('Every request must originate from one actual native event')
    for event in native:
        identity=event['delegation']['id'];task=ledger['tasks'][identity]
        prefix=[dict(speaker='user',text=e['delta'],received_at_s=e['received_at_s']) for e in events
                if e['type']=='session.input_transcript.delta' and e.get('delta','').strip()
                and events.index(e)<events.index(event)]
        path=folder/('backend-'+hashlib.sha256(identity.encode()).hexdigest()[:24])
        payload=json.loads((path/'request.json').read_text());trace=json.loads((path/'transport.json').read_text())
        response=json.loads((path/'response.json').read_text())
        if (task['context']!=prefix or canonical_hash(payload)!=task['payload_sha256']
                or json.loads(payload['input'])['arrived_user_transcript']!=prefix
                or trace['actual_provider_transport'] is not True or not trace.get('provider_request_id')
                or trace['automatic_retry'] is not False or trace['maximum_attempts']!=1
                or trace['response_sha256']!=sha(path/'response.json') or task['response']!=response
                or task['actual_transport'] is not True):raise ValueError('Actual arrived context and provider response association differ')
    if r['native_frame_budget']!=frame_budget(r,ledger,tokenizer):raise ValueError('Complete unchanged native frame accounting differs')
    closes=[e for e in events if e['type']=='local.session.close.requested']
    if len(closes)!=1:raise ValueError('One actual output-quiescent close required')
    close=closes[0];cursor=0;last_active=0
    for packet in packets:
        first=max(cursor,__import__('math').ceil(packet['arrival_s']*RATE))
        block=audio[packet['byte_offset']:packet['byte_offset']+packet['bytes']]
        active=last_active_sample(block,first)
        if active is not None:last_active=max(last_active,active)
        cursor=first+len(block)//2
    transcripts=json.loads((folder/'transcripts.json').read_text())
    last_text=max((e['received_at_s']for e in transcripts
                   if e['type']=='session.output_transcript.delta'and e.get('delta','').strip()),default=0.)
    if (close.get('required_output_quiet_s')!=1.
            or close['received_at_s']<quiet_deadline(last_active,last_text)
            or close.get('original_last_active_playout_sample')!=last_active
            or close.get('last_output_text_at_s')!=last_text
            or close.get('output_quiet_deadline_s')!=quiet_deadline(last_active,last_text)):
        raise ValueError('Original full output/text was not quiescent before close')
    return r


def review(source, quality, parent, failed, component, engine, tokenizer_path, output):
    original=component/'cutoff-component.json';value=json.loads(original.read_text())
    failed_value=json.loads(failed.read_text())
    expected=dict(code_hashes(),**{Path(original_probe.__file__).name:sha(Path(original_probe.__file__))})
    if (value['status']!='failed_preserved_no_retry' or value['error_type']!='NameError'
            or value['reason']!="name 'transcripts' is not defined" or value['code_hashes']!=expected
            or value['parent_collection_sha256']!=original_probe.PARENT_SHA
            or sha(parent/'teacher-collection.json')!=original_probe.PARENT_SHA
            or value['failed_whole32_speech_sha256']!=sha(failed)
            or failed_value['complete_cases']!=32 or failed_value['passed'] is not False
            or value['source_manifest_sha256']!=sha(source/'source-audio.json')
            or value['source_quality_sha256']!=sha(quality)
            or value['case_id']!=original_probe.CASE or value['maximum_new_voice_sessions']!=1
            or value['automatic_retry']is not False or value['teacher_approved']is not False
            or value['training_approved']is not False):
        raise ValueError('Exact immutable one-capture checker failure required')
    manifest,plan=source_manifest(source)
    scene=next(s for s in manifest['scenes']if s['id']==original_probe.CASE)
    case=next(r for r in plan['cases']if r['id']==original_probe.CASE)
    tokenizer=geometry(engine,tokenizer_path);folder=component/original_probe.CASE
    report=verify_record(source,scene,folder,value['recording'],tokenizer)
    lifecycle=case_review(case,json.loads((folder/'lifecycle.json').read_text()))
    if report['native_frame_budget']['frame_count_diagnostic_pass']is not True or lifecycle['structural_pass']is not True:
        raise ValueError('Actual complete lifecycle/frame gate remains required')
    result=dict(schema='dl-native-reasoning32-cutoff-independent-readback-v1',
        at_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_sha256=sha(Path(__file__)),original_failed_component_sha256=sha(original),
        raw_recording_canonical_sha256=canonical_hash(value['recording']),
        original_recording_code_hashes=expected,original_failure_and_files_unchanged=True,
        complete_actual_transport_and_quiescent_output=True,full_frame_budget=report['native_frame_budget'],
        actual_lifecycle_diagnostic=lifecycle,additional_provider_requests=0,
        independent_speech_and_spoken_semantics_still_required=True,
        whole32_approved=False,teacher_approved=False,training_approved=False)
    with output.open('x')as stream:stream.write(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','quality','parent','failed','component','engine','tokenizer','output'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();v=review(a.source,a.quality,a.parent,a.failed,a.component,a.engine,a.tokenizer,a.output)
    print(json.dumps(dict(component_actual_transport_and_quiescence_passed=True,
                         additional_provider_requests=0,whole32_approved=False,training_approved=False)))
