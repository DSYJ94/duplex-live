"""Collect the exact qualified reasoning32 source with real asynchronous workers.

One immutable whole collection, at most32 sessions/two concurrent/six requests
each. Failed attempts remain raw and are never retried. No teacher approval,
GPU provisioning, training, output filter, oracle or reference answer is provided.
"""
import argparse
import asyncio
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

from evals.native_control_teacher import atomic_json,collection_lock,local_key
from evals.native_control_teacher_artifacts import input_availability
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_quality import verify as verify_source
from evals.native_reasoning_source_repair import verify as source_manifest
from evals.native_reasoning_stream import capture,configuration,RATE
from evals.native_reasoning_lifecycle import canonical_hash

SCHEMA='dl-native-reasoning32-actual-teacher-collection-v1'
ENGINE_SHA='8e14b636983ba4db9e31ca1179b00516383d8ccb5e56acc19d3b57e34b593d7e'
TOKENIZER_SHA='78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d'
MODULES=('native_reasoning_teacher.py','native_reasoning_stream.py','native_reasoning_backend.py','native_reasoning_lifecycle.py')


def code_hashes():return {name:sha(Path(__file__).with_name(name)) for name in MODULES}


def geometry(engine,tokenizer):
    import sentencepiece as spm
    if sha(engine)!=ENGINE_SHA or sha(tokenizer)!=TOKENIZER_SHA:raise ValueError('Exact retained native prefix audit and tokenizer required')
    value=json.loads(engine.read_text())
    if (len(value['cases'])!=40 or any(r['ownership']['prefix_frames']!=232 or r['passed'] is not True for r in value['cases'])
            or value['prefix_sha256']!='95a557f5a7865b262da055535c4fe216a165d62e739aecf2449acac7ee9b1325'):
        raise ValueError('Complete unchanged approved native prefix geometry required')
    return spm.SentencePieceProcessor(str(tokenizer))


def frame_budget(report,ledger,tokenizer):
    # Same span layout as the preserved Context Spanning sequence convention:
    # OPEN + actual returned content tokens + CLOSE. Never omit an old span
    # just because it was later superseded. Native assembly must still verify.
    spans=[dict(task_id=k,frames=2+len(tokenizer.encode(v['receipt']['content'])))
           for k,v in ledger['tasks'].items() if 'wire_write_completed_at_s' in v]
    live=math.ceil(report['observed_until_s']*12.5)
    full=232+live+sum(r['frames'] for r in spans)
    return dict(prefix_frames=232,full_measured_real_frames=live,actual_sent_context_spans=spans,
        full_measured_prefix_live_and_span_frames=full,unchanged_frame_limit=1100,
        frame_count_diagnostic_pass=full<=1100,complete_native_assembly_and_numerical_approval=False,
        original_audio_cropped=False)


async def capture_one(source,scene,row,folder,quality_sha,backend_model,env_file,tokenizer,*,connector=None):
    import numpy as np
    import soundfile as sf
    from evals.audio_playout import render_packets
    actual=connector is None
    if actual:
        from websockets.asyncio.client import connect
        connector=connect
    manifest,_=source_manifest(source);objects={r['object_id']:r for r in manifest['objects']}
    clips=[];bindings=[]
    for c in scene['clips']:
        obj=objects[c['object_id']];p=source/obj['wav'];pcm,rate=sf.read(p,dtype='int16')
        if rate!=RATE or pcm.ndim!=1 or sha(p)!=obj['wav_sha256']:raise ValueError('Changed qualified source waveform')
        clips.append(pcm.astype('<i2',copy=False));bindings.append(dict(object_id=c['object_id'],wav_sha256=sha(p),samples=len(pcm)))
    folder.mkdir(parents=True,exist_ok=False)
    env=dict(schema='dl-native-reasoning32-teacher-environment-v1',source_id=scene['id'],source_manifest_sha256=sha(source/'source-audio.json'),
        source_quality_sha256=quality_sha,original_source_bindings=bindings,actual_provider_transport=actual,
        backend_model=backend_model,session_config=configuration(),code_hashes=code_hashes(),
        operator_annotations_or_oracles_in_model_inputs=False,native_request_forcing=False,
        output_suppression=False,automatic_retry=False,training_approved=False)
    atomic_json(folder/'environment.json',env)
    report=None
    try:
        with (folder/'events.jsonl').open('x') as log,(folder/'output-stream.pcm').open('xb') as output,(folder/'input-stream.pcm').open('xb') as inputs:
            def emit(e):log.write(json.dumps(e)+'\n');log.flush()
            def output_sink(b):output.write(b);output.flush()
            def input_sink(b):inputs.write(b);inputs.flush()
            key=local_key(env_file) if actual else 'fixture-only-no-credential'
            async with connector('wss://api.openai.com/v1/live/sessions',additional_headers={
                    'Authorization':'Bearer '+key,'OpenAI-Safety-Identifier':hashlib.sha256(b'duplex-live-evaluation').hexdigest()},
                    open_timeout=20,max_size=2**24) as ws:
                report,packets,transcripts=await capture(ws,clips,folder,emit,backend_model=backend_model,
                    env_file=env_file,minimum_delay_s=row['annotations_not_model_inputs']['minimum_result_delivery_delay_s'],
                    audio_sink=output_sink,input_sink=input_sink)
    except Exception as error:
        # Raw spools and HTTP bodies stay in place even if context-manager
        # finalization fails after capture returned. Reconstruct metadata only.
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        packets=[dict(arrival_s=e['received_at_s'],byte_offset=e['output_byte_offset'],bytes=e['audio_bytes'],pcm_sha256=e['pcm_sha256'])
                 for e in events if e['type']=='session.output_audio.delta']
        transcripts=[e for e in events if e['type'] in ('session.input_transcript.delta','session.output_transcript.delta')]
        if report is None:report=dict(transport_complete=False,session_finalized=False,observed_until_s=0.,errors=[])
        report['errors'].append(dict(type='connection_context_failure',error_type=type(error).__name__));report['transport_complete']=False
    audio=(folder/'output-stream.pcm').read_bytes();input_bytes=(folder/'input-stream.pcm').read_bytes()
    atomic_json(folder/'audio-packets.json',packets);atomic_json(folder/'transcripts.json',transcripts)
    if packets:
        chunks=[(p['arrival_s'],np.frombuffer(audio[p['byte_offset']:p['byte_offset']+p['bytes']],dtype='<i2')) for p in packets]
        original,playout=render_packets(chunks,RATE,0);sf.write(folder/'model_out.wav',original,RATE,subtype='PCM_16')
        sf.write(folder/'packet-stream.wav',np.concatenate([p for _,p in chunks]),RATE,subtype='PCM_16')
        atomic_json(folder/'playout.json',playout)
    if input_bytes:sf.write(folder/'user_in.wav',np.frombuffer(input_bytes,dtype='<i2'),RATE,subtype='PCM_16')
    if (folder/'lifecycle.json').is_file():report['native_frame_budget']=frame_budget(report,json.loads((folder/'lifecycle.json').read_text()),tokenizer)
    if code_hashes()!=env['code_hashes']:
        report['errors'].append(dict(type='capture_code_changed'));report['transport_complete']=False
    report.update(actual_provider_transport=actual,teacher_or_training_approved=False)
    atomic_json(folder/'report.json',report)
    files={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file()}
    return dict(id=scene['id'],dir=scene['id'],transport_complete=report['transport_complete'],
        frame_count_diagnostic_pass=report.get('native_frame_budget',{}).get('frame_count_diagnostic_pass',False),artifacts=files)


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
    return r


async def collect(source,quality,out,env_file,engine,tokenizer_path,backend_model):
    verify_source(source,quality);manifest,plan=source_manifest(source);tokenizer=geometry(engine,tokenizer_path)
    if backend_model!='gpt-6-luna':raise ValueError('Explicit existing bounded backend model required')
    with collection_lock(out.parent/'.private/native-reasoning32-teacher.lock'):
        out.mkdir(parents=True,exist_ok=False);code=code_hashes();semaphore=asyncio.Semaphore(2);stop=False
        receipt=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),source_manifest_sha256=sha(source/'source-audio.json'),
            source_quality_sha256=sha(quality),code_hashes=code,repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            backend_model=backend_model,maximum_sessions=32,maximum_concurrent_sessions=2,maximum_backend_requests=192,
            attempted_ids=[],recordings=[],status='collecting_raw_unapproved',automatic_retry=False,training_approved=False)
        target=out/'teacher-collection.json';atomic_json(target,receipt)
        async def one(scene,row):
            nonlocal stop
            async with semaphore:
                if stop:return
                receipt['attempted_ids'].append(scene['id']);atomic_json(target,receipt)
                try:
                    record=await capture_one(source,scene,row,out/scene['id'],sha(quality),backend_model,env_file,tokenizer)
                    receipt['recordings'].append(record);receipt['recordings'].sort(key=lambda r:r['id']);atomic_json(target,receipt)
                    if not record['transport_complete']:stop=True
                    else:verify_record(source,scene,out/scene['id'],record,tokenizer)
                    print(json.dumps(dict(id=scene['id'],transport_complete=record['transport_complete'],frame_count_diagnostic_pass=record['frame_count_diagnostic_pass'],completed=len(receipt['recordings']))),flush=True)
                except Exception as error:
                    stop=True;receipt.setdefault('failures',[]).append(dict(id=scene['id'],error_type=type(error).__name__));atomic_json(target,receipt)
        try:await asyncio.gather(*(one(scene,row) for scene,row in zip(manifest['scenes'],plan['cases'])))
        finally:
            receipt['status']='raw_complete_unapproved' if len(receipt['recordings'])==32 and all(r['transport_complete'] for r in receipt['recordings']) and not receipt.get('failures') else 'incomplete_failed_preserved_no_retry'
            if code_hashes()!=code:receipt['status']='capture_code_changed_failed'
            atomic_json(target,receipt)
        return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','quality','out','env-file','engine','tokenizer'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--backend-model',default='gpt-6-luna');a=p.parse_args()
    r=asyncio.run(collect(a.source,a.quality,a.out,a.env_file,a.engine,a.tokenizer,a.backend_model))
    print(json.dumps(dict(status=r['status'],recordings=len(r['recordings']),training_approved=False)))
    if r['status']!='raw_complete_unapproved':raise SystemExit(1)
