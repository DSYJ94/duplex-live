"""V4-source recorder with documented neural voice-delegation prompt and unchanged backend.

The original v1 code and raw receipts remain immutable. Source words, PCM,
thresholds, native prefix and backend/lifecycle remain unchanged. The new source
validator explicitly binds the one diagnosed same-words pacing repair. No collection
or teacher approval can be inferred from one recording.
"""
import hashlib
import json
from pathlib import Path
from evals.native_control_teacher import atomic_json,local_key
from evals.native_control_teacher_artifacts import input_availability
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_pacing import verify as source_manifest
from evals.native_reasoning_stream_v5 import capture,configuration,RATE
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_teacher import geometry,frame_budget

MODULES=('native_reasoning_teacher_v6.py','native_reasoning_stream_v5.py','native_reasoning_lifecycle_v3.py','native_reasoning_lifecycle_v2.py',
         'native_reasoning_teacher.py','native_reasoning_backend.py','native_reasoning_lifecycle.py',
         'native_reasoning_source_pacing.py','native_reasoning_source_quality_v4.py')
def code_hashes():return {name:sha(Path(__file__).with_name(name)) for name in MODULES}

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
    env=dict(schema='dl-native-reasoning32-teacher-environment-v6',source_id=scene['id'],source_manifest_sha256=sha(source/'source-audio.json'),
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
