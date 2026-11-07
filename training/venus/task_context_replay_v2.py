"""Bind ALL actual native dispatches using the unchanged production parser.

Malformed/nested spans remain failures, never repaired words or qualification.
The legacy replay stays unchanged and rejects the captured nested opening.
"""
import argparse,hashlib,json,re,wave
from pathlib import Path
from portable.venus.native_app import DelegateParser
from portable.venus.task_context import CausalAudioBuffer


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_frames(frames,pcm):
    buffer=CausalAudioBuffer();parser=DelegateParser(capture_opening=buffer.freeze_at_request_opening)
    records=[];malformed=[];opened=None
    for index,frame in enumerate(frames):
        raw=pcm[index*32000:(index+1)*32000]
        if len(raw)!=32000 or frame['sequence']!=index or hashlib.sha256(raw).hexdigest()!=frame['input_sha256']:raise ValueError('Original causal input/frame hash differs')
        buffer.append(index,raw)
        for token in re.findall(r'</?delegate>',frame['raw_text']):
            if token=='<delegate>':
                if opened is not None:malformed.append(dict(kind='nested_opening_prior_span_abandoned',abandoned_opening_sequence=opened,new_opening_sequence=index))
                opened=index
            else:
                if opened is None:malformed.append(dict(kind='closing_without_active_opening',sequence=index))
                opened=None
        queries=parser.feed(frame['raw_text'])
        events=[e for e in frame['events'] if e['type']=='native_delegate']
        if len(queries)!=len(events) or len(queries)!=len(parser.last_contexts):raise ValueError('All production-parser dispatches and captured events must match')
        for query,raw_query,evidence,event in zip(queries,parser.last_raw_queries,parser.last_contexts,events):
            if query!=event['query'] or raw_query!=event['raw_query'] or evidence.last_sequence>index:raise ValueError('Actual dispatch bytes/causal boundary differs')
            records.append(dict(work_id=event['work_id'],native_selected_description=query,native_raw_description=raw_query,description_is_not_verified_user_transcript=True,request_close_sequence=index,evidence=evidence.receipt(),evaluation_only=True))
    if opened is not None:malformed.append(dict(kind='unclosed_final_span_not_dispatched',opening_sequence=opened))
    return records,malformed


def build(collection,artifacts):
    root=Path(collection);artifacts=Path(artifacts);report=json.loads((root/'results.json').read_text());cases=report['cases']
    if len(cases)!=32 or len({c['id'] for c in cases})!=32:raise ValueError('Complete32 preserved cases required')
    records=[];malformed=[];expected=[]
    for case in cases:
        timeline='paced-timelines-v22' if case['evaluation_set']=='fresh24' else 'timelines-v7-repaired'
        audio=artifacts/timeline/case['id']/'user.wav'
        if sha(audio)!=case['input_sha256']:raise ValueError('Original full user audio hash differs')
        with wave.open(str(audio)) as f:
            if (f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):raise ValueError('Original16kmonoPCM16 required')
            pcm=f.readframes(f.getnframes())
        trace=root/case['id']/'trace.jsonl';frames=[json.loads(x) for x in trace.read_text().splitlines()]
        if len(frames)!=case['input_frames'] or len(pcm)!=len(frames)*32000:raise ValueError('Whole original case required')
        accepted,broken=replay_frames(frames,pcm)
        actual=[e['work_id'] for e in case['actual_worker_events'] if e['type']=='native_delegate']
        if [r['work_id'] for r in accepted]!=actual:raise ValueError('Complete case event identities differ')
        expected.extend(actual)
        for ordinal,row in enumerate(accepted,1):records.append(dict(row,scene_id=case['id'],request_ordinal=ordinal,audio_file=str(audio),audio_file_sha256=case['input_sha256'],trace_sha256=sha(trace)))
        malformed.extend(dict(scene_id=case['id'],**row) for row in broken)
    if not expected or len(records)!=len(expected) or len(set(expected))!=len(expected):raise ValueError('All unique actual dispatches required')
    return dict(schema='dl-native-causal-task-context-replay-v2',status='OFFLINE_CONTEXT_BINDING_ONLY_NOT_BACKEND_OR_NATIVE_QUALIFICATION',complete32=True,native_requests=len(records),all_actual_requests_bound=True,records=records,malformed_spans_preserved=malformed,malformed_stream_cases=len({r['scene_id'] for r in malformed}),parser_profile='Unchanged production DelegateParser; last-opening semantics; exact query/raw-query matching; malformed spans remain unqualified',source_parser_sha256=sha('portable/venus/native_app.py'),source_collection_sha256=sha(root/'results.json'),source_audio_bytes_unchanged=True,scoring_answers_or_synthetic_worlds_in_payload=False,external_calls=0,new_optimizer_updates=0,training_use_prohibited=True,qualified=False,adopted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('collection','artifacts','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Preserve earlier outputs')
    result=build(a.collection,a.artifacts);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','malformed_spans_preserved')}))
