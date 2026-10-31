"""Bind actual native requests to causal original audio; evaluation-only replay."""
import argparse
import hashlib
import json
import re
import wave
from pathlib import Path
from portable.venus.task_context import CausalAudioBuffer


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(collection, artifacts):
    root=Path(collection);artifacts=Path(artifacts)
    result=json.loads((root/'results.json').read_text())
    assert len(result['cases'])==32 and len({x['id'] for x in result['cases']})==32
    records=[];open_spans=[];calls=0
    for case in result['cases']:
        timeline='paced-timelines-v22' if case['evaluation_set']=='fresh24' else 'timelines-v7-repaired'
        audio=artifacts/timeline/case['id']/'user.wav'
        assert sha(audio)==case['input_sha256']
        with wave.open(str(audio)) as w:
            assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
            pcm=w.readframes(w.getnframes())
        trace=root/case['id']/'trace.jsonl'
        frames=[json.loads(line) for line in trace.read_text().splitlines()]
        assert len(frames)==case['input_frames'] and len(pcm)==len(frames)*32000
        buffer=CausalAudioBuffer();pending=None;ordinal=0
        for i,frame in enumerate(frames):
            raw=pcm[i*32000:(i+1)*32000]
            assert frame['sequence']==i and hashlib.sha256(raw).hexdigest()==frame['input_sha256']
            buffer.append(i,raw);closed=[]
            for token in re.findall(r'</?delegate>',frame['raw_text']):
                if token=='<delegate>':
                    assert pending is None, 'Nested native request needs separate diagnosis'
                    pending=buffer.freeze_at_request_opening()
                elif pending is not None:
                    closed.append(pending);pending=None
            events=[e for e in frame['events'] if e['type']=='native_delegate']
            assert len(closed)==len(events), 'Cannot infer an accepted task from an unclosed span'
            for evidence,event in zip(closed,events):
                ordinal+=1;calls+=1
                assert evidence.last_sequence<=i
                records.append(dict(scene_id=case['id'],request_ordinal=ordinal,work_id=event['work_id'],native_selected_description=event['query'],native_raw_description=event['raw_query'],description_is_not_verified_user_transcript=True,request_close_sequence=i,audio_file=str(audio),audio_file_sha256=case['input_sha256'],trace_sha256=sha(trace),evidence=evidence.receipt(),evaluation_only=True))
        if pending is not None:open_spans.append(case['id'])
    expected=sum(e['type']=='native_delegate' for case in result['cases'] for e in case['actual_worker_events'])
    assert calls==expected and expected>0
    return dict(schema='dl-native-causal-task-context-replay-v1',complete32=True,native_requests=expected,all_actual_requests_bound=True,records=records,unclosed_spans_preserved=open_spans,source_collection_sha256=sha(root/'results.json'),source_audio_bytes_unchanged=True,scoring_answers_or_synthetic_worlds_in_payload=False,external_calls=0,new_optimizer_updates=0,training_use_prohibited=True,status='OFFLINE_CONTEXT_BINDING_ONLY_NOT_BACKEND_OR_NATIVE_QUALIFICATION')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('collection','artifacts','output'):p.add_argument('--'+x,required=True)
    a=p.parse_args();d=build(a.collection,a.artifacts);Path(a.output).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k!='records'}))
