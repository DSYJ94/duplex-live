"""Collect only authored reasoning32 USER speech; never collect a teacher here.

Matched questions share the same immutable provider WAV. No retry, gain change,
resampling, answer annotation, native inference, optimizer or approval path.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from data.synthetic.native_reasoning_delegation import validate
from data.synthetic.recovery_curriculum import MODEL
from evals.native_control_teacher import atomic_json, collection_lock, local_key

ENDPOINT='https://api.openai.com/v1/audio/speech'
INSTRUCTIONS='Speak naturally in clear conversational English. Read exactly the supplied words, without additions.'
SCHEMA='dl-native-reasoning32-user-source-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def requests_for(plan):
    validate(plan)
    objects={};scenes=[]
    for row in plan['cases']:
        clips=[]
        for index,text in enumerate(row['inference_inputs']['authored_user_utterances']):
            payload=dict(model=MODEL,input=text,voice=row['user_voice'],response_format='wav',instructions=INSTRUCTIONS)
            identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            objects.setdefault(identity,payload)
            clips.append(dict(index=index,object_id=identity,text=text))
        scenes.append(dict(id=row['id'],split='train',user_voice=row['user_voice'],clips=clips))
    if len(objects)!=40 or len(scenes)!=32 or sum(len(r['clips']) for r in scenes)!=56:
        raise ValueError('Exact complete32/56 source membership and40 unique matched requests required')
    return objects,scenes


def collect(plan_path,out,env_file,*,post=None):
    plan=json.loads(plan_path.read_text());objects,scenes=requests_for(plan)
    # The real transport is selected explicitly; fixture output can never acquire
    # the actual-API evidence flag merely by returning a plausible WAV.
    actual=post is None
    if actual:
        import requests
        post=requests.post
        key=local_key(env_file)
    else:key='fixture-only-no-credential'
    with collection_lock(out.parent/'.private/native-reasoning32-source.lock'):
        out.mkdir(parents=True,exist_ok=False)
        (out/'objects').mkdir()
        (out/'source-plan.json').write_bytes(plan_path.read_bytes())
        report=dict(schema=SCHEMA,phase='collecting_unapproved_user_source',created_utc=datetime.now(timezone.utc).isoformat(),
            plan_sha256=sha(plan_path),code_sha256=sha(Path(__file__)),source_model=MODEL,
            actual_api=actual,endpoint=ENDPOINT,maximum_requests=40,actual_requests=0,
            expected_scenes=32,expected_scene_clips=56,expected_unique_objects=40,
            scenes=scenes,objects=[],complete32=False,training_approved=False,
            source_quality_approved=False,teacher_collection_completed=False,
            payload_fields=['model','input','voice','response_format','instructions'],
            operator_or_answer_annotations_sent=False,owner_conversations_used=False,
            original_WAVs_unchanged=True,gain_changed=False,resampled=False,automatic_retry=False,
            timeline_assembled=False,actual_native_episode_frame_bound_still_required=1100)
        def save():atomic_json(out/'source-audio.json',report)
        save()
        try:
            for identity,payload in objects.items():
                wav=out/'objects'/(identity+'.wav')
                report['actual_requests']+=1;save()
                response=post(ENDPOINT,headers={'Authorization':'Bearer '+key},json=payload,timeout=90)
                if response.status_code!=200:
                    report['failed_object_id']=identity;report['http_status']=response.status_code
                    raise RuntimeError('Authored source speech HTTP '+str(response.status_code)+'; no retry')
                wav.write_bytes(response.content)
                receipt=dict(request_id=response.headers.get('x-request-id'),source_model=MODEL,
                    voice=payload['voice'],input_text=payload['input'],instructions=INSTRUCTIONS,
                    sha256=sha(wav),attempt=0,actual_api=actual,request_payload_sha256=identity,
                    response_bytes=len(response.content))
                receipt_path=wav.with_suffix('.receipt.json');atomic_json(receipt_path,receipt)
                report['objects'].append(dict(object_id=identity,wav=str(wav.relative_to(out)),
                    wav_sha256=receipt['sha256'],receipt_sha256=sha(receipt_path)))
                save()
                print(json.dumps(dict(source_object=len(report['objects']),expected=40,unapproved=True)),flush=True)
            report.update(phase='complete_unapproved_user_source',complete32=True)
            save()
        except BaseException as error:
            report.update(phase='failed_preserved_no_retry',error_type=type(error).__name__)
            save();raise
    return report


def verify(root,*,require_actual=True):
    root=Path(root);p=root/'source-plan.json';plan=json.loads(p.read_text());objects,scenes=requests_for(plan)
    r=json.loads((root/'source-audio.json').read_text())
    if (r['schema']!=SCHEMA or r['phase']!='complete_unapproved_user_source'
            or r['complete32'] is not True or (require_actual and r['actual_api'] is not True)
            or r['plan_sha256']!=sha(p) or r['code_sha256']!=sha(Path(__file__))
            or r['source_model']!=MODEL or r['endpoint']!=ENDPOINT
            or r['maximum_requests']!=40 or r['actual_requests']!=40
            or r['expected_scenes']!=32 or r['expected_scene_clips']!=56 or r['expected_unique_objects']!=40
            or r['scenes']!=scenes or [x['object_id'] for x in r['objects']]!=list(objects)
            or r['training_approved'] is not False or r['source_quality_approved'] is not False
            or r['teacher_collection_completed'] is not False or r['operator_or_answer_annotations_sent'] is not False
            or r['owner_conversations_used'] is not False or r['original_WAVs_unchanged'] is not True
            or r['gain_changed'] is not False or r['resampled'] is not False or r['automatic_retry'] is not False
            or r['timeline_assembled'] is not False or r['actual_native_episode_frame_bound_still_required']!=1100):
        raise ValueError('Complete exact unapproved user source and real provider transport required')
    expected_files={'source-plan.json','source-audio.json'}
    for row in r['objects']:
        identity=row['object_id'];name='objects/'+identity+'.wav'
        wav=root/name;receipt=wav.with_suffix('.receipt.json');v=json.loads(receipt.read_text());payload=objects[identity]
        if (row['wav']!=name or wav.is_symlink() or receipt.is_symlink()
                or row['wav_sha256']!=sha(wav) or row['receipt_sha256']!=sha(receipt)
                or v!=dict(request_id=v.get('request_id'),source_model=MODEL,voice=payload['voice'],
                    input_text=payload['input'],instructions=INSTRUCTIONS,sha256=sha(wav),attempt=0,
                    actual_api=r['actual_api'],request_payload_sha256=identity,response_bytes=wav.stat().st_size)
                or (require_actual and not v['request_id'])):
            raise ValueError('Changed original source bytes, payload or provider receipt')
        expected_files.update((name,str(receipt.relative_to(root))))
    if {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}!=expected_files:
        raise ValueError('Unexpected or missing source object/receipt inventory')
    return r,plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--env-file',type=Path,required=True)
    a=p.parse_args();collect(a.plan,a.out,a.env_file);verify(a.out)


if __name__=='__main__':main()
