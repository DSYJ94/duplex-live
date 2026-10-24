"""Complete32 semantic reapproval retaining30 exact prior passing responses.

Only two diagnosed independent clock inputs change. No questions, recognized
words, recordings, receipts, prompts, model or semantic rule change. At most two
new requests; every unchanged response remains bound to its exact original input.
"""
import argparse
import asyncio
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil

from evals import native_reasoning_timing_reapproval as timing
from evals import native_reasoning_semantic as semantic
from evals import native_reasoning_semantic_prepared as original
from evals import native_reasoning_teacher_mixed as mixed
from evals.native_control_teacher import atomic_json,collection_lock,local_key
from evals.native_reasoning_source import sha
from training.control.teacher_data import request


def validated(a,version):
    c=mixed.load(a['config'])
    old_sent=original.validate(c,a['batch'],a['old_speech'],a['old_prepared'])
    failed=timing.checked_failure(a,old_sent)
    old=json.loads(a['old_speech'].read_text())
    new=timing.checked_bursts(a,old)
    qualified=json.loads((version/'teacher-speech-quality.json').read_text())
    prepared=json.loads((version/'prepared-semantic.json').read_text())
    proof=json.loads((version/'independent-readback.json').read_text())
    if (qualified['schema']!=timing.SCHEMA+'-speech'
            or qualified['code_sha256']!=sha(Path(timing.__file__))
            or qualified['old_speech_sha256']!=sha(a['old_speech'])
            or qualified['old_prepared_sha256']!=sha(a['old_prepared'])
            or qualified['failed_semantic_sha256']!=sha(a['failed_semantic'])
            or qualified['independent_clock_recognition_sha256']!=sha(a['burst']/'recognition.json')
            or qualified['complete_cases']!=32 or qualified['passed']is not True
            or [r['id']for r in qualified['results']]!=mixed.IDS
            or qualified['repaired_clock_ids']!=list(timing.IDS)
            or any(qualified[k]is not False for k in ('original_audio_words_receipts_folds_changed',
                'passing30_redecoded','passing_subset_accepted','semantic_approved','training_approved'))
            or proof['schema']!=timing.SCHEMA+'-readback'
            or proof['source_commit']!='3b29b9936196010b5228fa15a2c0bd32ed7a67fb'
            or any(proof[k]is not True for k in ('complete32_original_transport_and_speech_checked',
                'two_new_speech_decisions_recomputed_twice','unchanged_input_snapshot','all32_speech_passed'))
            or proof['speech_sha256']!=sha(version/'teacher-speech-quality.json')
            or proof['prepared_sha256']!=sha(version/'prepared-semantic.json')
            or proof['provider_requests']!=0 or proof['training_approved']is not False):
        raise ValueError('Complete unchanged remote32 readback and diagnosed clock lineage required')
    for before,after in zip(old['results'],qualified['results']):
        if before['id']not in timing.IDS:
            if before!=after:
                raise ValueError('All30 original passing speech records must remain exact')
            continue
        raw,binding=new[before['id']]
        expected=deepcopy(before)
        expected.update(asr_text=raw['text'],measured_words=[w for s in raw['segments']for w in s.get('words',[])],
            recognition_binding=dict(kind='independent-burst-clock-v11',wav_sha256=binding['wav_sha256'],
                asr_sha256=binding['asr_sha256'],recognition_sha256=sha(a['burst']/'recognition.json')))
        if expected!=after:
            raise ValueError('Only independently recognized clocks/punctuation may change; all bars and verdicts remain exact')
    sent=timing.changed_messages(old_sent,qualified)
    expected=dict(schema=timing.SCHEMA+'-prepared',code_sha256=sha(Path(timing.__file__)),
        old_prepared_sha256=sha(a['old_prepared']),speech_sha256=sha(version/'teacher-speech-quality.json'),
        failed_semantic_sha256=sha(a['failed_semantic']),model=semantic.MODEL,
        messages=sent,input_sha256={k:semantic.canonical_hash(v)for k,v in sent.items()},
        new_review_ids=list(timing.IDS),unchanged30_inputs_exact=True,maximum_new_requests=2,
        maximum_concurrency=2,original_audio_words_receipts_folds_changed=False,
        complete32=True,semantic_approved=False,training_approved=False)
    if prepared!=expected:
        raise ValueError('Exact whole32 derived inputs and unchanged30 messages required')
    return c,sent,failed


async def review(a,version,out,env_file):
    c,sent,failed=validated(a,version)
    with collection_lock(out.parent/'.private/native-reasoning32-semantic.lock'):
        out.mkdir(parents=True,exist_ok=False)
        rows=[]
        for row in failed['results']:
            if row['id']in timing.IDS:
                continue
            shutil.copyfile(a['failed_semantic'].parent/row['response_file'],out/row['response_file'])
            rows.append(deepcopy(row))
        value=dict(schema=semantic.SCHEMA,started_utc=datetime.now(timezone.utc).isoformat(),model=semantic.MODEL,
            code_sha256=sha(Path(semantic.__file__)),request_code_sha256=sha(Path('training/control/teacher_data.py')),
            teacher_manifest_sha256=sha(a['batch']/'teacher-collection.json'),
            speech_approval_sha256=sha(version/'teacher-speech-quality.json'),source_quality_sha256=sha(c.source_quality),
            selected_ids=mixed.IDS,maximum_requests=32,concurrency=2,maximum_new_requests=2,
            actual_reused_responses=30,new_review_ids=list(timing.IDS),actual_new_requests_started=0,
            failed_speech_sha256=None,recognition_plan_sha256=None,
            prepared_inputs_sha256=sha(version/'prepared-semantic.json'),
            original_failed_semantic_sha256=sha(a['failed_semantic']),derived_reviewer_code_sha256=sha(Path(__file__)),
            results=rows,status='reviewing_unapproved',passed=False,automatic_retry=False,
            source_oracles_sent=False,training_approved=False,native_conversion_and_numerical_required=True,
            native_behavior_qualified=False)
        target=out/'semantic.json';atomic_json(target,value)
        previous=os.environ.get('OPENAI_API_KEY');os.environ['OPENAI_API_KEY']=local_key(env_file)
        def one(identity):
            path=out/(identity+'.response.json');data,call,seconds=request(sent[identity],[semantic.TOOL],path)
            result=json.loads(call['arguments'])
            if (call['name']!=semantic.TOOL['name']or set(result)!={'verdict','reason'}
                    or result['verdict']not in ('pass','fail')or not result['reason'].strip()
                    or data['model']!=semantic.MODEL):
                raise ValueError('Malformed independent timing semantic judgment')
            return dict(id=identity,passed=result['verdict']=='pass',reason=result['reason'],response_file=path.name,
                response_sha256=sha(path),response_id=data['id'],actual_model=data['model'],seconds=seconds,
                input_sha256=semantic.canonical_hash(sent[identity]))
        async def capture(identity):
            value['actual_new_requests_started']+=1;atomic_json(target,value)
            try:
                row=await asyncio.to_thread(one,identity)
                value['results'].append(row);value['results'].sort(key=lambda r:r['id'])
                atomic_json(target,value);print(json.dumps(dict(id=identity,passed=row['passed'],complete=len(value['results']))),flush=True)
            except Exception as error:
                value.setdefault('failures',[]).append(dict(id=identity,error_type=type(error).__name__))
                atomic_json(target,value);raise
        try:
            outcomes=await asyncio.gather(*(capture(k)for k in timing.IDS),return_exceptions=True)
            if any(isinstance(v,BaseException)for v in outcomes):
                raise ValueError('Timing reapproval transport failed; raw responses retained, no retry')
            value['status']='complete_independent_review'
        finally:
            if previous is None:os.environ.pop('OPENAI_API_KEY',None)
            else:os.environ['OPENAI_API_KEY']=previous
            value['passed']=len(value['results'])==32 and all(r['passed']for r in value['results'])
            atomic_json(target,value)
        validated(a,version)
        semantic.verify_responses(sent,c.source_quality,a['batch'],version/'teacher-speech-quality.json',target,
            prepared_inputs=version/'prepared-semantic.json')
        return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('validate','review'))
    for n in ('inputs','version'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('out','env-file'):
        p.add_argument('--'+n,type=Path)
    a=p.parse_args();inputs=timing.load(a.inputs)
    if a.mode=='validate':
        _,sent,_=validated(inputs,a.version)
        print(json.dumps(dict(complete32=len(sent),unchanged30=True,new_requests_required=2,provider_requests=0,training_approved=False)))
    else:
        if a.out is None or a.env_file is None:p.error('Review needs a new output and existing local key file')
        v=asyncio.run(review(inputs,a.version,a.out,a.env_file))
        print(json.dumps(dict(complete=len(v['results']),passed=v['passed'],actual_new_requests=v['actual_new_requests_started'],training_approved=False)))
