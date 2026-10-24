"""Reapprove complete32 with two independently diagnosed word-clock repairs.

Original recordings, words, provider deltas, receipts and old audits are frozen.
All32 original speech/transport checks run remotely; only029/030 use independent
per-region decoding. This module prepares evidence, never sends provider calls.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from data.synthetic.filter import norm_words
from evals import native_reasoning_burst_asr as burst
from evals import native_reasoning_semantic as semantic
from evals import native_reasoning_semantic_prepared as original
from evals import native_reasoning_source_quality as quality
from evals import native_reasoning_teacher_mixed as mixed
from evals import native_reasoning_teacher_mixed_speech as speech
from evals import native_reasoning_teacher_speech_v2 as region
from evals.native_reasoning_source import sha
from evals.native_reasoning_verification_session import source_readbacks

IDS=tuple(f'native-reasoning4-train-{i:03d}'for i in (29,30))
SCHEMA='dl-native-reasoning32-two-diagnosed-timing-reapproval-v1'


def load(path):
    v=json.loads(path.read_text())
    expected={'config','batch','old_speech','old_prepared','failed_semantic','burst','burst_plan'}
    if set(v)!=expected:
        raise ValueError('Explicit original approval and independent clock lineage required')
    return {k:Path(p)for k,p in v.items()}


def checked_failure(a,sent):
    r=json.loads(a['failed_semantic'].read_text())
    if (r['schema']!=semantic.SCHEMA or r['model']!=semantic.MODEL
            or r['status']!='complete_independent_review' or r['passed']is not False
            or [x['id']for x in r['results']]!=mixed.IDS
            or tuple(x['id']for x in r['results']if not x['passed'])!=IDS
            or r['prepared_inputs_sha256']!=sha(a['old_prepared'])
            or r['speech_approval_sha256']!=sha(a['old_speech'])
            or r.get('failures') or r['automatic_retry']is not False):
        raise ValueError('Exact complete32 independent failure029/030 required')
    for row in r['results']:
        p=a['failed_semantic'].parent/row['response_file'];data=json.loads(p.read_text())
        calls=[x for x in data['output']if x['type']=='function_call']
        if (row['input_sha256']!=semantic.canonical_hash(sent[row['id']])
                or row['response_sha256']!=sha(p) or row['response_id']!=data['id']
                or data['model']!=semantic.MODEL or data['status']!='completed'
                or len(calls)!=1 or calls[0]['name']!=semantic.TOOL['name']
                or json.loads(calls[0]['arguments'])!=dict(verdict='pass'if row['passed']else'fail',reason=row['reason'])):
            raise ValueError('Changed independent original semantic response')
    return r


def checked_bursts(a,qualified):
    state=json.loads((a['burst']/'recognition.json').read_text())
    plan=json.loads(a['burst_plan'].read_text())
    if (state['schema']!=burst.SCHEMA or state['phase']!='complete_diagnostic_not_approval'
            or state['code_sha256']!=sha(Path(burst.__file__)) or state['expected']!=2
            or state['plan_sha256']!=sha(a['burst_plan'])
            or state['region_policy']!=region.POLICY or state['asr_weight_sha256']!=region.WEIGHT
            or state['asr_packages']!=region.PACKAGES or state['provider_requests']!=0
            or any(state[k]is not False for k in ('original_audio_changed','reference_words_used','whole32_approved','semantic_approved','training_approved'))
            or tuple(r['id']for r in state['results'])!=IDS
            or tuple(r['id']for r in plan['recordings'])!=IDS):
        raise ValueError('Only two completed independent original-waveform timing probes allowed')
    old={r['id']:r for r in qualified['results']};new={}
    for row,planned in zip(state['results'],plan['recordings']):
        identity=row['id'];p=a['burst']/row['asr_file']
        if (row['wav_sha256']!=planned['wav_sha256']
                or row['wav_sha256']!=old[identity]['original_rendered_wav_sha256']
                or row['asr_file']!=identity+'-asr.json' or sha(p)!=row['asr_sha256']):
            raise ValueError('Independent timing probe WAV or decoder output changed')
        if [[r['first_sample'],r['last_sample']]for r in row['regions']]!=[[round(x*24000),round(y*24000)]for x,y in row['original_regions']]:
            raise ValueError('Every complete waveform region must be decoded exactly once')
        reconstructed=[];texts=[]
        for i,record in enumerate(row['regions']):
            raw_path=a['burst']/(identity+'-region-%02d.json'%i)
            if sha(raw_path)!=record['raw_sha256']:
                raise ValueError('Independent original-region decoder output changed')
            raw=json.loads(raw_path.read_text())
            restored=burst.restore(raw,record['first_sample'],record['last_sample'],24000,len(reconstructed))
            reconstructed.extend(restored.get('segments',[]));texts.append(raw['text'])
        actual=json.loads(p.read_text())
        if (actual!=dict(text=' '.join(texts),segments=reconstructed)
                or norm_words(actual['text'])!=norm_words(old[identity]['asr_text'])):
            raise ValueError('Clock repair must preserve all independently recognized words')
        new[identity]=(actual,row)
    return new


def changed_messages(sent,qualified):
    result=deepcopy(sent);rows={r['id']:r for r in qualified['results']}
    for identity in IDS:
        payload=json.loads(result[identity][1]['content'])
        payload['independently_measured_assistant_words']=rows[identity]['measured_words']
        payload['independent_asr_text']=rows[identity]['asr_text']
        result[identity][1]['content']=json.dumps(payload,sort_keys=True)
    if any(result[k]!=sent[k]for k in sent if k not in IDS):
        raise ValueError('Passing30 semantic inputs must remain byte-identical')
    return result


def finalize(a,out):
    import soundfile as sf
    c=mixed.load(a['config'])
    roots=[*mixed.input_roots(c),a['batch'],a['old_speech'].parent,
        a['old_prepared'],a['failed_semantic'].parent,a['burst'],a['burst_plan']]
    if any(out.resolve().is_relative_to(r.resolve())for r in roots if r.is_dir()):
        raise ValueError('New derived version must be outside all immutable input roots')
    out.mkdir(parents=True,exist_ok=False)
    with source_readbacks(roots,include_old_speech=True):
        header,_,_=mixed.verify(c,a['batch'])
        old=json.loads(a['old_speech'].read_text())
        speech.verify_value(c,a['batch'],old,speech.rows(c,a['batch'],header))
        sent=original.validate(c,a['batch'],a['old_speech'],a['old_prepared'])
        checked_failure(a,sent);new=checked_bursts(a,old)
        results=deepcopy(old['results'])
        for row in results:
            if row['id']not in new:
                continue
            raw,binding=new[row['id']];folder=a['batch']/row['id']
            pcm,rate=sf.read(folder/'model_out.wav',dtype='float32')
            if region.regions(pcm,rate)!=binding['original_regions']:
                raise ValueError('Waveform-only complete region boundaries changed')
            verdict=quality.assess(row['actual_provider_transcript'],raw,pcm,rate)
            if verdict!=quality.assess(row['actual_provider_transcript'],json.loads((a['burst']/binding['asr_file']).read_text()),pcm,rate):
                raise ValueError('Independent timing decisions must reproduce exactly twice')
            if not verdict['passed']:
                raise ValueError('Unchanged speech thresholds still apply to each repaired clock')
            row.update(verdict)
            row['recognition_binding']=dict(kind='independent-burst-clock-v11',
                wav_sha256=binding['wav_sha256'],asr_sha256=binding['asr_sha256'],
                recognition_sha256=sha(a['burst']/'recognition.json'))
        assert len(results)==32 and all(r['passed']for r in results)
        value=dict(schema=SCHEMA+'-speech',code_sha256=sha(Path(__file__)),
            at_utc=datetime.now(timezone.utc).isoformat(),complete_cases=32,results=results,passed=True,
            old_speech_sha256=sha(a['old_speech']),old_prepared_sha256=sha(a['old_prepared']),
            failed_semantic_sha256=sha(a['failed_semantic']),
            independent_clock_recognition_sha256=sha(a['burst']/'recognition.json'),
            repaired_clock_ids=list(IDS),original_audio_words_receipts_folds_changed=False,
            passing30_redecoded=False,passing_subset_accepted=False,semantic_approved=False,training_approved=False)
        region.write(out/'teacher-speech-quality.json',value)
        messages=changed_messages(sent,value)
        prepared=dict(schema=SCHEMA+'-prepared',code_sha256=sha(Path(__file__)),
            old_prepared_sha256=sha(a['old_prepared']),speech_sha256=sha(out/'teacher-speech-quality.json'),
            failed_semantic_sha256=sha(a['failed_semantic']),model=semantic.MODEL,
            messages=messages,input_sha256={k:semantic.canonical_hash(v)for k,v in messages.items()},
            new_review_ids=list(IDS),unchanged30_inputs_exact=True,maximum_new_requests=2,
            maximum_concurrency=2,original_audio_words_receipts_folds_changed=False,
            complete32=True,semantic_approved=False,training_approved=False)
        region.write(out/'prepared-semantic.json',prepared)
    proof=dict(schema=SCHEMA+'-readback',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        complete32_original_transport_and_speech_checked=True,two_new_speech_decisions_recomputed_twice=True,
        unchanged_input_snapshot=True,all32_speech_passed=True,
        speech_sha256=sha(out/'teacher-speech-quality.json'),prepared_sha256=sha(out/'prepared-semantic.json'),
        provider_requests=0,semantic_approved=False,training_approved=False)
    region.write(out/'independent-readback.json',proof)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();v=finalize(load(a.inputs),a.out)
    print(json.dumps(dict(complete32=v['complete_cases'],speech_passed=v['passed'],provider_requests=0,training_approved=False)))
