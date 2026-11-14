"""Complete independent Scribe3 word audit; no reference text or bias supplied.

A fixed whole-cohort protocol, not selection of passing records from different
recognizers. Earlier Whisper results remain immutable diagnostics. No retries,
transcript editing, keyterms, non-verbatim filtering, or waveform changes.
"""
import argparse,concurrent.futures,hashlib,json,math,re,threading,urllib.request,uuid
from pathlib import Path
from . import math_spoken_words
from .math_spoken_words import wer


POLICY='math-spoken-v3-typographic-apostrophe-speed-unit'
FIELDS=dict(model_id='scribe_v3',language_code='en',numbers_format='words',timestamps_granularity='word',diarize='false',tag_audio_events='false',no_verbatim='false',temperature='0',seed='42424242')


def reusable(row, previous):
    return bool(previous and previous['id']==row['id'] and previous['source_sha256']==row['sha256'] and previous['source_words']==row['source_text'] and previous['fold']==row['fold'])


def main():
    p=argparse.ArgumentParser()
    for n in ('plan','audio','inventory','cost-ledger','key-file','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reuse-from',type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text());inventory=json.loads(a.inventory.read_text());cost=json.loads(a.cost_ledger.read_text())
    if not plan['owner_approved'] or plan['maximum_total_API_USD']!=100 or not inventory['complete'] or len(inventory['rows'])!=2144 or not inventory['all_raw_PCM_checks_pass']:raise ValueError('Full approved2144 unclipped source cohort required')
    sources={r['id']:r for r in plan['exact_speech_payloads']};rows=inventory['rows']
    if len(sources)!=2144 or {r['id'] for r in rows}!=set(sources):raise ValueError('Exact whole source membership required')
    for row in rows:
        if row['source_text']!=sources[row['id']]['text'] or row['sha256']!=hashlib.sha256((a.audio/row['file']).read_bytes()).hexdigest():raise ValueError('Source hash/text changed')
    reused=[]; pending_rows=[]
    if a.reuse_from:
        parent=json.loads((a.reuse_from/'report.json').read_text())
        admission=json.loads((a.reuse_from/'admission.json').read_text())
        if not parent['complete'] or parent['requested']!=2144 or parent['recognizer']!='scribe_v3' or admission['protocol']!='complete2144_scribe3_no_reference_or_bias_v1' or admission['fields']!=FIELDS:raise ValueError('Complete same-protocol parent required')
        old={r['id']:r for r in parent['rows']}
        if len(old)!=2144:raise ValueError('Parent membership differs')
        for row in rows:
            prev=old.get(row['id'])
            if reusable(row,prev):
                # Recompute every score under this source version. Never reuse a
                # verdict across changed PCM, words, folds or recognition policy.
                if prev['recognizer']!='scribe_v3' or prev['word_policy'] not in ('math-spoken-v2-sentence-currency-unit-fix',POLICY):raise ValueError('Parent word protocol differs')
                result=dict(prev);result['word_policy']=POLICY;result['WER']=wer(row['source_text'],prev['recognized']);result['word_pass']=result['WER']<=.05;result['peak_pass']=row['raw_peak_pass'];reused.append(result)
            else:pending_rows.append(row)
    else:pending_rows=rows
    reserve=sum(max(60,r['seconds']) for r in pending_rows)/3600
    if cost['combined_usage_upper_USD']+reserve>100:raise ValueError('Entire cohort plus prior unknown calls exceeds100USD')
    secret=a.key_file.read_text();match=re.search(r'eleven\s*labs',secret,re.I);keys=re.findall(r'(?<![A-Za-z0-9])(sk_[A-Za-z0-9]{20,}|[a-f0-9]{32,64})(?![A-Za-z0-9])',secret[match.start():match.start()+800]);assert len(set(keys))==1;key=keys[0]
    with urllib.request.urlopen(urllib.request.Request('https://api.elevenlabs.io/v1/user/subscription',headers={'xi-api-key':key}),timeout=30) as response:sub=json.load(response)
    if sub.get('max_credit_limit_extension')!=0 or sub['character_limit']-sub['character_count']<20000:raise ValueError('Existing prepaid quota and disabled overage required')
    a.out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (a.out/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    fields=dict(FIELDS)
    write('admission.json',dict(word_policy_source_sha256=hashlib.sha256(Path(math_spoken_words.__file__).read_bytes()).hexdigest(),required=2144,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),protocol='complete2144_scribe3_no_reference_or_bias_v1',fields=fields,word_policy=POLICY,WER_limit=.05,peak_limit=.995,carried_usage_upper_USD=cost['combined_usage_upper_USD'],reservation_USD=reserve,combined_reservation_USD=cost['combined_usage_upper_USD']+reserve,published_hourly_USD=.2,conservative_hourly_reservation_USD=1,conservative_minimum_seconds_per_request=60,price_source='https://elevenlabs.io/pricing/api',included_credits_before=sub['character_limit']-sub['character_count'],overage_disabled=True,maximum_calls=len(pending_rows),reused_exact_records=len(reused),parent_report_sha256=hashlib.sha256((a.reuse_from/'report.json').read_bytes()).hexdigest() if a.reuse_from else None,concurrency=2,automatic_retries=0,owner_conversations=False,reference_text_sent=False,earlier_Whisper_reports_preserved=True,native_training_approved=False))
    stop=threading.Event()
    def one(row):
        if stop.is_set():return None
        ident=row['id'];write(ident+'-inflight.json',dict(source_sha256=row['sha256'],reserved=True))
        boundary='duplex-'+uuid.uuid4().hex
        parts=[]
        for name,value in fields.items():parts.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+name+'"\r\n\r\n'+value+'\r\n').encode())
        parts.append(('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="synthetic.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+(a.audio/row['file']).read_bytes()+('\r\n--'+boundary+'--\r\n').encode())
        req=urllib.request.Request('https://api.elevenlabs.io/v1/speech-to-text',data=b''.join(parts),headers={'xi-api-key':key,'Content-Type':'multipart/form-data; boundary='+boundary})
        try:
            with urllib.request.urlopen(req,timeout=300) as response:raw=json.load(response);request_id=response.headers.get('request-id')
            write(ident+'-response.json',raw);error=wer(row['source_text'],raw['text']);ws=[w for w in raw.get('words',[]) if w.get('type')=='word'];spans=[(w['start'],w['end']) for w in ws]
            timing=bool(spans) and all(all(isinstance(x,(int,float)) and math.isfinite(x) for x in t) and 0<=t[0]<=t[1]<=row['seconds']+.02 for t in spans)
            result=dict(id=ident,fold=row['fold'],source_sha256=row['sha256'],source_words=row['source_text'],recognized=raw['text'],words=[dict(text=w['text'],timestamp=[w['start'],w['end']])for w in ws],WER=error,word_pass=error<=.05,peak_pass=row['raw_peak_pass'],timing_pass=timing,request_id=request_id,recognizer='scribe_v3',word_policy=POLICY,native_training_approved=False)
            write(ident+'.json',result);return result
        except Exception as exc:
            write(ident+'-error.json',dict(error_type=type(exc).__name__,status='failed_or_unknown',retry=False));stop.set();return None
    first=one(pending_rows[0]) if pending_rows else None;results=reused+([first] if first else [])
    if first:

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(one,pending_rows[1:]):
                if result:results.append(result)
                if len(results)%25==0:print(json.dumps(dict(completed=len(results),required=2144)),flush=True)
    report=dict(source_plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),complete=len(results)==2144,requested=2144,passed=sum(r['word_pass'] and r['peak_pass'] and r['timing_pass'] for r in results),all_passed=len(results)==2144 and all(r['word_pass'] and r['peak_pass'] and r['timing_pass'] for r in results),recognizer='scribe_v3',protocol='complete2144_scribe3_no_reference_or_bias_v1',rows=results,native_training_approved=False,independent_semantic_and_authored_word_alignment_still_required=True)
    write('report.json',report)
    if not report['complete']:raise RuntimeError('Incomplete source audit preserved; no retry/subset acceptance')


if __name__=='__main__':main()
