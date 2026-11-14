"""Measure all1072 authored assistant timelines after independent word approval.

Alignment receives audio plus authored text; it is not a recognition verdict.
No retries, interpolation, trimming or changed folds are allowed.
"""
import argparse, concurrent.futures, hashlib, json, math, re, threading, urllib.request, uuid
from pathlib import Path



def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_words(text,response,seconds):
    characters=response.get('characters')
    if characters is not None:
        if any(len(c['text'])!=1 for c in characters) or ''.join(c['text']for c in characters)!=text:raise ValueError('Exact authored character sequence required')
        # Some provider word entries group parenthesized expressions. Its
        # CHARACTER endpoints are also measured: take exact endpoints, never
        # divide a group duration or interpolate a word boundary.
        spans=list(re.finditer(r'[^\s:]+:?',text))
        words=[dict(text=m[0],start=characters[m.start()]['start'],end=characters[m.end()-1]['end'],measurement='provider-character-endpoints')for m in spans]
    else:
        words=[dict(w)for w in response['words'] if w['text'].strip()]
    # Provider receives the authored script. Preserve its measured words as
    # indivisible labels: 22 is ONE authored numeric token, not two invented
    # word intervals. The native interleaver owns tokenizer subword placement.
    if [w['text'].strip()for w in words] != ([m[0]for m in re.finditer(r'[^\s:]+:?',text)] if characters is not None else text.split()):raise ValueError('Every exact authored word needs its own measured interval')
    previous=0.
    for w in words:
        lo,hi=w['start'],w['end']
        if not all(isinstance(x,(int,float)) and math.isfinite(x)for x in (lo,hi)) or not 0<=lo<=hi<=seconds+.02 or lo<previous:raise ValueError('Actual finite monotonic source-bound intervals required')
        previous=lo
    if not words or words[-1]['end']<=words[-1]['start']:raise ValueError('Positive measured final word ending required')
    return words


def main():
    p=argparse.ArgumentParser()
    for n in ('plan','quality','semantics','inventory','audio','cost-ledger','key-file','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reuse-from',type=Path)
    p.add_argument('--reuse-verified',type=Path)
    a=p.parse_args();plan=json.loads(a.plan.read_text());quality=json.loads(a.quality.read_text());sem=json.loads(a.semantics.read_text());inv=json.loads(a.inventory.read_text());cost=json.loads(a.cost_ledger.read_text());pin=sha(a.plan)
    if not plan['owner_approved'] or plan['maximum_total_API_USD']!=100 or not quality['complete'] or not quality['all_passed'] or len(quality['rows'])!=2144 or quality['source_plan_sha256']!=pin:raise ValueError('Full exact2144 independent word approval required')
    if not sem['complete1072'] or not sem['all_passed'] or len(sem['rows'])!=1072 or sem['source_plan_sha256']!=pin:raise ValueError('Full exact1072 semantic approval required')
    q={r['id']:r for r in quality['rows']};inventory={r['id']:r for r in inv['rows']};payload={r['id']:r for r in plan['exact_speech_payloads']};rows=[]
    if len(q)!=2144 or set(q)!=set(payload) or set(q)!=set(inventory):raise ValueError('Exact complete source membership required')
    for scene in plan['exact_semantic_records']:
        ident=scene['id'];key=ident+'-assistant';r=q[key];v=inventory[key]
        if not r['word_pass'] or not r['peak_pass'] or not r['timing_pass'] or r['source_words']!=scene['response'] or r['fold']!=scene['fold'] or sha(a.audio/(key+'.wav'))!=r['source_sha256'] or v['sha256']!=r['source_sha256']:raise ValueError('Qualified unchanged assistant ownership required')
        rows.append(dict(id=ident,text=scene['response'],fold=scene['fold'],source_sha256=r['source_sha256'],seconds=v['seconds'],file=key+'.wav'))
    if len(rows)!=1072 or len({r['id'] for r in rows})!=1072:raise ValueError('Every1072 assistant required')
    reused={}
    if a.reuse_verified:
        report_path=a.reuse_verified/'report.json';previous=json.loads(report_path.read_text())
        if not previous['complete1072'] or not previous['all_authored_words_measured'] or previous['source_plan_sha256']!=pin or len(previous['rows'])!=1072 or len({r['id']for r in previous['rows']})!=1072:raise ValueError('Complete exact-plan passing alignment parent required')
        old={r['id']:r for r in previous['rows']}
        if set(old)!={r['id']for r in rows}:raise ValueError('Exact whole1072 alignment membership required')
        for row in rows:
            prior=old[row['id']]
            if any(prior[k]!=row[k]for k in ('id','fold','text','source_sha256')):continue
            spans=[w for w in prior['measured_words'] if w['text'].strip()]
            if [w['text']for w in spans]!=[m[0]for m in re.finditer(r'[^\s:]+:?',row['text'])]:raise ValueError('Prior authored measurement sequence changed')
            previous_start=0.
            for w in spans:
                if not all(isinstance(w[k],(int,float)) and math.isfinite(w[k])for k in ('start','end')) or not 0<=w['start']<=w['end']<=row['seconds']+.02 or w['start']<previous_start:raise ValueError('Prior measured endpoints changed')
                previous_start=w['start']
            reused[row['id']]=dict(prior,reused_alignment_report_sha256=sha(report_path))
    if a.reuse_from:
        history=[];cursor=a.reuse_from
        while cursor is not None:
            if cursor.resolve() in {x.resolve()for x in history}:raise ValueError('Cyclic recovery lineage')
            parent=json.loads((cursor/'admission.json').read_text())
            if parent['source_plan_sha256']!=pin or parent['quality_sha256']!=sha(a.quality) or parent['semantic_sha256']!=sha(a.semantics) or parent['maximum_calls']+parent.get('reused_verified_responses',0)!=1072 or parent['carried_usage_upper_USD']!=cost['combined_usage_upper_USD']:raise ValueError('Exact original complete-cohort reservation required')
            history.append(cursor);cursor=Path(parent['reservation_shared_with_parent']) if parent.get('reservation_shared_with_parent') else None
        for row in rows:
            found=[]
            for folder in history:
                response=folder/(row['id']+'-response.json');inflight=folder/(row['id']+'-inflight.json')
                if inflight.exists() and json.loads(inflight.read_text())!=row:raise ValueError('Previous request source identity changed')
                if inflight.exists() and not response.exists():raise ValueError('Unknown previous request cannot be repeated')
                if response.exists():
                    if not inflight.exists():raise ValueError('Response without original source receipt')
                    found.append(response)
            if found:
                if len({sha(f)for f in found})!=1:raise ValueError('Multiple distinct responses cannot be selected for passing')
                response=found[0];raw=json.loads(response.read_text());reused[row['id']]=dict(row,measured_words=measured_words(row['text'],raw,row['seconds']),alignment_loss=raw.get('loss'),reused_response_sha256=sha(response))
    pending=[r for r in rows if r['id']not in reused]
    reserve=json.loads((history[-1]/'admission.json').read_text())['reservation_USD'] if a.reuse_from else sum(max(60,r['seconds'])for r in pending)/3600
    if cost['combined_usage_upper_USD']+reserve>100:raise ValueError('Whole alignment plus all prior reserves must fit100USD')
    secret=a.key_file.read_text();match=re.search(r'eleven\s*labs',secret,re.I)
    if not match:raise ValueError('Authorized credential section missing')
    keys=re.findall(r'(?<![A-Za-z0-9])(sk_[A-Za-z0-9]{20,}|[a-f0-9]{32,64})(?![A-Za-z0-9])',secret[match.start():match.start()+800])
    if len(set(keys))!=1:raise ValueError('Ambiguous credential')
    key=keys[0]
    with urllib.request.urlopen(urllib.request.Request('https://api.elevenlabs.io/v1/user/subscription',headers={'xi-api-key':key}),timeout=30) as response:sub=json.load(response)
    if sub.get('max_credit_limit_extension')!=0 or sub['character_limit']-sub['character_count']<20000:raise ValueError('Included quota and disabled overage required')
    a.out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (a.out/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write('admission.json',dict(source_plan_sha256=pin,quality_sha256=sha(a.quality),semantic_sha256=sha(a.semantics),maximum_calls=len(pending),reused_verified_responses=len(reused),verified_parent_report_sha256=sha(a.reuse_verified/'report.json') if a.reuse_verified else None,reservation_shared_with_parent=str(a.reuse_from) if a.reuse_from else None,carried_usage_upper_USD=cost['combined_usage_upper_USD'],reservation_USD=reserve,combined_reservation_USD=cost['combined_usage_upper_USD']+reserve,cap_USD=100,conservative_hourly_USD=1,minimum_reserved_seconds=60,price_source='https://elevenlabs.io/docs/overview/capabilities/forced-alignment',pricing='same as speech-to-text; conservatively reserved at1USD/hour',automatic_retries=0,concurrency=2,alignment_is_not_independent_recognition=True))
    stop=threading.Event()
    def one(row):
        if stop.is_set():return None
        ident=row['id'];write(ident+'-inflight.json',row);boundary='duplex-'+uuid.uuid4().hex
        body=(('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="synthetic.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+(a.audio/row['file']).read_bytes()+('\r\n--'+boundary+'\r\nContent-Disposition: form-data; name="text"\r\n\r\n').encode()+row['text'].encode()+('\r\n--'+boundary+'--\r\n').encode())
        try:
            request=urllib.request.Request('https://api.elevenlabs.io/v1/forced-alignment',data=body,headers={'xi-api-key':key,'Content-Type':'multipart/form-data; boundary='+boundary})
            with urllib.request.urlopen(request,timeout=180) as response:raw=json.load(response);request_id=response.headers.get('request-id')
            write(ident+'-response.json',raw);words=measured_words(row['text'],raw,row['seconds'])
            result=dict(row,measured_words=words,alignment_loss=raw.get('loss'),request_id=request_id);write(ident+'.json',result);return result
        except Exception as exc:
            write(ident+'-error.json',dict(error_type=type(exc).__name__,no_retry=True));stop.set();return None
    for ident,result in reused.items():write(ident+'.json',result)
    first=one(pending[0]) if pending else None;results=list(reused.values())+([first]if first else [])
    if first:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(one,pending[1:]):
                if result:results.append(result)
                if len(results)%25==0:print(json.dumps(dict(completed=len(results),required=1072)),flush=True)
    report=dict(source_plan_sha256=pin,complete1072=len(results)==1072,all_authored_words_measured=len(results)==1072,rows=results,original_audio_unchanged=True,timestamps_interpolated=False,native_training_approved=False)
    write('report.json',report)
    if not report['complete1072']:raise RuntimeError('Incomplete alignment preserved; no blind retry')


if __name__=='__main__':main()
