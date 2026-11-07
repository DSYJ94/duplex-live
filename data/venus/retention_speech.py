"""Complete new96 synthetic TRAIN speech sources, exact text and private key.

No owner feedback/context/credentials enter request bodies. All originals and
failed attempts stay intact; one immutable batch claim prevents duplicate calls.
This produces unapproved data, never modifies the active frozen training corpus.
"""
import argparse,array,hashlib,json,math,os,urllib.request,urllib.error,wave
from datetime import datetime,timezone
from pathlib import Path

PLAN=Path('evals/fixtures/venus-retention-request-source-proposal-v2-20261107.json')
AUTH=Path('docs/NATIVE-RETENTION96-DATA-GENERATION-AUTHORITY-2026-11-07.json')
MODEL='eleven_v4'
VOICES={'user-british':'Xb7hH8MSUJpSbSDYk0k2','user-scottish':'eh9GHWIaYsZB3Alk8wrY','assistant-british':'JBFqnCBsd6RMkjVDRZzb'}
RATE_UPPER=.0005


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def prepare():
    plan=json.loads(PLAN.read_text())
    if len(plan['scenes'])!=96 or any(s['fold']!='TRAIN' or s.get('owner_conversations') for s in plan['scenes']):
        raise ValueError('Exact new96 authored TRAIN only required')
    unique={};scenes=[]
    def add(text,profile):
        key=(text,profile)
        if key not in unique:
            body=dict(text=text,model_id=MODEL,language_code='en',seed=20261107,
                      voice_settings=dict(stability=.5,similarity_boost=.75))
            ident=canonical(dict(voice_id=VOICES[profile],body=body))
            unique[key]=dict(id=ident,fold='TRAIN',profile=profile,voice_id=VOICES[profile],body=body,payload_sha256=canonical(body))
        return unique[key]['id']
    for s in plan['scenes']:
        profile='user-scottish' if s['family'] in ('delegate','pending') else 'user-british'
        users=[add(t,profile) for t in s['user_turns']]
        p=s['teacher_program'];spoken=[]
        for key in ('answer','result_answer','followup_answer','ordinary_answer','cancel_acknowledgment'):
            if p.get(key):spoken.append(dict(program_field=key,utterance_id=add(p[key],'assistant-british')))
        scenes.append(dict(id=s['id'],fold='TRAIN',user_utterance_ids=users,assistant_spoken_targets=spoken))
    rows=list(unique.values());characters=sum(len(r['body']['text']) for r in rows)
    if len(rows)!=182 or characters>30000:raise ValueError('Whole exact source collection required')
    return dict(schema='dl-retention96-speech-payload-v1',source_plan_sha256=sha(PLAN),requests=rows,
                scenes=scenes,required_scenes=96,required_requests=len(rows),characters=characters,
                conservative_cost_upper_usd=characters*RATE_UPPER,maximum_prepaid_character_debit=characters,
                maximum_additional_cash_charge_usd=0,no_overages_or_purchase=True,
                model_id=MODEL,original_sources_unchanged=True,owner_conversations=False,
                human_ratings_sent=False,score_annotations_sent=False,
                payload_scope='Exact authored user words and synthetic TRAIN assistant target words only; no scene metadata or worker-world dictionaries uploaded.',
                outputs_require_complete_source_ASR_peak_and_teacher_alignment=True,training_approved=False)


def execute(prepared,output):
    prepared=Path(prepared);p=json.loads(prepared.read_text());output=Path(output)
    if p!=prepare():raise ValueError('Exact committed prepared batch required')
    auth=json.loads(AUTH.read_text())
    if (auth['status']!='OWNER_INSTRUCTED_MORE_TRAINING_DATA' or auth['payload_sha256']!=sha(prepared)
            or auth['maximum_synthesis_calls']!=182 or auth['maximum_prepaid_character_debit']!=12746
            or auth['new_purchase_or_overage_allowed'] or auth['maximum_additional_cash_charge_usd']!=0
            or 165.7900361111111+175+p['conservative_cost_upper_usd']>350):
        raise ValueError('Exact new data instruction, whole prepaid batch and preserved project envelope required')
    key=os.environ.get('ELEVENLABS_API_KEY')
    if not key:raise ValueError('Use the already authorized existing local key; never save it in outputs')
    if output.exists():raise ValueError('Observe existing output; never restart or overwrite it')
    def request(path,body=None):
        headers={'xi-api-key':key}
        if body is not None:headers['Content-Type']='application/json'
        req=urllib.request.Request('https://api.elevenlabs.io/v1/'+path,headers=headers,
             data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(req,timeout=100) as response:return response.read(),response.headers.get('request-id')
        except urllib.error.HTTPError as e:raise RuntimeError('Speech HTTP '+str(e.code)+'; no automatic retry') from None
    subscription=json.loads(request('user/subscription')[0]);models=json.loads(request('models')[0])
    model=next(m for m in models if m['model_id']==MODEL and m['can_do_text_to_speech'])
    factor=model.get('token_cost_factor',1) or 1
    required=math.ceil(p['characters']*factor)
    if factor!=1 or subscription.get('can_extend_character_limit') is not False or subscription['character_limit']-subscription['character_count']<required:
        raise ValueError('Entire batch must fit existing prepaid quota with overages disabled; no purchase')
    for voice in sorted({r['voice_id'] for r in p['requests']}):
        v=json.loads(request('voices/'+voice)[0]);sharing=v.get('sharing') or {}
        if v['voice_id']!=voice or v.get('category') not in ('premade','professional') or sharing.get('rate',1)!=1:
            raise ValueError('Existing authorized stock/library voice required; no clone, addition or multiplier')
    claim_root=Path('runs/isambard-native-20261107/retention96-speech-claims');claim_root.mkdir(parents=True,exist_ok=True)
    with (claim_root/(sha(prepared)+'.json')).open('x') as f:
        json.dump(dict(payload_sha256=sha(prepared),output=str(output.resolve()),at=datetime.now(timezone.utc).isoformat()),f)
    output.mkdir(parents=True,exist_ok=False)
    ledger=dict(payload_sha256=sha(prepared),required=182,attempts=[],complete=False,
                automatic_retries=0,training_approved=False,source_model=MODEL,
                external_API_scope='New owner-authorized synthetic data production; active trainer/optimizer remain API0',
                maximum_additional_cash_charge_usd=0,prepaid_characters_reserved=required)
    def save():
        tmp=output/'ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');tmp.replace(output/'ledger.json')
    save()
    for row in p['requests']:
        attempt=dict(id=row['id'],status='RESERVED_OUTCOME_PENDING',payload_sha256=row['payload_sha256'],
                     voice_id=row['voice_id'],model_id=MODEL,characters=len(row['body']['text']),
                     conservative_usage_value_upper_usd=len(row['body']['text'])*RATE_UPPER)
        ledger['attempts'].append(attempt);save()
        try:
            raw,request_id=request('text-to-speech/'+row['voice_id']+'?output_format=pcm_24000',row['body'])
            folder=output/row['id'];folder.mkdir();(folder/'original.pcm').write_bytes(raw)
            if not raw or len(raw)%2:raise ValueError('Invalid original PCM; preserve it')
            wav=folder/'original.wav'
            with wave.open(str(wav),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(raw)
            values=array.array('h',raw);peak=max(abs(v) for v in values)/32768
            attempt.update(status='COMPLETE',request_id=request_id,original_pcm_sha256=hashlib.sha256(raw).hexdigest(),
                original_wav_sha256=sha(wav),seconds=len(values)/24000,peak=peak,peak_pass=peak<=.995,
                independent_words='PENDING_ISAMBARD_ASR',naturalness='PENDING',provenance_class='closed-content',
                provenance_authority='ADR0003 Amendment A; owner explicit new data instruction7Nov',training_approved=False)
            save();print(json.dumps(dict(completed=len(ledger['attempts']),required=182,id=row['id'],peak_pass=peak<=.995)),flush=True)
        except Exception as error:
            attempt.update(status='FAILED_OR_UNKNOWN_NO_RETRY',error_type=type(error).__name__);save()
            raise RuntimeError('Attempt preserved; never retry blindly or accept a subset') from None
    ledger.update(complete=True,conservative_cost_upper_usd=p['conservative_cost_upper_usd'],qualified=False)
    save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','execute'));p.add_argument('--prepared',required=True);p.add_argument('--output');a=p.parse_args()
    if a.stage=='prepare':
        out=Path(a.prepared)
        if out.exists():raise ValueError('Preserve previous prepared batches')
        result=prepare();out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('required_scenes','required_requests','characters','conservative_cost_upper_usd','maximum_additional_cash_charge_usd')}))
    else:execute(a.prepared,a.output)
