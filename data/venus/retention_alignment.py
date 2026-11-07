"""Complete73 assistant-word alignment after independent full182 recognition.

Only original synthetic TRAIN WAVs and their authored words go to ElevenLabs.
This is conditioned timing measurement, never an independent word/semantic test.
"""
import argparse
import hashlib
import json
import os
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .retention_speech import prepare as speech_plan, sha
from .retention_quality import inventory
from .context_alignment import measured_units


def prepare(quality, original, resumed, payload):
    quality=Path(quality);q=json.loads(quality.read_text())
    if not q['complete'] or not q['all_sources_pass']:
        raise ValueError('Complete derived182 independent word review required')
    if q['required']!=182 or q['passed']!=182 or not q['all_lookup_keys_exact']:
        raise ValueError('Every original182 and all exact lookup keys required')
    verified=inventory(original,resumed,payload);by={r['id']:r for r in q['rows']}
    if len(by)!=182 or set(by)!={r['id'] for r in verified}:
        raise ValueError('No source subset')
    rows=[]
    for r in verified:
        old=by[r['id']]
        if not old['passed'] or old['source_sha256']!=r['source_sha256'] or old['text']!=r['text']:
            raise ValueError('Actual unchanged recognized source required')
        if r['profile']=='assistant-british':
            rows.append(dict(id=r['id'],text=r['text'],wav=r['wav'],
                source_sha256=r['source_sha256'],seconds=r['seconds'],fold='TRAIN'))
    if len(rows)!=73 or {r['id'] for r in rows}!={r['id'] for r in speech_plan()['requests'] if r['profile']=='assistant-british'}:
        raise ValueError('All73 original assistant targets required')
    upper=sum(max(60,r['seconds']) for r in rows)/3600
    return dict(schema='dl-retention96-exact73-alignment-payload-v1',rows=rows,maximum_calls=73,
        independent_quality_sha256=sha(quality),maximum_fee_reserve_usd=2.5,
        duration_billed_at_conservative_one_USD_per_hour_with_minute_floor_upper_usd=upper,
        official_published_duration_rate_USD_per_hour=.22,
        pricing_source='https://elevenlabs.io/pricing/api',
        endpoint_documentation='https://elevenlabs.io/docs/api-reference/forced-alignment/create',
        pricing_verified_utc='2026-11-07',project_envelope_upper_usd=165.7900361111111+175+6.373+2.5,
        existing_phase_reserve_unchanged_usd=175,prior_project_carry_unchanged_usd=165.7900361111111,
        no_purchase_or_overage=True,owner_conversations=False,human_ratings=False,
        payload_scope='Original generated synthetic TRAIN assistant WAV and exact authored text only; no scene/world/scoring metadata.',
        independent_recognition_already_complete=True,alignment_is_not_independent_recognition=True,
        training_approved=False)


def execute(payload,output,authority):
    payload,output,authority=map(Path,(payload,output,authority));p=json.loads(payload.read_text());a=json.loads(authority.read_text())
    if (a['status']!='OWNER_AUTHORIZED_ELEVENLABS_TRAINING_DATA_PROCESSING'
            or a['payload_sha256']!=sha(payload) or a['maximum_calls']!=73 or p['maximum_calls']!=73
            or a['maximum_fee_reserve_usd']!=2.5 or p['project_envelope_upper_usd']>350
            or p['duration_billed_at_conservative_one_USD_per_hour_with_minute_floor_upper_usd']>2.5):
        raise ValueError('Exact whole73 and preserved financial scope required')
    if a['maximum_calls']!=73 or p['maximum_calls']!=73 or len(p['rows'])!=73:
        raise ValueError('Exact73 requests required')
    if output.exists():raise ValueError('Observe existing attempt; never overwrite or repeat it')
    key=os.environ.get('ELEVENLABS_API_KEY')
    if not key:raise ValueError('Existing authorized key must stay local')
    req=urllib.request.Request('https://api.elevenlabs.io/v1/user/subscription',headers={'xi-api-key':key})
    with urllib.request.urlopen(req,timeout=60) as response:subscription=json.load(response)
    if subscription.get('can_extend_character_limit') is not False or subscription['character_limit']-subscription['character_count']<50000:
        raise ValueError('Existing prepaid quota and disabled overages required; no purchase')
    for row in p['rows']:
        if row['fold']!='TRAIN' or sha(row['wav'])!=row['source_sha256']:
            raise ValueError('Original synthetic TRAIN audio only')
    claim=Path('runs/isambard-native-20261107/retention96-speech-claims')/(sha(payload)+'-alignment.json')
    with claim.open('x') as f:json.dump(dict(payload_sha256=sha(payload),output=str(output.resolve())),f)
    output.mkdir(parents=True,exist_ok=False)
    ledger=dict(payload_sha256=sha(payload),attempts=[],complete=False,automatic_retries=0,
        prepaid_remaining_characters_before=subscription['character_limit']-subscription['character_count'],overages_disabled=True,
        maximum_fee_reserve_usd=2.5,new_purchase=False,training_approved=False)
    def save():
        tmp=output/'ledger.tmp';tmp.write_text(json.dumps(ledger,indent=2)+'\n');tmp.replace(output/'ledger.json')
    save();rows=[]
    for row in p['rows']:
        attempt=dict(id=row['id'],source_sha256=row['source_sha256'],status='RESERVED_OUTCOME_PENDING')
        ledger['attempts'].append(attempt);save()
        try:
            boundary='duplex-'+uuid.uuid4().hex
            body=(('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="original.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
                +Path(row['wav']).read_bytes()+('\r\n--'+boundary+'\r\nContent-Disposition: form-data; name="text"\r\n\r\n').encode()
                +row['text'].encode()+('\r\n--'+boundary+'--\r\n').encode())
            req=urllib.request.Request('https://api.elevenlabs.io/v1/forced-alignment',data=body,
                headers={'xi-api-key':key,'Content-Type':'multipart/form-data; boundary='+boundary})
            with urllib.request.urlopen(req,timeout=120) as response:
                raw=response.read();request_id=response.headers.get('request-id')
            path=output/(row['id']+'.json');path.write_bytes(raw)
            attempt.update(request_id=request_id,response_sha256=sha(path),response_received=True)
            response=json.loads(raw);units=measured_units(row['text'],response,row['seconds'])
            result=dict(id=row['id'],fold='TRAIN',text=row['text'],source_sha256=row['source_sha256'],
                measured_words=response['words'],native_speech_units=units,alignment_loss=response.get('loss'))
            rows.append(result);attempt['status']='COMPLETE';save()
            try:print(json.dumps(dict(completed=len(rows),required=73,id=row['id'])),flush=True)
            except OSError:pass
        except Exception as error:
            attempt.update(status='FAILED_OR_UNKNOWN_NO_RETRY',error_type=type(error).__name__);save()
            raise RuntimeError('Original response/attempt preserved; no automatic repeat') from None
    report=dict(complete=len(rows)==73,all73_exact_authored_word_alignments=True,rows=rows,
        payload_sha256=sha(payload),independent_source_review_sha256=p['independent_quality_sha256'],
        no_audio_changes=True,no_timestamp_interpolation=True,independent_semantic_review_pending=True,
        native_replay_and_cache_approval_pending=True,training_approved=False,
        actual_calls=len(ledger['attempts']),actual_billed_cost_not_verified=True,
        maximum_fee_reserve_usd=2.5,completed_utc=datetime.now(timezone.utc).isoformat())
    ledger.update(complete=True);save();(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','execute'))
    for name in ('payload','output','authority','quality','original','resumed','speech-payload'):
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.stage=='prepare':
        if args.payload.exists():raise ValueError('Preserve prior prepared payloads')
        result=prepare(args.quality,args.original,args.resumed,args.speech_payload)
        args.payload.write_text(json.dumps(result,indent=2)+'\n');print('Prepared all73 synthetic TRAIN alignments; no API requests')
    else:execute(args.payload,args.output,args.authority)
