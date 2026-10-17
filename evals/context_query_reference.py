"""Bind a real backend response to its actual arrived-audio ASR context."""
import hashlib
import math


ASR_SOURCE='actual arrived user PCM; Whisper timed recognition, not reference words'


def compose(reference,context,input_available_s,route):
    if route=='error':return reference,{'supplied':False,'reason':'recognition/backend failure remains unverified'}
    if not math.isfinite(input_available_s) or input_available_s<=0:
        raise RuntimeError('Query anchor needs the actual arrived-audio cutoff')
    users=[e for e in context if e['role']=='user'][-3:]
    if not users:raise RuntimeError('Query anchor has no actual recognized user speech')
    for e in users:
        if (e.get('source')!=ASR_SOURCE or not isinstance(e.get('text'),str) or not e['text'].strip()
                or not all(math.isfinite(e[k]) for k in ('start_s','end_s'))
                or not 0<=e['start_s']<e['end_s']<=input_available_s+.04):
            raise RuntimeError('Query anchor cannot contain reference, invalid or future user words')
    if sum(len(e['text']) for e in users)>4000:
        raise RuntimeError('Query anchor exceeds the bounded full-text context; no silent truncation')
    lines=['User speech recognized before this backend request:']
    for i,e in enumerate(users):
        label='Latest user speech' if i==len(users)-1 else 'Earlier user speech'
        lines.append(label+': '+e['text'].strip())
    query='\n'.join(lines)
    return query+'\nBackend reply for this request:\n'+reference,{
        'supplied':True,'source':ASR_SOURCE,'input_available_s':input_available_s,
        'asr_timestamp_tolerance_s':.04,'user_entries':len(users),
        'query_sha256':hashlib.sha256(query.encode()).hexdigest(),
        'reference_transcript_supplied':False,'spoken_tokens_forced':False}
