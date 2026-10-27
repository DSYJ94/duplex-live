"""Build one-second native speech units from independently aligned words."""
import math,re
from .repair import words as previous_words

def words(text):
    # Predeclared source-timing orthography, not phonetic correction. Whisper
    # commonly writes e.g.60 where the authored ordinary answer says sixty.
    # Applies only to these new timing sources; old audit scores stay intact.
    small='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
    tens={20:'twenty',30:'thirty',40:'forty',50:'fifty',60:'sixty',70:'seventy',80:'eighty',90:'ninety'}
    def number(n):
        if n<20:return [small[n]]
        if n==100:return ['one','hundred']
        return [tens[n//10*10]]+([small[n%10]] if n%10 else [])
    result=[]
    for token in previous_words(text):
        result.extend(number(int(token)) if token.isdigit() and len(token)<=3 and int(token)<=100 else [token])
    return result

def assistant_texts(plan):
    keys=('answer','release_answer','first_text','remaining_text','result_answer','followup_answer','cancel_acknowledgment','ordinary_answer')
    return sorted({p[k] for s in plan['scenes'] for p in [s['teacher_program']] for k in keys if k in p})

def units(text,aligned_words):
    spans=list(re.finditer(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?",text));observed=[]
    for row in aligned_words:
        start,end=row['timestamp']
        if start is None or end is None or not math.isfinite(start+end) or not 0<=start<end:raise ValueError('Complete finite word timings required')
        observed.extend((word,start,end) for word in words(row['text']))
    # Source WER approval alone cannot justify assigning a missing or different
    # word a time. Orthographic normalization is the existing fixed rule only.
    if [w[0] for w in observed]!=words(text) or len(spans)!=len(observed):
        raise ValueError('Every authored word needs its own exact aligned target')
    if any(observed[i+1][1]<observed[i][1] for i in range(len(observed)-1)):
        raise ValueError('Word onsets must be monotonic')
    last=max(math.floor(observed[-1][2]-1e-6),math.floor(observed[-1][1]));result=[]
    for frame in range(last+1):
        indices=[i for i,(_,start,end) in enumerate(observed) if math.floor(start)==frame]
        if indices:
            lo,hi=min(indices),max(indices);start=spans[lo].start();end=spans[hi+1].start() if hi+1<len(spans) else len(text)
            piece=text[start:end].strip()
        else:piece=''
        result.append(dict(offset=frame,text=piece,final=frame==last))
    assert words(' '.join(r['text'] for r in result))==words(text)
    return result
