"""Deterministic spoken notation for source maths, never a semantic correction."""
import re
from data.venus.context_source_quality import number as small_number


def integer(n):
    if not 0<=n<10**12:raise ValueError('Explicit numeric range exceeded')
    if n<1000000:return small_number(n)
    for size,name in [(10**9,'billion'),(10**6,'million')]:
        if n>=size:return integer(n//size)+' '+name+(' '+integer(n%size) if n%size else '')


def verbalize(text,*,spell_numbers=True):
    say=integer if spell_numbers else str
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',text)
    # Expand operators while numeric adjacency is still visible. A hyphen
    # inside an ordinary word stays a hyphen; 100-20 is subtraction.
    s=s.replace('×',' times ').replace('*',' times ').replace('+',' plus ').replace('=',' equals ').replace('%',' percent ')
    s=re.sub(r'(?<![A-Za-z])-(?![A-Za-z])',' minus ',s)
    s=re.sub(r'(?:(?<=\d)|\b)x(?=[ \t]*\d)',' times ',s)
    def money(m):
        major=int(m[2]);fraction=m[3] or '';minor=int(fraction.ljust(2,'0')) if fraction else 0
        unit=('pound' if major==1 else 'pounds') if m[1]=='£' else ('dollar' if major==1 else 'dollars')
        sub=('penny' if minor==1 else 'pence') if m[1]=='£' else ('cent' if minor==1 else 'cents')
        return say(major)+' '+unit+(' and '+say(minor)+' '+sub if minor else '')
    s=re.sub(r'([$£])(\d+)(?:\.(\d{1,2}))?(?!\d|\.\d)',money,s)
    s=re.sub(r'\b(\d{1,2}):([0-5]\d)\b',lambda m:integer(int(m[1]))+(" o'clock" if int(m[2])==0 else ' '+integer(int(m[2]))),s)
    if spell_numbers:
        s=re.sub(r'\b(\d+)\.(\d+)\b',lambda m:integer(int(m[1]))+' point '+' '.join(integer(int(c)) for c in m[2]),s)
        s=re.sub(r'\b\d+\b',lambda m:integer(int(m[0])),s)
    # Joined values/units are present in the public source (e.g.16marbles).
    s=re.sub(r'(?<![\w.])(\d+)(?=[A-Za-z])',lambda m:say(int(m[1]))+' ',s)
    s=s.replace('×',' times ').replace('*',' times ').replace('+',' plus ').replace('=',' equals ').replace('%',' percent ')
    s=re.sub(r'(?<=[A-Za-z])/(?=[A-Za-z])',' per ',s).replace('/',' divided by ')
    s=re.sub(r'(?<![A-Za-z])-(?![A-Za-z])',' minus ',s)
    s=re.sub(r'\bsq\s+in\b','square inches',s,flags=re.I)
    s=re.sub(r'[ \t]+',' ',s)
    return s.strip()
