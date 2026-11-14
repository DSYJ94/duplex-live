"""Literal source preparation for diagnosed synthetic utterance defects.

This changes teaching scripts before rendering. It never edits an ASR response,
adds phonetic aliases to a scorer, or changes a mathematical result.
"""
import re
from .spoken_math import integer,verbalize


def ordinal(n):
    special={1:'first',2:'second',3:'third',5:'fifth',8:'eighth',9:'ninth',12:'twelfth'}
    if n in special:return special[n]
    if n<20:return integer(n)+'th'
    if n<100:
        return integer(n)[:-1]+'ieth' if n%10==0 else integer(n//10*10)+' '+ordinal(n%10)
    return integer(n)+'th'


def prepare(text):
    s=text.replace('’',"'").replace('‘',"'")
    s=re.sub(r'\b(\d+)(?:st|nd|rd|th)\b',lambda m:ordinal(int(m[1])),s)
    s=re.sub(r'(\d+/\d+)ths\b',r'\1',s)
    s=re.sub(r'(?<=\d)\s*\(', ' times (',s)
    s=s.replace('−',' minus ').replace('–',' minus ')
    s=re.sub(r'\bcm³\b','cubic centimetres',s)
    for short,long in [('sq ft','square feet'),('mph','miles per hour'),('kgs','kilograms'),('kg','kilograms'),('mg','milligrams'),('ml','millilitres'),('oz','ounces'),('cm','centimetres'),('mm','millimetres'),('km','kilometres'),('g','grams'),('m','metres')]:
        s=re.sub(r'(?<![\w.])'+re.escape(short)+r'\b',long,s)
    s=re.sub(r'\bper (kilograms|grams|milligrams|metres|millilitres|ounces)\b',lambda m:'per '+m[1][:-1],s)
    # These are source spelling/spacing defects, not scoring equivalences.
    for old,new in [('co-workers','coworkers'),('see-saw','seesaw'),('bookbag','book bag'),('bodyweight','body weight'),('cash back','cashback'),('key chains','key rings'),('key chain','key ring'),('MegaCorp','Mega Corporation'),('Lake View','Lakeview'),('payed','paid'),("bring there's",'bring theirs'),('cuts her law ','cuts her lawn ')]:s=s.replace(old,new)
    s=s.replace('&',' and ')
    s=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',s)
    s=verbalize(s)
    return re.sub(r'[ \t]+',' ',s).strip()
