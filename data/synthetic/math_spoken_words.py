"""Explicit numeric/UK orthography for a NEW audit version; old scores stay raw.

Do not equate homophones, remove repeated words, solve arithmetic expressions or
substitute synonyms. Symbols become spoken operators, so missing operators fail.
"""
import re
from .spoken_math import verbalize
from data.venus.asr import words as tokens

NUMBER_START='zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety'.split()


def words(text):
    s=verbalize(text).lower().replace('metres','meters').replace('litres','liters').replace('litre','liter')
    s=re.sub(r'\bper cent\b','percent',s)
    # Optional conjunctions inside numeric place-value notation only. Never
    # remove an ordinary conjunction or any item/value from the utterance.
    s=re.sub(r'\b(hundred|thousand|million|billion)\s+(?!and\b)(?=(?:'+ '|'.join(NUMBER_START)+r')\b)',r'\1 and ',s)
    return tokens(s)


def wer(reference,hypothesis):
    a,b=words(reference),words(hypothesis);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(row[j]+1,nxt[j-1]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]/max(1,len(a))
