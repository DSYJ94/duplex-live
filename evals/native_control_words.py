"""Finite spelling comparison for authored control3 TRAIN speech only.

Raw text and raw WER stay recorded. No synonyms, names, cue substitutions or
recognizer swaps. This does not alter old49 or any sealed evaluation policy.
"""
from data.synthetic.filter import norm_words, wer

POLICY = 'control3-finite-spoken-orthography-v1'
VARIANTS = {'vapor': 'vapour'}


def spoken_words(text):
    return [VARIANTS.get(word, word) for word in norm_words(text)]


def agreement(reference, actual):
    return dict(policy=POLICY, variants=VARIANTS.copy(),
                raw_wer=wer(norm_words(reference), norm_words(actual)),
                canonical_word_wer=wer(spoken_words(reference), spoken_words(actual)))
