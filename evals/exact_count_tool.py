"""Bounded terminal count action; the model selects inputs, code computes the value.

No eval oracle, arbitrary code, filesystem, network or private-record access.
Quotes bind numerical arguments to the actual original question. They do not
prove the model selected the intended operation or the complete intended list.
"""
import hashlib
import json
import re
from data.synthetic.filter import norm_words


def quote_words(quote):
    if re.search(r'[^A-Za-z0-9\s,\[\]();-]', quote):
        raise ValueError('Unsupported numeric quote notation')
    # WER normalization erases signs and decimal punctuation. Never use that
    # erasure to approve numeric inputs. Only conventional tens-word hyphens
    # are allowed; signs, ranges, fractions and expressions stay unsupported.
    phrase = re.sub(r'\b(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-(?:one|two|three|four|five|six|seven|eight|nine)\b',
                    lambda m: m.group().replace('-', ' '), quote, flags=re.I)
    if '-' in phrase:
        raise ValueError('Signed numbers and ranges are not positive-integer lists')
    return [word for word in norm_words(phrase) if word != 'and']


def bound_quote(question, quote):
    for match in re.finditer(re.escape(quote), question):
        before, after = question[:match.start()], question[match.end():]
        if quote[0].isalnum():
            if before and before[-1].isalnum():
                continue
            if re.search(r'(?:[-+./−]|\bminus|\bnegative)\s*$', before, re.I):
                continue
        if quote[-1].isalnum():
            if after and after[0].isalnum():
                continue
            if re.match(r'\s*[./−+-]\s*\d', after):
                continue
        return True
    return False


def integer_words(number):
    if type(number) is not int or not 0 <= number <= 2**24:
        raise ValueError('Count outside declared range')
    small = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
    tens = ('zero ten twenty thirty forty fifty sixty seventy eighty ninety').split()
    if number < 20:
        return small[number]
    if number < 100:
        return tens[number // 10] + (' ' + integer_words(number % 10) if number % 10 else '')
    for scale, word in ((1000000, 'million'), (1000, 'thousand'), (100, 'hundred')):
        if number >= scale:
            return integer_words(number // scale) + ' ' + word + (' ' + integer_words(number % scale) if number % scale else '')
    raise AssertionError('Unreachable count')


def definition():
    return dict(type='function', name='answer_subset_count', strict=True,
                description='Terminal answer action: count subsets of distinct positive integers exactly and voice only the computed count. Use only when the user requests the count alone. Quote the complete number list and target exactly from the original question; do not invent, omit or change values. No extra explanation can accompany this terminal action.',
                parameters=dict(type='object', properties=dict(
                    numbers=dict(type='array', minItems=1, maxItems=24, items=dict(type='integer', minimum=1, maximum=100000)),
                    target=dict(type='integer', minimum=0, maximum=100000),
                    source_numbers=dict(type='string', minLength=1, maxLength=8192),
                    source_target=dict(type='string', minLength=1, maxLength=256)),
                    required=['numbers', 'target', 'source_numbers', 'source_target'], additionalProperties=False))


def execute(question, arguments):
    if not isinstance(question, str) or not question.strip() or len(question) > 16384:
        raise ValueError('Invalid bounded original question')
    if not isinstance(arguments, dict) or set(arguments) != {'numbers', 'target', 'source_numbers', 'source_target'}:
        raise ValueError('Exact count argument inventory differs')
    numbers, target = arguments['numbers'], arguments['target']
    if (not isinstance(numbers, list) or not 1 <= len(numbers) <= 24
            or any(type(n) is not int or not 1 <= n <= 100000 for n in numbers)
            or len(set(numbers)) != len(numbers) or type(target) is not int or not 0 <= target <= 100000):
        raise ValueError('Count requires bounded distinct positive integer values')
    for key, expected, bound in [('source_numbers', ' '.join(map(str, numbers)), 8192), ('source_target', str(target), 256)]:
        quote = arguments[key]
        if not isinstance(quote, str) or not 1 <= len(quote) <= bound or not bound_quote(question, quote) or quote_words(quote) != quote_words(expected):
            raise ValueError('Numeric arguments are not bound to an exact question quote')
    # Positive inputs allow pruning without losing any subset. Each item is
    # visited once; no permutation counting and no reuse of the same item.
    counts = {0: 1}
    for number in numbers:
        updated = counts.copy()
        for total, count in counts.items():
            if total + number <= target:
                updated[total + number] = updated.get(total + number, 0) + count
        counts = updated
    value = counts.get(target, 0)
    identity = hashlib.sha256(json.dumps(dict(question=question, arguments=arguments, value=value), sort_keys=True).encode()).hexdigest()
    return dict(tool='answer_subset_count', receipt_id='exact_' + identity[:24], executed=True, read_only=True,
                question_sha256=hashlib.sha256(question.encode()).hexdigest(), arguments=arguments, value=value,
                content=integer_words(value) + '.', algorithm='bounded positive-integer subset dynamic programming',
                state_bound=100001, selected_input_quotes_exact=True, semantic_input_completeness_proven=False)
