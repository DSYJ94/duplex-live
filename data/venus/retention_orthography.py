"""Versioned written-form review; originals and original failed audit stay intact.

Only four explicit same-word conventions change. No cue, key, phonetic or
numeric equivalence is introduced; the original WER/peak limits stay unchanged.
This is not forced alignment or independent semantic approval.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from .context_source_quality import words as original_words
from .retention_quality import inventory, critical_keys_match

RULES = {'programme': 'program', 'optimiser': 'optimizer', 'paperclip': 'paper clip',
         'non-deterministic / non deterministic': 'nondeterministic'}


def words(text):
    text = text.lower()
    for old, new in list(RULES.items())[:3]:
        text = re.sub(r'\b'+old+r'\b', new, text)
    text = re.sub(r'\bnon[ -]+deterministic\b', 'nondeterministic', text)
    return original_words(text)


def wer(reference, hypothesis):
    a, b = words(reference), words(hypothesis); row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(row[j]+1, nxt[j-1]+1, row[j-1]+(x != y)))
        row = nxt
    return row[-1]/max(1, len(a))


def review(parent, original, resumed, payload):
    raw = Path(parent).read_bytes(); old = json.loads(raw)
    sources = inventory(original, resumed, payload)
    if not old['complete'] or old['required'] != 182 or len(old['rows']) != 182:
        raise ValueError('Complete original182 independent recognition required')
    observed = {r['id']: r for r in old['rows']}
    if set(observed) != {r['id'] for r in sources}:
        raise ValueError('No subset or alternate recordings')
    rows = []
    for source in sources:
        r = observed[source['id']]
        if (r['source_sha256'] != source['source_sha256'] or r['text'] != source['text']
                or r['original_peak'] != source['original_peak']):
            raise ValueError('Exact original source/transcript required')
        error = wer(r['text'], r['recognized'])
        keys = critical_keys_match(r['text'], r['recognized'], source['expected_lookup_keys'])
        passed = error <= .05 and r['original_peak'] <= .995 and keys and r['word_timestamps_complete']
        rows.append(dict(r, original_word_error_rate=r['word_error_rate'], original_passed=r['passed'],
            word_error_rate=error, critical_lookup_keys_pass=keys, passed=passed,
            assistant_authored_word_alignment_pending=source['profile']=='assistant-british'))
    return dict(schema='dl-retention96-orthographic-derived182-v1', complete=True, required=182,
        passed=sum(r['passed'] for r in rows), all_sources_pass=all(r['passed'] for r in rows),
        parent_report_sha256=hashlib.sha256(raw).hexdigest(), original_failed_report_preserved=True,
        original_audio_preserved=True, recognition_not_rerun=True, rules=RULES, rows=rows,
        wer_threshold=.05, raw_peak_threshold=.995, no_homophone_equivalence=True,
        all_lookup_keys_exact=all(r['critical_lookup_keys_pass'] for r in rows),
        no_timestamp_interpolation=True, complete_assistant_alignment_pending=True,
        independent_semantic_review_pending=True, training_approved=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','original','resumed','payload','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier derived reports')
    r=review(a.parent,a.original,a.resumed,a.payload)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'}))
    if not r['all_sources_pass']:raise RuntimeError('Whole182 still fails; no subset approval')
