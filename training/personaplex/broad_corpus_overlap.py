"""Input-only lexical contamination screen; never a semantic clean-data claim."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(text, *, numeric_template=False):
    value = unicodedata.normalize('NFKC', text).casefold()
    result = re.findall(r'\d+(?:\.\d+)?|[^\W\d_]+(?:[\u2019\x27-][^\W\d_]+)*', value)
    return ['<number>' if re.fullmatch(r'\d+(?:\.\d+)?', word) else word for word in result] if numeric_template else result


def key(words):
    return hashlib.sha256(' '.join(words).encode()).hexdigest()


def shingles(words):
    return {key(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def patterns(text, source_id):
    result = []
    for mode in ('literal', 'numeric_template'):
        words = tokens(text, numeric_template=mode == 'numeric_template')
        # Generic short cues such as Continue are not distinctive content tests.
        if len(words) < 4:
            continue
        result.append(dict(source_id=source_id, mode=mode, token_count=len(words),
                           exact_sha256=key(words), shingles=sorted(shingles(words)) if len(words) >= 8 else []))
    return result


def user_texts(row):
    result = []
    for field in ('question', 'user', 'user_context'):
        if isinstance(row.get(field), str) and row[field].strip():
            result.append(row[field])
    streams = [row.get(field, []) for field in ('user_turns', 'turns', 'context')]
    inference = row.get('inference_inputs', {})
    if isinstance(inference, dict):
        streams.append(inference.get('authored_user_utterances', []))
    for turns in streams:
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if isinstance(turn, str) and turn.strip():
                result.append(turn)
            elif isinstance(turn, dict) and turn.get('role') == 'user' and isinstance(turn.get('text'), str) and turn['text'].strip():
                result.append(turn['text'])
    return result


def registry(source):
    source = Path(source)
    files = sorted((source / 'evals/fixtures').glob('*.json'))
    reviews = sorted((source / 'docs').glob('*SEMANTIC-REVIEW-*.json'))
    records, parents, counts, unparsed = [], [], Counter(), []
    for path in files + reviews:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            continue
        reviewed = path in reviews
        protected_file = reviewed or value.get('do_not_train') is True or path.name in (
            'venus-numeric16-text-20261108.json', 'venus-reasoning128-text-20261107.json')
        if not any(isinstance(value.get(field), list) and value[field] for field in ('scenes', 'cases')):
            unparsed.append(str(path.relative_to(source)))
        for field in ('scenes', 'cases'):
            rows = value.get(field, [])
            if not isinstance(rows, list):
                continue
            used = False
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                fold = str(row.get('fold', row.get('split', ''))).upper()
                if fold == 'TRAIN':
                    continue
                if not protected_file and fold not in ('DEV', 'DEVELOPMENT', 'VALIDATION', 'HOLDOUT', 'TEST', 'EVAL', 'EVALUATION-ONLY'):
                    if fold != 'TRAIN':
                        counts['fixture_rows_without_supported_fold'] += 1
                        relative = str(path.relative_to(source))
                        if relative not in unparsed:
                            unparsed.append(relative)
                    continue
                texts = user_texts(row)
                if not texts:
                    counts['unhandled_reviewed_or_evaluation_rows'] += 1
                    continue
                used = True
                counts['source_evaluation_rows'] += 1
                identity = str(path.relative_to(source)) + ':' + str(row.get('id', index))
                for text in texts + ([' '.join(texts)] if len(texts) > 1 else []):
                    records.extend(patterns(text, identity))
            if used and not any(p['path'] == str(path.relative_to(source)) for p in parents):
                parents.append(dict(path=str(path.relative_to(source)), sha256=sha(path)))
    unique = {(r['source_id'], r['mode'], r['exact_sha256']): r for r in records}
    if not unique:
        raise ValueError('No known evaluation input fingerprints collected')
    return dict(schema='dl-known-evaluation-lexical-registry-v1', patterns=list(unique.values()),
                source_files=parents, counts=dict(counts), plaintext_questions_exported=False,
                unparsed_schema_paths=unparsed,
                expected_answers_exported=False, all_historical_evaluations_claimed=False,
                semantic_decontamination_complete=False,
                limits=['Known labelled fixture inputs and inspected semantic-review question fields only.',
                        'Unknown schemas are reported; generic cues shorter than4tokens are not fingerprinted.',
                        'This is not access to evaluation answers or permission to train on protected inputs.'])


class Screen:
    def __init__(self, value):
        if value.get('schema') != 'dl-known-evaluation-lexical-registry-v1' or not value.get('patterns'):
            raise ValueError('Nonempty declared input-only fingerprint registry required')
        self.rows = value['patterns']
        self.exact, self.index = defaultdict(list), defaultdict(set)
        for i, row in enumerate(self.rows):
            self.exact[(row['mode'], row['exact_sha256'])].append(i)
            for item in row['shingles']:
                self.index[(row['mode'], item)].add(i)

    def matches(self, turns):
        texts = [turn['text'] for turn in turns if turn['role'] == 'user']
        results = {}
        for text in texts + ([' '.join(texts)] if len(texts) > 1 else []):
            for mode in ('literal', 'numeric_template'):
                words = tokens(text, numeric_template=mode == 'numeric_template')
                for i in self.exact.get((mode, key(words)), []):
                    results[i] = dict(source_id=self.rows[i]['source_id'], mode=mode, method='exact_input_fingerprint', containment=1.)
                observed = shingles(words)
                possible = Counter(i for item in observed for i in self.index.get((mode, item), ()))
                for i, shared in possible.items():
                    required = len(self.rows[i]['shingles'])
                    if i not in results and shared >= 3 and required and shared / required >= .8:
                        results[i] = dict(source_id=self.rows[i]['source_id'], mode=mode,
                                          method='known_input_fivegram_containment', containment=shared / required)
        return list(results.values())


def run(candidates, fingerprints, out):
    candidates, fingerprints, out = Path(candidates), Path(fingerprints), Path(out)
    screen = Screen(json.loads(fingerprints.read_text()))
    prior = json.loads((candidates / 'eligibility.json').read_text())
    out.mkdir(exist_ok=False)
    summaries = {}
    inputs = []
    for source in ('OpenAssistant--oasst1', 'databricks--databricks-dolly-15k'):
        path = candidates / source / 'candidates.jsonl'
        before = sha(path)
        folder = out / source
        folder.mkdir()
        counts = Counter()
        with path.open(encoding='utf-8') as incoming, (folder / 'candidates.jsonl').open('x', encoding='utf-8') as kept, (folder / 'overlap-matches.jsonl').open('x', encoding='utf-8') as rejected:
            for line in incoming:
                row = json.loads(line)
                if row.get('admitted_for_training') is not False or row.get('semantic_qualified') is not False:
                    raise ValueError('Only unadmitted source candidates may enter this screen')
                counts['input_candidates'] += 1
                matches = screen.matches(row['turns'])
                if matches:
                    counts['withheld_known_input_overlap'] += 1
                    rejected.write(json.dumps(dict(id=row['id'], source_group_id=row['source_group_id'],
                                                   content_sha256=row['content_sha256'], matches=matches)) + '\n')
                else:
                    counts['retained_lexical_candidates'] += 1
                    kept.write(line)
        if sha(path) != before:
            raise ValueError('Input candidate writer changed during screening')
        inputs.append(dict(path=str(path), sha256=before))
        summaries[source] = dict(counts)
    expected = sum(v['eligible_text_candidates'] for v in prior['sources'].values())
    if sum(v['input_candidates'] for v in summaries.values()) != expected:
        raise ValueError('Whole original candidate cohort must be screened, never a subset')
    report = dict(status='KNOWN_INPUT_LEXICAL_SCREEN_COMPLETE_NOT_SEMANTIC_ADMISSION', sources=summaries,
                  input_files=inputs, fingerprint_registry_sha256=sha(fingerprints),
                  prior_eligibility_sha256=sha(candidates / 'eligibility.json'),
                  semantic_near_duplicate_checks_pending=True, family_split_assignment_pending=True,
                  factual_qualification_complete=False, native_audio_created=False,
                  optimizer_updates=0, model_API_calls=0, admitted_for_training=False)
    (out / 'screening.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='stage')
    build = commands.add_parser('registry')
    build.add_argument('--source', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True)
    check = commands.add_parser('screen')
    check.add_argument('--candidates', type=Path, required=True)
    check.add_argument('--registry', type=Path, required=True)
    check.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.stage is None:
        parser.error('An explicit registry or screen stage is required')
    if args.stage == 'registry':
        with args.out.open('x') as stream:
            json.dump(registry(args.source), stream, indent=2)
            stream.write('\n')
    else:
        run(args.candidates, args.registry, args.out)
