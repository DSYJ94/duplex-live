"""Verified text-only curriculum, separate from the unchanged native audio corpus."""
import argparse,hashlib,json,random,re
from pathlib import Path
from .reasoning_pool import arithmetic

SYSTEM='You are a helpful voice assistant. Answer straightforward questions accurately and briefly. Do not invent missing information.'


def normalized(text):return ' '.join(re.sub(r'[^a-z0-9 ]',' ',text.lower()).split())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().split('\n') if line]


def synthetic(kind,n):
    a,b,c=23+n,3+n%7,2+n%5
    rows=[
      (f'A workshop has {a} washers. It uses {b} and then receives {c}. How many washers does it have now?',f'{a}-{b}+{c}','washers'),
      (f'A kit has {80+7*n} screws. Four assemblies each use {b} screws. How many screws are left?',f'{80+7*n}-4*{b}','screws'),
      (f'A cable measures {3+n} metres. After cutting off {15+n%80} centimetres, how many centimetres are left?',f'{3+n}*100-{15+n%80}','centimetres'),
      (f'A process starts at elapsed minute {8+n} and lasts {18+n%17} minutes. At what elapsed minute does it finish?',f'{8+n}+{18+n%17}','minutes'),
      (f'A rectangle has length {7+n} metres and width {3+n%12} metres. What is its perimeter?',f'2*({7+n}+{3+n%12})','metres'),
      (f'One quarter of a collection of {4*(10+n)} counters is set aside. How many counters are set aside?',f'{4*(10+n)}/4','counters'),
      (f'A pump moves {3+n%7} litres each minute for {4+n} minutes. How many litres does it move?',f'{3+n%7}*{4+n}','litres'),
      (f'{2+n} tickets cost {3+n%13} pounds each. One booking fee adds {2+n%5} pounds to the whole purchase. What is the total cost?',f'{2+n}*{3+n%13}+{2+n%5}','pounds'),
      (f'A register contains {a}. Add {b}, double the result, and subtract {c}. What value remains?',f'({a}+{b})*2-{c}',''),
      (f'Tank A holds {30+n} litres and tank B holds {12+n%8}. Move {3+n%7} litres from A to B without losing any. What is the combined volume?',f'{30+n}+{12+n%8}','litres'),
      (f'Jo has {a} tokens. Kai has {b} fewer than Jo. Lee has three times as many as Kai. How many tokens does Lee have?',f'({a}-{b})*3','tokens'),
      (f'A delivery was scheduled for project day {7+n}. A delay adds {3+n%7} days and a later revision brings it forward {1+n%3} days. What is its new project day?',f'{7+n}+{3+n%7}-{1+n%3}',''),
      (f'A resistor carries {2+n%5} amperes under {((2+n%5)*(4+n))} volts. Using voltage equals current times resistance, what is its resistance?',f'{(2+n%5)*(4+n)}/{2+n%5}','ohms'),
      (f'A vehicle covers {(3+n%7)*(8+n)} metres in {3+n%7} seconds. What is its average speed?',f'{(3+n%7)*(8+n)}/{3+n%7}','metres per second'),
      (f'What is the mean of the three measurements {a}, {a+3} and {a+6}?',f'({a}+{a+3}+{a+6})/3',''),
      (f'A shop sells {2+n%11} items at {9+n} pounds each. Each item cost the shop {3+n%4} pounds. What is the total profit?',f'({9+n}-{3+n%4})*{2+n%11}','pounds'),
    ]
    question,expression,unit=rows[kind];value=arithmetic(expression)
    if value.denominator!=1:raise ValueError('This generated tranche uses exact integer targets')
    target=str(value.numerator)
    return dict(question=question,response='The answer is '+target+(' '+unit if unit else '')+'.',
        numeric_target=target,verification=dict(method='exact-rational-program',expression=expression),
        source_kind='authored_verified_arithmetic',template=kind)


def build(pool,excluded):
    excluded={normalized(q) for q in excluded};seen=set(excluded);rows=[]
    def add(row,fold):
        identity=hashlib.sha256(normalized(row['question']).encode()).hexdigest()
        if normalized(row['question']) in seen:raise ValueError('Question overlaps evaluation or another row')
        seen.add(normalized(row['question']));rows.append(dict(id='reasoning-text-'+identity[:20],fold=fold,**row))
    for kind in range(16):
        for n in range(32):add(synthetic(kind,n),'TRAIN')
        for n in (100,101):add(synthetic(kind,n),'DEV')
    eligible=[r for r in pool if r['source_split']=='train' and r['all_present_annotations_exact']
              and normalized(r['question']) not in seen and len(r['question'])+len(r['original_solution'])<=1800]
    random.Random(20261108).shuffle(eligible)
    if len(eligible)<528:raise ValueError('All496 public TRAIN plus32 derived DEV examples required')
    for i,r in enumerate(eligible[:528]):
        body,answer=r['original_solution'].rsplit('####',1)
        response=re.sub(r'<<[^<>]+>>','',body).strip()+'\nThe answer is '+answer.strip()+'.'
        add(dict(question=r['question'],response=response,numeric_target=r['numeric_target'],source_kind='gsm8k_official_train',source_id=r['id'],source_row=r['source_row'],original_solution=r['original_solution'],verification=dict(method='original-public-gold-plus-exact-present-arithmetic-annotations',all_present_annotations_exact=True,full_semantic_verification=False)),'TRAIN' if i<496 else 'DEV')
    assert sum(r['fold']=='TRAIN' for r in rows)==1008 and sum(r['fold']=='DEV' for r in rows)==64
    return dict(schema='dl-reasoning-text-curriculum-v1',system=SYSTEM,rows=rows,
        native_audio_created=False,official_test_used=False,source_pool_rows_preserved=True)


def messages(corpus,row,*,answer=False):
    out=[dict(role='system',content=corpus['system']),dict(role='user',content=row['question'])]
    if answer:out.append(dict(role='assistant',content=row['response']))
    return out


def main(a):
    reference=json.loads(Path('docs/NATIVE-GSM8K-SOURCE-POOL-2026-11-07.json').read_text())
    if sha(a.pool)!=reference['pool_sha256'] or sha(a.license)!=reference['license_file_sha256']:
        raise ValueError('Exact preserved public TRAIN pool and original MIT license required')
    excluded=[]
    for p in ('evals/fixtures/venus-reasoning128-text-20261107.json','evals/fixtures/venus-numeric16-text-20261108.json'):
        excluded.extend(r['question'] for r in json.loads(Path(p).read_text())['cases'])
    for p in ('evals/fixtures/venus-control-corpus-v4-20261026.json','evals/fixtures/venus-control-repair-source-v2-20261027.json','evals/fixtures/venus-retention-request-source-proposal-v2-20261107.json'):
        excluded.extend(q for r in json.loads(Path(p).read_text())['scenes'] for q in r['user_turns'])
    # JSONL is delimited by LF, not Unicode line separators inside valid strings.
    corpus=build(read_jsonl(a.pool),excluded)
    a.output.mkdir(parents=True,exist_ok=False)
    path=a.output/'curriculum.json';path.write_text(json.dumps(corpus,indent=2)+'\n')
    (a.output/'LICENSE-GSM8K').write_bytes(a.license.read_bytes())
    manifest=dict(schema=corpus['schema'],curriculum_sha256=sha(path),TRAIN=1008,DEV=64,
        TRAIN_sources=dict(authored_verified_arithmetic=512,gsm8k_official_train=496),
        DEV_sources=dict(authored_verified_arithmetic=32,gsm8k_official_train=32),
        official_GSM8K_split_used='train',original_pool=reference,license_sha256=sha(a.license),
        excluded_question_hashes=sorted({hashlib.sha256(normalized(q).encode()).hexdigest() for q in excluded}),
        annotation_check_is_not_full_semantic_verification=True,native_audio_created=False,
        new_external_API_calls=0,maximum_token_length_pending_actual_tokenizer=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('schema','curriculum_sha256','TRAIN','DEV','TRAIN_sources','DEV_sources','new_external_API_calls')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('pool','license','output'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
