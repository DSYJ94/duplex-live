"""Local necessary-fact diagnostic only; never a semantic or training approval."""
from pathlib import Path
import argparse,datetime,hashlib,itertools,json,re,shutil
BASE=Path('runs/local-migration-20261023')
PREP=BASE/'native-reasoning32-derived-v10-prepared-semantic-v1.json'
small=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),range(20)))
tens={'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90}
def number(text):
 parts=text.lower().replace('-',' ').split()
 if len(parts)==1 and parts[0].isdigit():return int(parts[0])
 if len(parts)==1 and parts[0]in small:return small[parts[0]]
 if len(parts)==1 and parts[0]in tens:return tens[parts[0]]
 if len(parts)==2 and parts[0]in tens and parts[1]in small and 0<small[parts[1]]<10:return tens[parts[0]]+small[parts[1]]
 raise ValueError('Unsupported number '+text)
pattern=re.compile(r'\b(?:\d+|(?:'+ '|'.join(tens)+r')(?:[ -]+(?:one|two|three|four|five|six|seven|eight|nine))?|'+ '|'.join(small)+r')\b',re.I)
def numbers(text):return [number(m.group())for m in pattern.finditer(text)]
NAMES={'badge','candle','marker','poster','pouch','ribbon'}
def names(text):return sorted(set(re.findall(r'\b(?:'+ '|'.join(sorted(NAMES))+r')\b',text.lower())))
def problem(text):
 assert text.startswith('Choose exactly three different items.')
 facts=re.findall(r'The (\w+) costs ([a-z ]+) tokens and earns ([a-z ]+) points\.',text)
 assert len(facts)==6 and {x[0]for x in facts}==NAMES
 costs={n:number(c)for n,c,p in facts};points={n:number(p)for n,c,p in facts}
 budget=number(re.search(r'Spend at most ([a-z ]+) tokens\.',text)[1])
 required=re.search(r'Include the (\w+)\.',text)[1]
 forbidden=set(re.search(r'Never choose both the (\w+) and the (\w+)\.',text).groups())
 assert required in NAMES and len(forbidden)==2 and forbidden<=NAMES
 assert 'Maximize total points. Break ties by lower cost, then alphabetical order of the three names.'in text
 return dict(costs=costs,points=points,budget=budget,required=required,forbidden=sorted(forbidden))
def solve(p,budget=None):
 budget=p['budget']if budget is None else budget;feasible=[]
 for triple in itertools.combinations(sorted(NAMES),3):
  cost=sum(p['costs'][n]for n in triple);points=sum(p['points'][n]for n in triple)
  if p['required']in triple and not set(p['forbidden'])<=set(triple)and cost<=budget:feasible.append((triple,cost,points))
 assert feasible
 triple,cost,points=min(feasible,key=lambda x:(-x[2],x[1],x[0]))
 return dict(items=list(triple),cost=cost,points=points,feasible_three_item_combinations=len(feasible),enumerated_combinations=20,budget=budget)
def necessary(text,answer):
 nums=numbers(text)
 return dict(item_names_match=names(text)==answer['items'],cost_number_present=answer['cost']in nums,points_number_present=answer['points']in nums,observed_numbers=nums)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=BASE/'native-reasoning32-derived-v10-prepared-semantic-v1.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();PREP=args.prepared;OUT=args.out
    raw=PREP.read_bytes();assert hashlib.sha256(raw).hexdigest()=='63de073379fc9871f062f7c1616ab54f27bdc43037fb5deee739339dbb2fb6ee'
    v=json.loads(raw);ids=['native-reasoning4-train-%03d'%i for i in range(32)];assert v['selected_ids']==ids and list(v['messages'])==ids
    rows=[];echoes=[];all_receipts=[]
    for identity in ids:
     m=json.loads(v['messages'][identity][1]['content']);u=m['authored_user_utterances'];asr=m['independent_asr_text'];hard=u[0].startswith('Choose exactly three');p=problem(u[0])if hard else None
     family='direct-basic'if not hard else('replace-reasoning'if u[-1].startswith('Change the budget')else'cancel-reasoning'if u[-1].startswith('Cancel')else'delegate-reasoning')
     row=dict(id=identity,family=family,independent_asr_text=asr,derived_from_authored_premises_only=True)
     if family!='replace-reasoning':
      basic=re.search(r'one (\w+) costs ([a-z ]+) tokens\. How many tokens do two (\w+) cost\?',u[-1],re.I);assert basic
      unit=number(basic[2]);expected=unit*2
      row['basic']=dict(item=basic[1],unit_cost=unit,expected_two_cost=expected,correct_number_present_in_full_ASR=expected in numbers(asr))
      if hard:
       tail=m['actual_arrived_user_transcript'][-1]['received_at_s'];mentions=[]
       for w in m['independently_measured_assistant_words']:
        found=numbers(w['text'])
        if unit in found and w['end']<tail:mentions.append(dict(text=w['text'],start=w['start'],end=w['end'],unit_cost=unit,last_arrived_user_word_s=tail))
       if mentions:echoes.append(dict(id=identity,mentions=mentions,interpretation='Numeric one-item echo before the complete two-item question; requires semantic/listening review, not an automatic factual failure.'))
     if hard:
      original=solve(p);row['original_bundle_solution']=original
      if family=='replace-reasoning':
       changed=number(re.fullmatch(r'Change the budget to ([a-z ]+) tokens\. Keep all the other rules the same\.',u[-1])[1]);row['revised_bundle_solution']=solve(p,changed)
      if family in('delegate-reasoning','replace-reasoning'):
       expected=row.get('revised_bundle_solution',original);row['final_spoken_necessary_facts']=necessary(asr,expected)
      tasks=m['actual_tasks_and_receipts'];row['receipt_checks']=[]
      for task_id,t in tasks.items():
       r=t['receipt'];op=r['operation'];check=dict(task_id=task_id,operation=op,revision=r['revision'],cancelled=t['cancelled'],content=r['content'],actual_transport=r['actual_transport'])
       if op in('respond_to_user','replace_reasoning'):
        expected=row['revised_bundle_solution']if op=='replace_reasoning'else original
        check.update(expected_solution=expected,necessary_facts=necessary(r['content'],expected));all_receipts.append(check)
       elif op=='cancel_reasoning':
        retired=r['retired_requests'];original_ids=[n for n,z in tasks.items()if z['revision']==1 and z['cancelled']]
        check['all_actual_original_tasks_retired']=bool(original_ids)and sorted(z['task_id']for z in retired)==sorted(original_ids)
       else:raise ValueError('Unknown operation')
       row['receipt_checks'].append(check)
     rows.append(row)
    fields=('item_names_match','cost_number_present','points_number_present')
    math_ok=sum(all(r['necessary_facts'][k]for k in fields)for r in all_receipts)
    spoken=[r['final_spoken_necessary_facts']for r in rows if'final_spoken_necessary_facts'in r]
    basic=[r['basic']for r in rows if'basic'in r]
    report=dict(schema='dl-native-reasoning32-local-necessary-fact-diagnostic-v1',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),diagnostic_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prepared_source_sha256=hashlib.sha256(raw).hexdigest(),all32_examined=True,case_count=len(rows),independent_exhaustive_solver_uses_source_oracles=False,backend_bundle_receipts_examined=len(all_receipts),backend_bundle_receipts_matching_necessary_facts=math_ok,final_bundle_utterances_examined=len(spoken),final_bundle_utterances_matching_necessary_facts=sum(all(x[k]for k in fields)for x in spoken),basic_utterances_examined=len(basic),basic_utterances_with_expected_number_present=sum(x['correct_number_present_in_full_ASR']for x in basic),pre_question_numeric_echoes=echoes,limitations=['Necessary facts only: number presence does not prove unambiguous wording, timing, good interaction, or correct use of delivered context.','The entire source-bound independent semantic review is still required; this local diagnostic does not replace it.','No audio was played or transcribed by this script; existing complete independent ASR was inspected.','All32 retained; no teacher rewritten, rejected source erased, gate changed or training run started.'],semantic_approved=False,training_approved=False,provider_requests=0,rows=rows)
    assert not OUT.exists();OUT.mkdir();(OUT/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,OUT/'reproduce.py')
    print(json.dumps({k:x for k,x in report.items()if k not in('rows','limitations')}))

if __name__=='__main__':main()
