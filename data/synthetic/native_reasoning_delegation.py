"""Author matched basic/hard reasoning and asynchronous lifecycle TRAIN proposals.

No audio/API calls, teacher approvals, held-out reads or optimizer work. Oracle
solutions belong only to annotation; a native worker result must actually arrive.
This is a new schema, deliberately incompatible with the old private-record data.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import random

NAMES=(('badge','badges'),('candle','candles'),('marker','markers'),
       ('poster','posters'),('pouch','pouches'),('ribbon','ribbons'))
FAMILIES=('direct-basic','delegate-reasoning','cancel-reasoning','replace-reasoning')


def words(number):
    small=('zero','one','two','three','four','five','six','seven','eight','nine','ten',
           'eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen')
    if type(number) is not int or not 0<=number<100:raise ValueError('Bounded exact integer source words required')
    if number<20:return small[number]
    tens=('', '', 'twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety')
    return tens[number//10]+(' '+small[number%10] if number%10 else '')


def solve(problem):
    """Exact finite optimization; no model answers or tool receipts are assumed."""
    options=[];items=problem['items']
    names={n for n,_ in NAMES}
    if (len(items)!=6 or {r['name'] for r in items}!=names or problem['choose_exactly']!=3
            or type(problem['budget']) is not int or not 8<=problem['budget']<=30
            or problem['required'] not in names or len(problem['incompatible'])!=2
            or len(set(problem['incompatible']))!=2 or not set(problem['incompatible']).issubset(names)
            or problem['required'] in problem['incompatible']
            or any(type(r[k]) is not int or not 1<=r[k]<=20 for r in items for k in ('cost','points'))):
        raise ValueError('Exact bounded six-item public reasoning problem required')
    for chosen in itertools.combinations(range(len(items)),3):
        names=sorted(items[i]['name'] for i in chosen)
        cost=sum(items[i]['cost'] for i in chosen);points=sum(items[i]['points'] for i in chosen)
        if (cost>problem['budget'] or problem['required'] not in names
                or set(problem['incompatible']).issubset(names)):continue
        options.append(dict(names=names,cost=cost,points=points))
    if not options:raise ValueError('No valid exact three-item bundle')
    # Stated tie rules: higher points, then lower cost, then alphabetical names.
    return min(options,key=lambda r:(-r['points'],r['cost'],r['names']))


def instance(index):
    rng=random.Random(23102026+index)
    for _ in range(1000):
        items=[dict(name=n,cost=rng.randint(2,11),points=rng.randint(3,19)) for n,_ in NAMES]
        required=NAMES[index%len(NAMES)][0]
        others=[n for n,_ in NAMES if n!=required]
        problem=dict(items=items,budget=rng.randint(15,23),required=required,
                     incompatible=rng.sample(others,2),choose_exactly=3)
        try:original=solve(problem)
        except ValueError:continue
        for change in (-5,-3,3,5):
            corrected={**problem,'budget':problem['budget']+change}
            if corrected['budget']<8:continue
            try:replacement=solve(corrected)
            except ValueError:continue
            if original['names']!=replacement['names']:return problem,corrected
    raise ValueError('Could not author a genuinely changed optimal bundle')


def question(problem):
    clauses=[f"The {r['name']} costs {words(r['cost'])} tokens and earns {words(r['points'])} points."
             for r in problem['items']]
    a,b=problem['incompatible']
    return ('Choose exactly three different items. '+ ' '.join(clauses)+
            f" Spend at most {words(problem['budget'])} tokens. Include the {problem['required']}."
            f' Never choose both the {a} and the {b}. Maximize total points.'
            ' Break ties by lower cost, then alphabetical order of the three names.'
            ' Tell me the items, total cost and total points.')


def answer(solution):
    a,b,c=solution['names']
    return (f'Choose the {a}, {b} and {c}. That costs {words(solution["cost"])} tokens'
            f' and earns {words(solution["points"])} points.')


def build():
    rows=[]
    for index in range(8):
        problem,corrected=instance(index);original=solve(problem);replacement=solve(corrected)
        item=problem['items'][index%len(NAMES)];plural=dict(NAMES)[item['name']]
        basic=f"One {item['name']} costs {words(item['cost'])} tokens. How many tokens do two {plural} cost?"
        direct=f"Two {plural} cost {words(2*item['cost'])} tokens."
        for family in FAMILIES:
            user=[basic] if family=='direct-basic' else [question(problem)]
            expected=direct if family in ('direct-basic','cancel-reasoning') else answer(original)
            predicates=[];returns=[]
            if family!='direct-basic':
                predicates=['native request selected without a forced RET marker',
                            'request context contains only actual arrived user words',
                            'real reasoning API response ID, input hash, completion and arrival times retained']
                returns=[dict(revision=1,problem=problem,oracle=original,
                              deliverable=family=='delegate-reasoning')]
                if family=='delegate-reasoning':
                    user.append('While that is pending, '+basic[0].lower()+basic[1:])
                    predicates+=['basic question answered directly while reasoning is pending',
                                 'final bundle spoken only after matching actual reasoning result arrival']
                elif family=='cancel-reasoning':
                    user.append('Cancel the bundle search. Instead, '+basic[0].lower()+basic[1:])
                    predicates+=['actual cancellation receipt precedes any cancellation confirmation',
                                 'late original result rejected; latest basic question answered directly']
                else:
                    user.append(f"Change the budget to {words(corrected['budget'])} tokens. Keep all the other rules the same.")
                    returns.append(dict(revision=2,problem=corrected,oracle=replacement,deliverable=True))
                    expected=answer(replacement)
                    predicates+=['original revision retired; one new reasoning request contains the correction',
                                 'late original result rejected; only current revision result spoken']
            annotations=dict(expected_foreground_action='answer-directly' if family=='direct-basic' else 'request-reasoning',
                worker_route='none' if family=='direct-basic' else 'answer',
                explanation='A native external request and the reasoning backend answer route are different decisions',
                expected_final_spoken_words=expected,expected_pending_basic_words=direct if family=='delegate-reasoning' else None,
                actual_worker_returns_required=returns,required_predicates=predicates,
                native_request_forcing_allowed=False,source_oracle_is_not_a_worker_receipt=True,
                minimum_result_delivery_delay_s=(16. if family in ('cancel-reasoning','replace-reasoning')
                                                 else (3.,8.,12.,16.)[index%4]),
                lifecycle_event_after_actual_dispatch=family in ('cancel-reasoning','replace-reasoning'))
            rows.append(dict(id=f'native-reasoning4-train-{len(rows):03d}',split='train',family=family,
                matched_instance=index,user_voice=('alloy','echo')[index%2],assistant_voice='marin',
                inference_inputs=dict(authored_user_utterances=user,scoring_answers_or_oracles_supplied=False),
                annotations_not_model_inputs=annotations,training_approved=False))
    plan=dict(schema='dl-native-reasoning-delegation-source-proposal-v1',cases=rows,
              status='authored source proposal only; no recordings, native teachers or new training',
              owner_conversations_used=False,existing_folds_or_heldouts_read_or_changed=False,
              api_calls=0,optimizer_updates=0,training_approved=False,
              existing_control40_or_retention_gates_changed=False,
              required_retention=dict(actual_histories=28,old_native_teachers=25,new_native_teachers=24,
                                      original_objectives=154,original_native_call_teachers=49,kl=1.,hidden=10.,word=64.,clock=64.),
              required_before_any_teacher_collection=[
                  'Explicit new source audio and real asynchronous reasoning/cancel/replacement teacher protocol',
                  'Measured lifecycle event must follow actual dispatch; do not invent a dispatch time',
                  'Cancellation/replacement must arrive while the original task is actually pending; retain real execution and transport-delay clocks separately',
                  'Each complete measured episode, native prefix and actual context spans must fit the existing1100-frame bound; never crop to fit',
                  'Same real-backend interface for both matched basic and hard questions; no hand-coded difficulty router',
                  'Separate operator/scoring annotations from every model and backend input'],
              required_before_training=['Whole32 source audio and actual native teacher receipts',
                  'Independent whole-corpus words, waveform quality, semantic and conversion approval',
                  'Native numerical and gradient approval plus combined retained-objective readiness',
                  'Distinct bounded pilot, full independent training-state restore and complete native behavior review'])
    validate(plan);return plan


def validate(plan):
    rows=plan['cases']
    if (len(rows)!=32 or Counter(r['family'] for r in rows)!=Counter({f:8 for f in FAMILIES})
            or len({r['id'] for r in rows})!=32 or any(r['split']!='train' or r['training_approved'] for r in rows)
            or plan['api_calls'] or plan['optimizer_updates'] or plan['training_approved']):
        raise ValueError('Complete source-only32 required; no approval or passing subset')
    for row in rows:
        inputs=row['inference_inputs'];labels=row['annotations_not_model_inputs']
        if set(inputs)!={'authored_user_utterances','scoring_answers_or_oracles_supplied'} or inputs['scoring_answers_or_oracles_supplied']:
            raise ValueError('Only authored user words belong in source inference inputs')
        if labels['native_request_forcing_allowed'] or not labels['source_oracle_is_not_a_worker_receipt']:
            raise ValueError('No forced request or oracle substituted for an actual worker')
        if (labels['expected_foreground_action']!=('answer-directly' if row['family']=='direct-basic' else 'request-reasoning')
                or labels['worker_route']!=('none' if row['family']=='direct-basic' else 'answer')):
            raise ValueError('Separate native delegation choice from backend answer production')
        for result in labels['actual_worker_returns_required']:
            if solve(result['problem'])!=result['oracle']:raise ValueError('Changed independently computable source target')
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    with args.out.open('x') as file:json.dump(build(),file,indent=2);file.write('\n')
