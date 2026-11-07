"""Freeze an authored text reasoning/routing probe; answers never enter prompts.

This isolates language reasoning from audio perception and speech rendering.
"""
import hashlib,json,random
from pathlib import Path

SYSTEM='Choose the best answer or action. Return exactly one capital letter A, B, C, or D. Use the supplied facts; do not invent missing information. A live voice assistant should handle basic reasoning directly, delegate live external lookups or substantial research, ask for essential missing information, and ignore cancelled or superseded results.'


def build():
    rows=[];rng=random.Random(20261107)
    def add(category,question,correct,wrong):
        options=[correct]+list(wrong)
        if len(options)!=4 or len(set(options))!=4:raise ValueError('Four distinct options required')
        rng.shuffle(options);rows.append(dict(id=f'reasoning128-{len(rows):03d}',category=category,
            question=question,options=options,expected='ABCD'[options.index(correct)]))
    def number(category,q,value):
        add(category,q,str(value),[str(value+1),str(value+4),str(value-2)])
    for n in range(4):
        number('everyday',f'A box has {19+n} screws. We use 7, then add 5. How many screws remain?',17+n)
        number('everyday',f'A recipe uses 3 grams per portion. We make {5+n} portions and spill 2 grams. How many grams must we start with?',3*(5+n)+2)
        number('everyday',f'A timer starts at minute {12+n} on a clock and runs 18 minutes. At which minute on the same hour does it stop?',30+n)
        number('everyday',f'A rectangle is {6+n} metres long and 4 metres wide. What is its perimeter in metres?',2*(10+n))
        number('everyday',f'We save one quarter of {48+4*n} pounds, then spend 3 pounds from those savings. How many saved pounds remain?',9+n)
        number('everyday',f'A ribbon is {3+n} metres long. We cut off 40 centimetres. How many centimetres remain?',260+100*n)
        number('everyday',f'A bag contains {6+n} red counters and 4 blue counters. After removing 2 red counters, how many counters are left in total?',8+n)
        number('everyday',f'We pack {17+3*n} items into boxes holding at most 5 items each. What is the minimum number of boxes?',(21+3*n)//5)
    facts=[
      ('A queue receives Cedar, then Elm, then Pine. It removes two items in FIFO order. Which item remains?','Pine',['Cedar','Elm','No item']),
      ('A stack receives Cedar, Elm, then Pine. It pops twice. Which item remains?','Cedar',['Elm','Pine','No item']),
      ('An algorithm repeatedly halves a sorted search interval. Which growth rate describes its search comparisons?','Logarithmic',['Linear','Quadratic','Factorial']),
      ('A hash table reports that two different keys share a bucket. What happened?','A collision',['The keys became equal','The table was sorted','The values were encrypted']),
      ('A function changes no external state and always returns the same value for the same input. What property is described?','Purity',['Randomness','Concurrency','Mutability']),
      ('A program must allow only one thread at a time to modify a shared counter. Which mechanism directly provides that exclusion?','A mutex',['A checksum','A filename extension','A floating point cast']),
      ('A procedure calls itself on a smaller instance until it reaches a base case. What technique is used?','Recursion',['Encryption','Hashing','Caching']),
      ('A cache stores a previous expensive computation. Why can a later request finish faster?','It can reuse the stored result',['It increases the mathematical answer','It removes every possible error','It changes the input into a password']),
      ('A transaction updates two related balances. If the second update fails, what does atomicity require?','Both updates are rolled back',['Only the first update remains','Both balances become zero','The failure is ignored']),
      ('A test set has influenced repeated parameter choices. Which new evidence best measures generalisation?','An untouched evaluation set',['The same test set again','Training accuracy alone','The filename of the checkpoint']),
      ('A model fits training examples well but performs poorly on new examples. What is the central concern?','Overfitting',['Perfect generalisation','Insufficient disk compression','A larger sample rate']),
      ('A gradient points in the direction of steepest local increase of a scalar loss. Which direction does basic gradient descent take?','The negative gradient',['The positive gradient','An unrelated filename','Only the largest weight']),
      ('Several small batches contribute gradients before one optimiser step. What does this directly emulate?','A larger effective batch',['A new vocabulary','An extra audio channel','A held-out test set']),
      ('A training loss is NaN. Which is the appropriate immediate interpretation?','The numerical result is invalid',['The model is perfectly trained','The loss must equal zero','The data has been encrypted']),
      ('A probability distribution assigns probabilities to three mutually exclusive exhaustive events. What must their probabilities sum to?','One',['Zero','Three','Any negative number']),
      ('Independent fair coin tosses produced five heads. What is the probability of heads on the next toss?','One half',['One sixth','One','Zero']),
      ('A vector has entries 3 and 4. What is its Euclidean length?','5',['7','12','25']),
      ('Two vectors have dot product zero and both have positive length. What relation does this establish?','They are orthogonal',['They are identical','They are parallel','They have equal lengths']),
      ('A positive number is multiplied by 0.5 twice. What fraction of the original remains?','One quarter',['One half','Two','Four']),
      ('A derivative at a point describes which local quantity?','Rate of change',['The total file size','A random sample','The number of rows in every matrix']),
      ('What does a 95 percent confidence interval procedure guarantee over repeated samples under its assumptions?','About 95 percent of its intervals cover the fixed parameter',['The fixed parameter changes with each sample','Every reported interval is certainly correct','95 percent of observations equal the mean']),
      ('Why should preprocessing learned from data be fitted on the training split before evaluating on a test split?','To avoid information leakage',['To guarantee perfect test accuracy','To increase test labels','To make every variable categorical']),
      ('A private result arrives for a request that was already cancelled. What should an asynchronous assistant do?','Ignore that cancelled result',['Read it aloud immediately','Pretend the cancellation never happened','Attach it to an unrelated new task']),
      ('An answer sounds fluent but contradicts the supplied evidence. How should its correctness be scored?','Incorrect',['Correct because fluent','Correct because long','Impossible to evaluate any evidence'])]
    for q,c,w in facts:add('technical',q,c,w)
    for n in range(6):
        number('context',f'The register starts at {8+n}. Instruction one adds 4. Instruction two doubles the new value. Instruction three subtracts 3. What is the final value?',2*(12+n)-3)
        number('context',f'Mina has {14+n} tokens. Sol has 3 fewer than Mina. Ren has twice as many as Sol. How many tokens does Ren have?',2*(11+n))
        number('context',f'A delivery was planned for day {9+n}. It is delayed by 3 days and then brought forward by 1 day. Which day is the revised delivery?',11+n)
        number('context',f'Tank A holds {22+n} litres and tank B holds 8 litres. We move 5 litres from A to B without losing any. What is their combined volume?',30+n)
    ambiguous=[
      'Convert the total into another currency, but no target currency is specified.',
      'Compare two measurements, but their units are omitted.',
      'Book a meeting tomorrow, but the timezone is unspecified and the participants are in different zones.',
      'Calculate a rectangle area, but only its width is provided.',
      'Estimate journey time, but the starting location is missing.',
      'Choose the cheaper offer, but neither offer has a price.',
      'Compute average speed, but the distance is absent.',
      'Summarise the attached document, but no document is present.',
      'Tell whether the room is larger than the other room, but neither room has dimensions.',
      'Calculate a percentage change, but the initial value is absent.',
      'Compare two dates, but the year is missing and changes the ordering.',
      'Schedule the task for next Friday, but two users mean different local dates.',
      'Pick a suitable cable, but neither device connector is specified.',
      'Calculate a weighted mean, but no weights are supplied.',
      'Find the volume of a cylinder, but its height is missing.',
      'Explain why this program failed, but no code or error message is available.']
    for q in ambiguous:add('ambiguity',q+' What should the assistant do first?','Ask for the essential missing information',['Invent a plausible value','State a definite result without the information','Ignore the request and announce completion'])
    tasks=[
      'Find the current train departures for a specified station using live data.',
      'Check whether a named flight is delayed right now.',
      'Research and compare the most recent releases of three specified software packages.',
      'Review a supplied fifty-page technical report and cross-check its references.',
      'Investigate an intermittent distributed-system failure across a large supplied log archive.',
      'Check the current opening hours of a named museum for a specified date.',
      'Retrieve the latest weather warning for a specified region.',
      'Compare current prices and stock of a specified component at several vendors.',
      'Check the newest official documentation for a recently changed API.',
      'Research a detailed multi-source account of a scientific controversy.',
      'Analyse a supplied large code repository for several interacting security bugs.',
      'Reconcile a large supplied set of inconsistent experimental results.',
      'Find current accessibility arrangements at a named venue.',
      'Research a complete installation migration using the latest official compatibility matrix.',
      'Find whether a named public service is reporting an outage now.',
      'Compare the latest published versions of three technical standards.',
      'Trace the causes of a complex performance regression across a supplied multi-service trace.',
      'Review a long mathematical proof and verify each nontrivial step.',
      'Research a detailed literature comparison with primary-source citations.',
      'Check the current status of a specified software issue in its public tracker.',
      'Retrieve a newly published dataset and examine its documented limitations.',
      'Investigate several competing explanations for a supplied complex simulation failure.',
      'Check a specified live event timetable against its official source.',
      'Compare a large supplied configuration set and produce a tested migration plan.']
    for q in tasks:add('delegation',q+' All essential identifiers and inputs are supplied. Which action fits the live-voice policy?','Delegate the research or analysis with the supplied context',['Invent the answer immediately','Ask for an identifier already supplied','Ignore the request as if cancelled'])
    for n in range(4):
        add('stale',f'Request R{n} was cancelled and cancellation was confirmed. Its result arrives afterwards. What should happen?','Ignore the cancelled result',['Speak the result immediately','Attach it to a new unrelated request','Claim the cancellation failed without evidence'])
        add('stale',f'The required lookup backend for request S{n} is unavailable. No cached answer is available. What should the assistant say?','Explain that the lookup is unavailable and offer an appropriate next step',['Invent a lookup result','Claim a successful lookup','Use a result from a cancelled unrelated task'])
    assert len(rows)==128 and len({r['question'] for r in rows})==128
    return dict(schema='dl-authored-text-reasoning128-v1',created_date='2026-11-07',not_audio_or_end_to_end_delegation=True,system=SYSTEM,cases=rows)


def messages(plan,row):
    return [dict(role='system',content=plan['system']),dict(role='user',content=row['question']+'\n'+'\n'.join(f'{label}. {option}' for label,option in zip('ABCD',row['options'])))]


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=build()
    with a.output.open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(hashlib.sha256(a.output.read_bytes()).hexdigest())
