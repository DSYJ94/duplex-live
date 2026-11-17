"""Deterministic complete-document language epochs; no qualification by inference.

The phase must supply already reviewed, tokenized TRAIN documents. Raw corpus
candidates cannot enter this schedule. Source and qualification ledgers remain
part of the corpus identity; this module does not create their approval.
"""
import hashlib
import json
import math
import random
from dataclasses import asdict

from .text_rehearsal import pack


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
                                     separators=(',',':')).encode()).hexdigest()


def hash_value(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def build(documents,spec,*,tokenizer_sha256,qualification_sha256,world_size,epochs,
          max_global_target_tokens,coefficient,seed,updates_per_epoch=None):
    if (any(type(v) is not int or v<1 for v in (world_size,epochs,max_global_target_tokens))
            or type(seed) is not int or not hash_value(tokenizer_sha256) or not hash_value(qualification_sha256)
            or isinstance(coefficient,bool) or not isinstance(coefficient,(int,float))
            or not math.isfinite(coefficient) or coefficient<=0):
        raise ValueError('Explicit bounded geometry, mixture and immutable qualification/tokenizer required')
    documents=sorted(documents,key=lambda row:row['id'])
    if not documents or len({row['id'] for row in documents})!=len(documents):
        raise ValueError('Distinct complete language documents required')
    if updates_per_epoch is not None and (type(updates_per_epoch) is not int or not 1<=updates_per_epoch<=len(documents)):
        raise ValueError('Explicit nonempty retained-cycle update count required')
    populations={}
    for row in documents:
        if (not isinstance(row['id'],str) or not row['id'] or row.get('fold')!='TRAIN'
                or row.get('semantic_qualified') is not True or row.get('leakage_qualified') is not True
                or row.get('qualification_sha256')!=qualification_sha256
                or row.get('tokenizer_sha256')!=tokenizer_sha256
                or not hash_value(row.get('source_sha256'))):
            raise ValueError('Source-bound independently qualified TRAIN documents only')
        batch=pack(row['tokens'],spec,supervise=row['mask'])
        count=int(batch['mask'].sum())
        if count>max_global_target_tokens:
            raise ValueError('Complete document exceeds update budget; no silent truncation')
        populations[row['id']]=count
    geometry=asdict(spec);geometry['reserved_tokens']=sorted(spec.reserved_tokens)
    corpus_sha=digest(dict(documents=documents,spec=geometry))
    updates=[]
    for epoch in range(epochs):
        order=[row['id'] for row in documents];random.Random(seed+epoch).shuffle(order)
        if updates_per_epoch is not None:
            # Match the existing retained/expanded cycle length rather than
            # rescaling or repeating its speech losses. Stable shuffled ties.
            buckets=[[] for _ in range(updates_per_epoch)];counts=[0]*updates_per_epoch
            for identity in sorted(order,key=lambda value:-populations[value]):
                index=min(range(updates_per_epoch),key=lambda value:counts[value])
                buckets[index].append(identity);counts[index]+=populations[identity]
            if any(not bucket for bucket in buckets) or max(counts)>max_global_target_tokens:
                raise ValueError('Complete language epoch cannot fit declared native cycle/token budget')
            updates.extend(dict(epoch=epoch,document_ids=bucket,global_target_tokens=count)
                           for bucket,count in zip(buckets,counts))
            continue
        current=[];count=0
        for identity in order:
            if current and count+populations[identity]>max_global_target_tokens:
                updates.append(dict(epoch=epoch,document_ids=current,global_target_tokens=count))
                current=[];count=0
            current.append(identity);count+=populations[identity]
        if current:updates.append(dict(epoch=epoch,document_ids=current,global_target_tokens=count))
    schedule=dict(schema='native-language-complete-epochs-v1',world_size=world_size,epochs=epochs,
                  seed=seed,max_global_target_tokens=max_global_target_tokens,updates=updates,
                  updates_per_epoch=updates_per_epoch,
                  corpus_sha256=corpus_sha,tokenizer_sha256=tokenizer_sha256,
                  qualification_sha256=qualification_sha256,coefficient=coefficient)
    schedule_sha=digest(schedule)
    return dict(schedule,identity=dict(corpus_sha256=corpus_sha,tokenizer_sha256=tokenizer_sha256,
                                      schedule_sha256=schedule_sha,coefficient=coefficient))


def window(plan,documents,spec,*,rank,next_update,device='cpu'):
    """Return exactly the next immutable update; never wrap/reset an exhausted run."""
    body={k:v for k,v in plan.items() if k!='identity'}
    expected=dict(corpus_sha256=body['corpus_sha256'],tokenizer_sha256=body['tokenizer_sha256'],
                  schedule_sha256=digest(body),coefficient=body['coefficient'])
    if expected!=plan['identity']:
        raise ValueError('Immutable language schedule changed')
    if (type(rank) is not int or not 0<=rank<plan['world_size'] or type(next_update) is not int
            or not 0<=next_update<len(plan['updates'])):
        raise ValueError('Exact unfinished next update and rank required')
    geometry=asdict(spec);geometry['reserved_tokens']=sorted(spec.reserved_tokens)
    ordered=sorted(documents,key=lambda row:row['id'])
    if digest(dict(documents=ordered,spec=geometry))!=plan['identity']['corpus_sha256']:
        raise ValueError('Language documents/spec changed after scheduling')
    indexed={row['id']:row for row in ordered}
    row=plan['updates'][next_update];ids=row['document_ids'];world=plan['world_size']
    batches=[];local_ids=[];padding=[]
    for start in range(0,len(ids),world):
        valid=start+rank<len(ids)
        identity=ids[start+rank] if valid else ids[start]
        document=indexed[identity]
        batch=pack(document['tokens'],spec,supervise=document['mask'],device=device)
        if not valid:batch['mask'].zero_()
        batches.append(batch);local_ids.append(identity);padding.append(not valid)
    return dict(identity=plan['identity'],step=next_update,batches=batches,spec=spec,
                document_ids=local_ids,padding_only=padding,epoch=row['epoch'])
