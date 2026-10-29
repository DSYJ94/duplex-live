"""Evidence for OPAL/supervisor reviews; signals are not causal diagnoses.

TRAIN/DEV aggregates only. Never inspect held-out answers, rewrite targets,
change folds, prune failing cases, or move data during an optimizer update.
"""
from collections import defaultdict
import math

FAMILIES=('ordinary','delegate','quiet','cancel','pause','pending')
FIELDS=('ce','decision_ce','text_ce','kl','hidden')


def aggregate(rows):
    """Each item is (family, parts, count); preserve missing group losses as None."""
    values=defaultdict(list)
    for family,parts,count in rows:
        if family not in FAMILIES or type(count) is not int or count<1:raise ValueError('Known family and actual sample count required')
        values[family].append((parts,count))
    out={}
    for family,items in values.items():
        result={'count':sum(n for _,n in items)}
        for key in FIELDS:
            measured=[(p[key],p.get(key+'_count',n)) for p,n in items if p.get(key) is not None]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v,_ in measured):raise ValueError('Finite actual losses required')
            result[key+'_count']=sum(n for _,n in measured)
            result[key]=sum(v*n for v,n in measured)/sum(n for _,n in measured) if measured else None
        out[family]=result
    return out


def coverage(source):
    result={}
    for fold in ('TRAIN','DEV'):
        groups={}
        for family in FAMILIES:
            rows=[r for r in source['scenes'] if r['fold']==fold and r['family']==family]
            hashes=[r.get('user_wav_sha256') for r in rows]
            known=all(isinstance(v,str) and len(v)==64 for v in hashes)
            groups[family]=dict(scenes=len(rows),unique_user_timelines=len(set(hashes)) if known else None)
        result[fold]=groups
    return result


def flatten(groups,prefix):
    return {prefix+'_'+family+'_'+key:float(v) for family,parts in groups.items() for key,v in parts.items() if v is not None}


def mean_ce(groups):
    measured=[(p['ce'],p['count']) for p in groups.values() if p.get('ce') is not None]
    return sum(v*n for v,n in measured)/sum(n for _,n in measured) if measured else None


def review(rows,step,previous,*,dev=None,data_coverage=None):
    entries=[(family,parts,parts['count']) for row in rows for family,parts in row.get('family_train_metrics',{}).items()]
    train=aggregate(entries);metrics=flatten(train,'train_window');signals=[]
    current_train=mean_ce(train);current_dev=mean_ce(dev or {})
    old_train=previous.get('train_ce');old_dev=previous.get('dev_ce')
    if current_train is not None and old_train is not None:metrics['train_ce_delta']=current_train-old_train
    if current_dev is not None and old_dev is not None:metrics['dev_ce_delta']=current_dev-old_dev
    if metrics.get('train_ce_delta',0)<0 and metrics.get('dev_ce_delta',0)>0:
        signals.append('train_dev_divergence: coverage or regularisation; loss alone does not distinguish them')
    clipped=[r['gradient_norm']>r['gradient_clip'] for r in rows if r.get('gradient_clip') is not None]
    if clipped:metrics['observed_reference_clip_fraction']=sum(clipped)/len(clipped)
    if dev:metrics.update(flatten(dev,'dev'))
    if data_coverage:
        for fold,groups in data_coverage.items():
            for family,parts in groups.items():
                for key,value in parts.items():
                    if value is not None:metrics['data_'+fold.lower()+'_'+family+'_'+key]=float(value)
    packet=dict(step=step,window_steps=[r['step'] for r in rows],train=train,dev=dev,
        data_coverage=data_coverage,signals=signals,agent_metrics=metrics,
        causal_diagnosis_established=False,heldout_available=False,
        review_contract=['Compare data quality/coverage, optimizer dynamics, objective/speech mismatch and runtime separately.',
            'Low unique-user counts can be intentional matched-world design; do not discard counterfactual pairs.',
            'If evidence supports a recipe change, change one factor with program/state recorded.',
            'If data/targets need repair, request a new version between phases; never silently alter current corpus or accept a passing subset.',
            'Use native DEV speech/control evidence alongside loss; absent native evidence is unknown, not a pass.'])
    state=dict(train_ce=current_train if current_train is not None else old_train,
        dev_ce=current_dev if current_dev is not None else old_dev,last_review_step=step)
    return packet,state
