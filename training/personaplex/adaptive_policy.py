"""Pure, zero-API evaluated-boundary policy; no training or file mutation."""
from dataclasses import dataclass, asdict
import hashlib,json,math,re,time
from pathlib import Path

FACTORS = (.5, .75, 1., 1.1)
SCHEMA = 'dl-native-opal-boundary-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


@dataclass(frozen=True)
class Policy:
    epoch_updates:int=49
    total_updates:int=196
    train_count:int=1008
    retained_count:int=323
    dev_count:int=64
    regression_fraction:float=.02
    improvement_fraction:float=.01
    absolute_noise:float=1e-6
    request_seconds:int=120
    behaviour_admission_sha256:str|None=None

    def __post_init__(self):
        for value in (self.epoch_updates,self.total_updates,self.train_count,self.retained_count,self.dev_count,self.request_seconds):
            if type(value)is not int or value<=0:raise ValueError('Positive exact policy counts required')
        if self.total_updates%self.epoch_updates:raise ValueError('Only complete epoch schedules admitted')
        for value in (self.regression_fraction,self.improvement_fraction,self.absolute_noise):
            if type(value)not in(int,float)or not math.isfinite(value)or value<=0:raise ValueError('Finite positive policy margins required')
        if self.behaviour_admission_sha256 is not None and not sha(self.behaviour_admission_sha256):raise ValueError('Pinned DEV qualification hash required')

    @property
    def identity(self):return digest(asdict(self))


def metrics(value):
    if not isinstance(value,dict) or set(value)!={'text','audio'}:return False
    return all(type(x)in(int,float) and math.isfinite(x) and x>=0 for x in value.values())


def decide(request, *, policy=Policy(), now=None):
    now=time.time()if now is None else now
    if request.get('schema')!=SCHEMA or request.get('policy_sha256')!=policy.identity:
        raise ValueError('Exact boundary policy and schema required')
    step,boundary=request['resume_step'],request['boundary']
    if type(step)is not int or type(boundary)is not int or not 0<boundary<=step<=policy.total_updates or boundary!=step//policy.epoch_updates*policy.epoch_updates:
        raise ValueError('Latest completed evaluation boundary required')
    if any(not sha(request.get(k))for k in ('data_sha256','checkpoint_sha256','backup_sha256','dev_sha256','previous_dev_sha256')):
        raise ValueError('Exact data, checkpoint, backup and DEV report hashes required')
    created=request.get('created_unix')
    if type(created)not in(int,float) or not math.isfinite(created) or not 0<=now-created<=policy.request_seconds:
        raise ValueError('Boundary request expired or has a future clock')
    allowed={'schema','policy_sha256','boundary','resume_step','data_sha256','checkpoint_sha256','backup_sha256','dev_sha256','previous_dev_sha256','created_unix','current_multiplier','coverage','backup_complete','previous_dev','dev','retention','behaviour','artifacts','train_summary'}
    if set(request)-allowed:raise ValueError('Unknown fields; only aggregate TRAIN/DEV evidence is allowed')
    current=request['current_multiplier']
    if type(current)not in(int,float)or current not in FACTORS:raise ValueError('Unknown current catalogue multiplier')
    result=dict(request_sha256=digest(request),policy_sha256=policy.identity,boundary=boundary,
                resume_step=step,action='keep',multiplier=current,reason='',recommendations=[])
    expected=dict(new_train=policy.train_count,retained=policy.retained_count,dev=policy.dev_count)
    if request.get('coverage')!=expected or not request.get('backup_complete')is True:
        return dict(result,action='defer',reason='Complete TRAIN/rehearsal/DEV coverage and verified backup required')
    if step==policy.total_updates:
        return dict(result,action='next_phase',reason='Scope complete; no extra updates authorized')
    before,after=request.get('previous_dev'),request.get('dev')
    if not metrics(before)or not metrics(after):return dict(result,action='defer',reason='Complete finite DEV text/audio metrics required')
    relative={k:(after[k]-before[k])/max(before[k],policy.absolute_noise)for k in before}
    regress=any(after[k]-before[k]>max(policy.absolute_noise,policy.regression_fraction*before[k])for k in before)
    improve=all(before[k]-after[k]>max(policy.absolute_noise,policy.improvement_fraction*before[k])for k in before)
    lower=[x for x in FACTORS if x<current]
    if regress:
        factor=max(lower)if lower else current
        return dict(result,action='adapt'if factor!=current else'keep',multiplier=factor,
                    reason='DEV regression beyond configured noise margin; reduce the effective scheduled rate'if lower else'LR floor reached; diagnose data/capacity instead of further shrinking',
                    recommendations=['review TRAIN failure families and retention; retain full rehearsal'])
    retention=request.get('retention')
    stable=False
    if isinstance(retention,dict)and set(retention)=={'previous','current'}:
        a,b=retention['previous'],retention['current']
        if isinstance(a,dict)and isinstance(b,dict)and set(a)==set(b)=={'kl','hidden'}:
            stable=all(type(x)in(int,float)and math.isfinite(x)and x>=-policy.absolute_noise for x in (*a.values(),*b.values()))and all(b[k]<=a[k]+max(policy.absolute_noise,policy.regression_fraction*abs(a[k]))for k in a)
    behaviour=request.get('behaviour')or{}
    if not isinstance(behaviour,dict):raise ValueError('Aggregate DEV behaviour mapping required')
    approved=behaviour.get('qualified_DEV')is True and behaviour.get('passed')is True and sha(behaviour.get('report_sha256'))and behaviour.get('report_sha256')==policy.behaviour_admission_sha256
    if behaviour.get('qualified_DEV')is True and behaviour.get('passed')is False:
        return dict(result,reason='Behaviour remains inadequate despite loss trends; do not increase rate',recommendations=['MiRL-style TRAIN failure recovery; qualify native behaviour before RL'])
    higher=[x for x in FACTORS if current<x<=1.1]
    if improve and stable and higher and (min(higher)<=1. or approved):
        return dict(result,action='adapt',multiplier=min(higher),reason='Both DEV streams improved with stable retention; return toward admitted rate or use pinned DEV behaviour qualification')
    result['reason']='No justified change: DEV trade-off/noise or missing retention/behaviour evidence'
    result['relative_dev_change']=relative
    return result


def program(multiplier):
    if type(multiplier)not in(int,float)or multiplier not in FACTORS:raise ValueError('Unapproved catalogue multiplier')
    path=Path(__file__).resolve().parents[1]/'venus/opal_programs/distributed_lr_v1/updater.py'
    text=path.read_text();marker='LR_MULTIPLIER = 1.0'
    if text.count(marker)!=1:raise ValueError('Catalogue template identity changed')
    return text.replace(marker,'LR_MULTIPLIER = '+repr(float(multiplier)))
