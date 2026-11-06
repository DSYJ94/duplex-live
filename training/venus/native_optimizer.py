"""Native OPAL ownership of updates, scheduling and complete resumable state.

No service/model calls are made here. Keep mode is the default. The training
recipe, data, loss, update admission and final evaluation stay with train.py.
"""
import math
import json,copy
from pathlib import Path
import torch
from .opal_adapter import Options, wrap, checkpoint_manifest, log_development,log_observation
from .state import exact_state


def lr_multiplier(step, *, warmup, updates):
    if step < warmup:
        return (step+1)/warmup
    return .5*(1+math.cos(math.pi*(step-warmup)/(updates-warmup)))


def gradient_norm(parameters):
    # Observe only. Clipping is owned exactly once by the OPAL updater.
    grads=[p.grad.detach() for p in parameters]
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g,2) for g in grads]),2)


class NativeOpal:
    def __init__(self, base, model, *, source, run_dir, recipe, backend='keep', agent_budget=0.,
                 goal='Improve native duplex control and ordinary retention; full native speech and behavior decide acceptance', initial_program=None):
        if recipe['updates'] <= recipe['warmup_updates']:
            raise ValueError('Native OPAL requires the admitted full schedule beyond warmup')
        self.run_dir=Path(run_dir)
        self.diagnostics_state=dict(train_ce=None,dev_ce=None,last_review_step=0)
        self.base=base
        for group in base.param_groups:
            group['lr']=group['initial_lr']=recipe['learning_rate']
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(base,
            lambda step:lr_multiplier(step,warmup=recipe['warmup_updates'],updates=recipe['updates']))
        self.options=Options(total_steps=recipe['updates'],goal=goal,backend=backend,
            maximum_agent_cost_usd=agent_budget,decision_updates=tuple(s for s in (16,32,64,128,256) if s<=recipe['updates']))
        self.optimizer=wrap(base,model,source=source,run_dir=run_dir,options=self.options,
            scheduler=self.scheduler,clip_grad_norm=recipe['gradient_clip'],initial_program=initial_program)
        self._closed=False

    def step(self,loss,step):
        before=checkpoint_manifest(self.optimizer)
        if before['update']!=step:
            raise ValueError('OPAL progress and native update index differ; never repeat saved updates')
        self.optimizer.step(loss=loss)
        self.scheduler.step()
        after=checkpoint_manifest(self.optimizer)
        if after['update']!=step+1:
            raise ValueError('Exactly one OPAL update is required per accumulated native batch')
        return after

    def development(self,values):
        log_development(self.optimizer,{'dev_'+family+'_ce':value for family,value in values.items()},fold='DEV')

    def training_batch(self,groups):
        from .diagnostics import flatten
        log_observation(self.optimizer,flatten(groups,'train_batch'),fold='TRAIN')

    def diagnose(self,rows,step,*,dev=None,data_coverage=None):
        from .diagnostics import review
        packet,self.diagnostics_state=review(rows,step,self.diagnostics_state,dev=dev,data_coverage=data_coverage)
        log_observation(self.optimizer,packet['agent_metrics'],fold='DEV' if dev else 'TRAIN')
        folder=self.run_dir/'diagnostics';folder.mkdir(parents=True,exist_ok=True)
        path=folder/(f"step-{step:04d}-"+('dev' if dev else 'train')+'.json')
        with path.open('x') as stream:json.dump(packet,stream,indent=2)
        return packet

    def restore_diagnostics(self,state):
        if set(state)!={'train_ce','dev_ce','last_review_step'}:raise ValueError('Exact diagnostic state required')
        self.diagnostics_state=copy.deepcopy(state)

    def close(self):
        if not self._closed:
            self.optimizer.close();self._closed=True


def assert_optimizer_equal(expected,actual):
    """Compare every saved field except explicit run/transport bookkeeping."""
    if expected.get('opal')==1:
        if actual.get('opal')!=1:raise ValueError('Wrapped OPAL checkpoint cannot resume as plain Adam')
        required=('update','version','files','updater','memory','user_optimizers','user_schedulers',
                  'observations','ledger','follows','next_analysis','decision_schedule','decisions','cost')
        for key in required:
            if key not in expected or key not in actual or not exact_state(expected[key],actual[key]):
                raise ValueError('Fresh-process OPAL state differs: '+key+'; '+state_difference(expected.get(key),actual.get(key),key))
        # Fresh transport/directory identities can differ. Everything else,
        # including predecessor code, devices and development history, must not.
        excluded={'run_dir','event_offset','backend'}
        if set(expected)-excluded != set(actual)-excluded:
            raise ValueError('Fresh-process OPAL field set differs')
        for key in set(expected)-excluded:
            if not exact_state(expected[key],actual[key]):
                raise ValueError('Fresh-process OPAL state differs: '+key)
    elif not exact_state(expected,actual):
        raise ValueError('Fresh-process Adam state differs')


def state_difference(left,right,path):
    """One exact mismatch location; no omitted fields or relaxed comparison."""
    if isinstance(left,dict) and isinstance(right,dict):
        if left.keys()!=right.keys():return path+' missing='+repr(sorted(set(left)-set(right),key=str))+' extra='+repr(sorted(set(right)-set(left),key=str))
        for key in left:
            if not exact_state(left[key],right[key]):return state_difference(left[key],right[key],path+'/'+str(key))
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        if len(left)!=len(right):return path+' lengths='+str((len(left),len(right)))
        for index,(a,b) in enumerate(zip(left,right)):
            if not exact_state(a,b):return state_difference(a,b,path+'/'+str(index))
    if torch.is_tensor(left) and torch.is_tensor(right):return path+' tensor values differ; shapes='+str((tuple(left.shape),tuple(right.shape)))
    return path+' values='+repr((left,right))[:300]


def verified_parent(state,sha256,*,allow244=False,allow298=False,allow406=False):
    """Pin the two actual states; continuation244 must not replay190→244."""
    meta=state['metadata']
    if sha256=='10354f93ec2b5e224ae7298b2f3b8687f9dc74d2fbf87c1db62bd79763155b63':
        assert state['step']==190 and meta['source']=='2dcdf0f8b62400bda989fa68e88ebf291b967f62'
        return 190
    if allow244 and sha256=='c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270':
        assert state['step']==54 and meta['parent_training_steps']==190
        assert meta['source']=='6ea80a5d2512d981d8ddf1d64c4675684667429a'
        return 244
    if allow298 and sha256=='e7340edd15537523db78120a97902399faa0af9caa46e875ec92e56889d6edef':
        assert state['step']==54 and meta['parent_training_steps']==244
        assert meta['source']=='dd767d65ac663e701142f2c882aeb5e1c2a2c6e5'
        assert state['optimizer']['opal']==1 and state['optimizer']['update']==54 and state['optimizer']['version']==1
        return 298
    if allow406 and sha256=='a934ce7e05f4728a0973408eafedaed219f89dacfbf5dc6fcb2bbf536f4378b6':
        assert state['step']==108 and meta['parent_training_steps']==298
        assert meta['source']=='90dc9ecf98e464f1754c0fda7763024663f82844'
        assert state['optimizer']['opal']==1 and state['optimizer']['update']==108
        return 406
    raise ValueError('Unknown or unapproved native continuation parent')
