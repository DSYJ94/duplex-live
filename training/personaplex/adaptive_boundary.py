"""Collective, checkpoint-boundary OPAL programme transactions.

Uses the reviewed pinned OPAL _install seam; no service calls, optimizer steps,
source hot edits or data changes. Native continuation owns checkpoint publication.
"""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from training.venus.state import rng_state,restore_rng
from training.venus.opal_adapter import UPSTREAM_COMMIT,checkpoint_manifest
from .adaptive_policy import Policy,decide,digest,program,FACTORS,sha


def state_hash(value):
    h=hashlib.sha256()
    def emit(tag,blob):
        h.update(tag.encode());h.update(len(blob).to_bytes(8,'big'));h.update(blob)
    def visit(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous();emit('tensor',str((str(t.dtype),tuple(t.shape))).encode());emit('bytes',t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):emit('array',str((x.dtype.str,x.shape)).encode());emit('bytes',x.tobytes())
        elif isinstance(x,dict):
            emit('dict',str(len(x)).encode())
            for k in sorted(x,key=lambda k:(type(k).__name__,repr(k))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            emit(type(x).__name__,str(len(x)).encode())
            for v in x:visit(v)
        else:emit(type(x).__name__,json.dumps(x,sort_keys=True,allow_nan=False).encode())
    visit(value);return h.hexdigest()


class BoundaryController:
    def __init__(self, active, *, policy=Policy()):
        self.active,self.policy=active,policy
        worker=active.optimizer.worker
        if active.options.backend!='boundary-local' or (worker is not None and getattr(worker.backend,'name',None)!='keep'):
            raise ValueError('Construct a boundary-local continuation; never attach to a live KEEP worker')
        if checkpoint_manifest(active.optimizer)['upstream_commit']!=UPSTREAM_COMMIT:
            raise ValueError('Pinned OPAL implementation required')
        if active.optimizer._clip_grad_norm is not None:raise ValueError('Distributed runtime must own clipping exactly once')
        self.state=dict(schema='dl-opal-boundary-state-v1',policy_sha256=policy.identity,
                        enabled=False,current_multiplier=1.,last_boundary=0,history=[])

    @property
    def opt(self):return self.active.optimizer

    def _gather(self,value):
        if not self.opt.dist:return [value]
        import torch.distributed as dist
        rows=[None]*self.opt.world;dist.all_gather_object(rows,value,group=self.opt.ctrl);return rows

    def _broadcast(self,value):
        if not self.opt.dist:return value
        import torch.distributed as dist
        packet=[value if self.opt.rank0 else None];dist.broadcast_object_list(packet,src=0,group=self.opt.ctrl);return packet[0]

    def _collective_error(self,error):
        errors=self._gather(error)
        if any(errors):raise ValueError('Boundary rejected on all ranks: '+repr(errors))

    def _snapshot(self):
        return dict(model=state_hash({n:p for n,p in self.opt.named}),
                    adam=state_hash(self.active.base.state_dict()),
                    scheduler=state_hash(self.active.scheduler.state_dict()),
                    update=self.opt.update,rng=rng_state())

    def _validate_context(self,context):
        if set(context)!={'resume_step','checkpoint_sha256','backup_sha256','data_sha256'}:raise ValueError('Exact checkpoint context fields required')
        if self.state['enabled']and self.opt.files['updater.py']!=program(self.state['current_multiplier']):raise ValueError('Active program changed outside the boundary controller')
        if self.opt.update!=context.get('resume_step'):raise ValueError('Use latest restored step; never replay')
        if any(not sha(context.get(k))for k in ('checkpoint_sha256','backup_sha256','data_sha256')):raise ValueError('Verified checkpoint/backup/data bindings required')
        if any(p.grad is not None for _,p in self.opt.named):raise ValueError('Checkpoint boundary requires cleared gradients')

    def _install(self,factor,context,reason):
        # Sharded model/Adam tensors legitimately differ by rank.
        # Compare shared clocks collectively and preserve each shard locally.
        before=self._snapshot();common={k:before[k] for k in ('scheduler','update')}
        peers=self._gather(common)
        if any(p!=peers[0]for p in peers):raise ValueError('Scheduler/update differ across ranks')
        files=dict(self.opt.files);files['updater.py']=program(factor)
        code_hash=hashlib.sha256(files['updater.py'].encode()).hexdigest()
        if len(set(self._gather(code_hash)))!=1:raise ValueError('Ranks disagree on catalogue program')
        previous_files=dict(self.opt.files)
        err=self.opt._install(files,previous=self.opt.updater)
        if not err:
            # Pinned OPAL restores a changed program with its saved fallback.
            # Publish that predecessor explicitly so fresh restore is exact.
            self.opt._prev_files=previous_files
            self.opt._prev_updater=None
        restore_rng(before['rng'])
        if err:raise RuntimeError('OPAL programme build failed; pre-decision checkpoint preserved: '+err)
        after=self._snapshot()
        unchanged=all(before[k]==after[k]for k in ('model','adam','scheduler','update'))and state_hash(before['rng'])==state_hash(after['rng'])
        checks=self._gather(dict(rank=self.opt.rank,unchanged=unchanged,rng_sha256=state_hash(after['rng'])))
        if not all(x['unchanged']for x in checks):raise RuntimeError('Transition postcondition failed; do not perform another update')
        manifest=checkpoint_manifest(self.opt)
        if len(set(self._gather(digest(manifest))))!=1:raise RuntimeError('Applied OPAL program manifests disagree')
        return dict(reason=reason,context=context,program=manifest,program_sha256=code_hash,
                    preserved_state=True,rank_checks=checks,new_optimizer_updates=0)

    def enable(self,context):
        error=None
        try:self._validate_context(context)
        except (ValueError,TypeError)as exc:error=str(exc)
        self._collective_error(error)
        if len(set(self._gather(digest(context))))!=1:raise ValueError('Checkpoint bindings differ across ranks')
        if self.state['enabled']:return dict(status='already_enabled',new_optimizer_updates=0)
        worker=self.opt.worker
        self._collective_error('Legacy decision still in flight'if worker is not None and worker.busy else None)
        # Preserve the old counters, cost, ledger and backend registry. Only
        # automatic decision scheduling changes in this explicit transition.
        previous_schedule=self._broadcast(self.opt.state_dict()['decision_schedule']if self.opt.rank0 else None)
        self.opt._set_schedule([])
        self.opt._sched_i=0
        record=self._install(self.state['current_multiplier'],context,'Enable local evaluated-boundary decisions; no LR change')
        record['previous_decision_schedule']=previous_schedule
        record['automatic_decision_schedule']=[]
        self.state['enabled']=True;self.state['history'].append(dict(kind='enabled',**record));return record

    def evaluate(self,request,context,*,now=None):
        error=None
        try:
            self._validate_context(context)
            if not self.state['enabled']:raise ValueError('Enable the boundary controller first')
        except (ValueError,TypeError)as exc:error=str(exc)
        self._collective_error(error)
        if len(set(self._gather(digest(context))))!=1:raise ValueError('Checkpoint bindings differ across ranks')
        packet=None
        if self.opt.rank0:
            try:
                if request['current_multiplier']!=self.state['current_multiplier']:raise ValueError('Stale recipe multiplier')
                if request['boundary']<=self.state['last_boundary']:raise ValueError('Boundary already consumed; no request replay')
                if any(request.get(k)!=context[k]for k in context):raise ValueError('Request/checkpoint binding mismatch')
                packet=dict(decision=decide(request,policy=self.policy,now=now),error=None)
            except (ValueError,KeyError,TypeError)as exc:packet=dict(decision=None,error=str(exc))
        packet=self._broadcast(packet)
        if packet['error']:
            record=dict(status='rejected',reason=packet['error'],new_optimizer_updates=0)
            self.state['history'].append(dict(kind='rejected',**record))
            return record
        decision=packet['decision'];record=dict(decision=decision,applied=False,new_optimizer_updates=0)
        if decision['action']=='adapt':
            record.update(self._install(decision['multiplier'],context,decision['reason']),applied=True)
            self.state['current_multiplier']=decision['multiplier']
        if decision['action']in('keep','adapt','next_phase'):self.state['last_boundary']=decision['boundary']
        self.state['history'].append(dict(kind='decision',**record))
        if len(self.state['history'])>32:raise RuntimeError('Boundary attempt limit exceeded')
        return record

    def state_dict(self):return copy.deepcopy(self.state)

    def load_state_dict(self,value):
        if value.get('schema')!='dl-opal-boundary-state-v1' or value.get('policy_sha256')!=self.policy.identity:raise ValueError('Exact controller policy state required')
        if type(value.get('enabled'))is not bool:raise ValueError('Invalid controller enabled flag')
        if value.get('current_multiplier')not in FACTORS or type(value.get('last_boundary'))is not int or value['last_boundary']>self.opt.update:raise ValueError('Invalid controller cursor')
        if value.get('enabled') and self.opt.files['updater.py']!=program(value['current_multiplier']):raise ValueError('Controller and restored OPAL program differ')
        if not isinstance(value.get('history'),list)or len(value['history'])>32:raise ValueError('Invalid decision ledger')
        self.state=copy.deepcopy(value)
