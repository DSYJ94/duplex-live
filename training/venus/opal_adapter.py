"""Optional upstream OPAL seam; no provider calls or runtime installation.

The native trainer must explicitly opt in after its keep/parity and restoration
checks. Dataset/loss/adoption gates remain owned by the native training driver.
"""
from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import subprocess
import sys

UPSTREAM_COMMIT = '8b16759a3e9930916ed582e28068267859f935f8'


def load_upstream(source):
    """Use an unmodified pinned checkout, never the unrelated `opal` package."""
    source = Path(source).resolve()
    head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True)
    if head != UPSTREAM_COMMIT or dirty:
        raise ValueError('Use the exact unmodified, reviewed OPAL checkout')
    sys.path.insert(0,str(source))
    try:
        opal = importlib.import_module('opal')
    finally:
        sys.path.remove(str(source))
    if Path(opal.__file__).resolve().parent != source/'opal':
        raise ValueError('Another package named opal was already imported')
    return opal


@dataclass(frozen=True)
class Options:
    total_steps: int
    goal: str
    decision_updates: tuple = (16, 64, 128, 256)
    every: int = 8
    backend: str = 'keep'
    maximum_agent_cost_usd: float = 0.
    decision_timeout_seconds: int = 120
    unavailable_timeout_seconds: int = 30
    local_git_repo: str = ''

    def validate(self):
        if type(self.total_steps) is not int or self.total_steps < 1 or not isinstance(self.goal,str) or not self.goal.strip():
            raise ValueError('Explicit total updates and the existing product goal are required')
        if type(self.every) is not int or self.every < 1:
            raise ValueError('Positive observation interval required')
        if not isinstance(self.backend,str) or (self.backend != 'keep' and not self.backend.startswith(('mailbox:','local-git:'))):
            raise ValueError('Only keep, an explicit mailbox, or the zero-API LOCAL Git backend is admitted')
        if self.backend.startswith('mailbox:') and not Path(self.backend[len('mailbox:'):]).is_absolute():
            raise ValueError('Mailbox requires an explicit absolute folder')
        if type(self.maximum_agent_cost_usd) not in (int,float) or not math.isfinite(self.maximum_agent_cost_usd) or self.maximum_agent_cost_usd < 0:
            raise ValueError('Finite nonnegative cost cap required')
        local=self.backend.startswith('local-git:')
        if local and (not Path(self.backend[len('local-git:'):]).is_absolute()
                or not self.local_git_repo or not Path(self.local_git_repo).is_absolute()
                or self.maximum_agent_cost_usd!=0):
            raise ValueError('LOCAL Git requires explicit absolute decision/repository paths and exactly zero API budget')
        if not local and self.local_git_repo:
            raise ValueError('LOCAL repository belongs only to the explicit zero-API Git backend')
        if self.backend != 'keep' and not local and self.maximum_agent_cost_usd == 0:
            raise ValueError('An explicit positive agent budget is required before mailbox requests')
        if (type(self.decision_timeout_seconds) is not int or type(self.unavailable_timeout_seconds) is not int
                or not 1 <= self.decision_timeout_seconds <= 300 or not 0 <= self.unavailable_timeout_seconds <= 60):
            raise ValueError('Agent waiting must be finite and explicitly bounded')
        if any(type(step) is not int or step < 1 for step in self.decision_updates):
            raise ValueError('Decision updates must be positive integers')


def wrap(optimizer, model, *, source, run_dir, options, scheduler=None, clip_grad_norm=None, initial_program=None):
    """Wrap once after restoring original Adam and constructing its scheduler.

    Move existing clipping into this argument, do not also clip in the driver.
    The opt-in driver must call step(loss=...), then its scheduler, and save the
    FULL wrapped state_dict. Keeping the old Adam-only comparison is insufficient.
    """
    options.validate()
    opal = load_upstream(source)
    runtime_backend=options.backend
    if options.backend.startswith('local-git:'):
        from .opal_supervisor import local_backend
        runtime_backend=local_backend(options.backend[len('local-git:'):],options.local_git_repo,
                                      timeout=options.decision_timeout_seconds)
    schedule = sorted({min(step,options.total_steps) for step in options.decision_updates})
    if not schedule:
        schedule = [options.total_steps]
    arguments = dict(total_steps=options.total_steps, goal=options.goal, scheduler=scheduler,
        schedule=schedule, every=options.every, max_decisions=len(schedule),
        backend=runtime_backend, blocking=False, summarize=False, share_code=False,
        run_dir=str(run_dir), clip_grad_norm=clip_grad_norm, snapshot_limit_gb=.25)
    if initial_program is not None:
        arguments['program'] = str(initial_program)
    if options.backend.startswith('local-git:'):
        # Upstream assumes a minimum $0.01 cost for each request even for a
        # local callback. This backend invokes no model/CLI/API and returns
        # exactly zero cost; finite decision counts and waiting still apply.
        # Paid/mailbox backends retain their original positive USD cap.
        arguments['max_cost_usd']=None
    elif options.backend != 'keep':
        arguments.update(max_cost_usd=options.maximum_agent_cost_usd,
            backend_options=dict(timeout=options.decision_timeout_seconds,
                unavailable_timeout=options.unavailable_timeout_seconds,broker_wait=15,poll=1))
    return opal.wrap(optimizer,model,**arguments)


def checkpoint_manifest(optimizer):
    """Require real OPAL program/state, instead of silently saving only Adam."""
    state = optimizer.state_dict()
    required = {'opal','update','version','files','updater','user_optimizers','user_schedulers','observations','memory'}
    if not required <= state.keys() or state['opal'] != 1 or state['updater'] is None:
        raise ValueError('Complete OPAL program/optimizer/observation state is required')
    return dict(upstream_commit=UPSTREAM_COMMIT, update=state['update'], program_version=state['version'],
        programs={name:hashlib.sha256(text.encode()).hexdigest() for name,text in state['files'].items()},
        complete_wrapped_state_required=True)


def log_observation(optimizer, values, *, fold):
    """Finite TRAIN/DEV aggregates only, never sample text or held-out answers."""
    if fold not in ('TRAIN','DEV') or not values or any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
        raise ValueError('Finite aggregate TRAIN/DEV values only')
    if any(word in key.lower() for key in values for word in ('holdout','test','answer','transcript','owner')):
        raise ValueError('Evaluation answers and owner content cannot be agent metrics')
    optimizer.log({str(key):float(value) for key,value in values.items()},show_agent=True)


def log_development(optimizer,values,*,fold):
    if fold!='DEV':raise ValueError('Development metrics require DEV fold')
    return log_observation(optimizer,values,fold=fold)
