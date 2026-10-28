"""Nonblocking, bounded agent decisions for the native adapter trainer.

This is an OPAL-inspired control seam, not a reproduction of OPAL. The agent
reads numerical observations and supplies immutable JSON decisions. Training
keeps going when there is no decision. No code evaluation, provider calls,
dataset changes or held-out examples enter this interface.
"""
import hashlib
import json
import math
import os
from pathlib import Path


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


class ReactiveControl:
    """One run identity, ordered decisions, and checkpointed control state."""

    def __init__(self, folder, identity, restored=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.identity = canonical(identity)
        self.state = dict(identity=self.identity, lr_multiplier=1., gradient_clip=1.,
                          decisions=[], observations=[])
        if restored is not None:
            if restored['identity'] != self.identity:
                raise ValueError('Reactive checkpoint run identity mismatch')
            self.state = json.loads(json.dumps(restored, allow_nan=False))
            self._validate_state()

    def _validate_state(self):
        s = self.state
        if not (.25 <= s['lr_multiplier'] <= 1. and .5 <= s['gradient_clip'] <= 1.):
            raise ValueError('Reactive controls exceed bounded recipe')
        for index, item in enumerate(s['decisions'], 1):
            if item['sequence'] != index or item['identity'] != self.identity:
                raise ValueError('Reactive decision history mismatch')
        if s['decisions'] and any(s[name] != s['decisions'][-1][name] for name in ('lr_multiplier','gradient_clip')):
            raise ValueError('Reactive checkpoint controls differ from their decision receipt')

    def snapshot(self):
        return json.loads(json.dumps(self.state, allow_nan=False))

    def observe(self, completed_step, rows, dev=None):
        if completed_step < 2 or not rows:
            raise ValueError('Observations follow the independent restore proof')
        keys = ('ce', 'kl', 'hidden', 'gradient_norm', 'seconds', 'lr')
        metrics = {k: sum(float(r[k]) for r in rows) / len(rows) for k in keys}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError('Nonfinite reactive observation')
        families = {}
        for row in rows:
            for family in set(row['families']):
                families.setdefault(family, []).append(row['ce'])
        value = dict(identity=self.identity, completed_step=completed_step,
                     mean=metrics, maximum_gradient_norm=max(r['gradient_norm'] for r in rows),
                     batch_ce_by_present_family={k: sum(v)/len(v) for k, v in families.items()},
                     family_metric_caveat='Batch CE grouped by families present; not per-scene CE',
                     full_dev=dev, holdout_available=False,
                     lr_multiplier=self.state['lr_multiplier'],
                     gradient_clip=self.state['gradient_clip'])
        observation = dict(**value, sha256=canonical(value))
        self.state['observations'].append(dict(step=completed_step, sha256=observation['sha256']))
        atomic(self.folder / ('observation-%04d.json' % completed_step), observation)
        atomic(self.folder / 'latest.json', observation)
        return observation

    def boundary(self, completed_step):
        """Apply at most one valid current decision; never wait for an agent."""
        if completed_step < 2:
            return False
        sequence = len(self.state['decisions']) + 1
        path = self.folder / ('decision-%03d.json' % sequence)
        if not path.exists():
            return False
        raw = path.read_bytes()
        if len(raw) > 16384:
            raise ValueError('Reactive decision exceeds bounded size')
        decision = json.loads(raw)
        required = {'identity', 'sequence', 'observation_sha256', 'not_before_step',
                    'expires_after_step', 'lr_multiplier', 'gradient_clip', 'reason'}
        if (set(decision) != required or decision['identity'] != self.identity
                or type(decision['sequence']) is not int or decision['sequence'] != sequence):
            raise ValueError('Reactive decision identity/schema mismatch')
        if len(self.state['decisions']) >= 8:
            raise ValueError('Reactive decision limit reached')
        steps = (decision['not_before_step'], decision['expires_after_step'])
        if any(type(v) is not int for v in steps) or not 2 <= steps[0] <= steps[1]:
            raise ValueError('Reactive decision step window invalid')
        if not isinstance(decision['reason'], str) or not decision['reason'].strip() or len(decision['reason']) > 2000:
            raise ValueError('Reactive decision requires a bounded explanation')
        for name, low, high in (('lr_multiplier', .25, 1.), ('gradient_clip', .5, 1.)):
            value = decision[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Reactive decision exceeds bounded recipe')
        if completed_step < steps[0]:
            return False
        known = [o for o in self.state['observations'] if o['sha256'] == decision['observation_sha256']]
        if not known or known[-1]['step'] > completed_step:
            raise ValueError('Reactive decision lacks a real earlier observation')
        if completed_step > steps[1]:
            raise ValueError('Reactive decision expired; do not apply stale advice')
        self.state['lr_multiplier'] = float(decision['lr_multiplier'])
        self.state['gradient_clip'] = float(decision['gradient_clip'])
        receipt = dict(**decision, applied_after_step=completed_step,
                       decision_sha256=hashlib.sha256(raw).hexdigest())
        self.state['decisions'].append(receipt)
        atomic(self.folder / ('applied-%03d.json' % sequence), receipt)
        return True
