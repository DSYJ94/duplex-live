"""One-factor stability trial: half the scheduled AdamW step, same state/loss.

This is a candidate program, admitted only by a live, rationale-bound OPAL
request. It is not evidence that learning rate caused any observed failure.
"""
import torch

LR_MULTIPLIER = .5
CLIP_GRAD_NORM = 1.


class Updater:
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opts = spec['user_optimizers']
        self.scheds = spec['user_schedulers']
        self.last_effective_rates = []
        self.last_update = None

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def step(self, grads, update):
        torch.nn.utils.clip_grad_norm_([p for _, p in self.named], CLIP_GRAD_NORM)
        groups = self.param_groups
        nominal = [g['lr'] for g in groups]
        effective = [rate * LR_MULTIPLIER for rate in nominal]
        try:
            for group, rate in zip(groups, effective):
                group['lr'] = rate
            for optimizer in self.opts:
                optimizer.step()
        finally:
            # The original scheduler owns the nominal trajectory. Do not halve
            # it repeatedly, reset moments, or edit the scheduler's base rate.
            for group, rate in zip(groups, nominal):
                group['lr'] = rate
        self.last_effective_rates = effective
        self.last_update = update

    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts],
                    scheds=[s.state_dict() for s in self.scheds],
                    last_effective_rates=self.last_effective_rates,
                    last_update=self.last_update)

    def load_state_dict(self, state):
        if len(state['opts']) != len(self.opts) or len(state['scheds']) != len(self.scheds):
            raise ValueError('Preserve all existing optimizer and scheduler states')
        for optimizer, saved in zip(self.opts, state['opts']):
            optimizer.load_state_dict(saved)
        for scheduler, saved in zip(self.scheds, state['scheds']):
            scheduler.load_state_dict(saved)
        self.last_effective_rates = list(state.get('last_effective_rates', []))
        self.last_update = state.get('last_update')


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
