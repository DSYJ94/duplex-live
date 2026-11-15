"""Catalogue program: scale scheduled AdamW rates, preserve clocks/moments.

Collective clipping belongs to the distributed driver, never this program.
The boundary controller renders only the multiplier literal from its catalogue.
"""
LR_MULTIPLIER = 1.0
CLIP_GRAD_NORM = None


class Updater:
    follow_user_scheduler = True

    def __init__(self, named_params, previous=None, spec=None):
        self.named = named_params
        self.opts, self.scheds = spec['user_optimizers'], spec['user_schedulers']
        self.last_effective_rates = list(getattr(previous, 'last_effective_rates', []))
        self.last_update = getattr(previous, 'last_update', spec['update'])

    @property
    def param_groups(self):
        return [group for opt in self.opts for group in opt.param_groups]

    def step(self, grads, update):
        if type(update) is not int or update != self.last_update + 1:
            raise ValueError('Catalogue updater requires the next unsaved update')
        groups = self.param_groups
        nominal = [g['lr'] for g in groups]
        effective = [rate * LR_MULTIPLIER for rate in nominal]
        try:
            for group, rate in zip(groups, effective):group['lr'] = rate
            for optimizer in self.opts:optimizer.step()
        finally:
            for group, rate in zip(groups, nominal):group['lr'] = rate
        self.last_effective_rates, self.last_update = effective, update

    def state_dict(self):
        return dict(opts=[o.state_dict() for o in self.opts],
                    scheds=[s.state_dict() for s in self.scheds],
                    multiplier=LR_MULTIPLIER, last_effective_rates=self.last_effective_rates,
                    last_update=self.last_update)

    def load_state_dict(self, state):
        if state['multiplier'] != LR_MULTIPLIER:
            raise ValueError('Program and saved multiplier differ')
        if len(state['opts']) != len(self.opts) or len(state['scheds']) != len(self.scheds):
            raise ValueError('All Adam and scheduler states are required')
        for opt, value in zip(self.opts, state['opts']):opt.load_state_dict(value)
        for scheduler, value in zip(self.scheds, state['scheds']):scheduler.load_state_dict(value)
        self.last_effective_rates = list(state['last_effective_rates'])
        self.last_update = state['last_update']


def build(named_params, previous=None, spec=None):
    return Updater(named_params, previous, spec)
