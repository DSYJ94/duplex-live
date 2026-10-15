"""Disclosed application scheduling diagnostic, never a learned timing policy."""


class QuietDelivery:
    def __init__(self, mode):
        if mode not in ('immediate','idle-quiet-diagnostic'):
            raise ValueError('Unknown completed-result delivery mode')
        self.mode=mode
        self.quiet_chunks=0

    def observe(self, quiet, native_state):
        self.quiet_chunks=self.quiet_chunks+1 if quiet else 0
        return self.mode=='immediate' or (self.quiet_chunks>=2 and native_state=='l')
