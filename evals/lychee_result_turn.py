"""One application-chosen result turn; ordinary conversation uses base weights."""


class ResultTurn:
    def __init__(self):
        self.attempted=False
        self.active=False
        self.started=False
        self.finished=False

    def eligible(self, observed, refreshed, state, quiet_chunks):
        return (observed and not refreshed and not self.attempted
            and state=='l' and quiet_chunks>=2)

    def activate(self):
        if self.attempted:raise RuntimeError('A result turn cannot be activated twice')
        self.attempted=True
        self.active=True

    def observe(self, event):
        if not self.active:return
        if event['type']=='state_change':
            target=event['to'].lower()
            if target=='s':self.started=True
            if target=='l' and self.started:self.finished=True
        if event['type']=='speaking_done' and self.started:self.finished=True

    def should_restore(self, state):
        return self.active and self.finished and state=='l'

    def restored(self):
        self.active=False

