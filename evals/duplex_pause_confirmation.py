"""One lexical confirmation per unfinished user utterance, without intent routing.

Energy alone is insufficient to stop ordinary output. This optional gate waits
for an actual alphanumeric ASR fragment accompanied by recent microphone sound.
It neither recognizes commands nor decides whether the model should resume.
"""


class PauseConfirmation:
    def __init__(self):self.seen=False
    def update(self,fragment,recent_sound,utterance_complete=False):
        if not isinstance(fragment,str) or not isinstance(recent_sound,bool) or not isinstance(utterance_complete,bool):raise ValueError('Invalid pause confirmation')
        confirmed=recent_sound and any(c.isalnum() for c in fragment)
        pause=confirmed and not self.seen
        if confirmed:self.seen=True
        if utterance_complete:self.seen=False
        return pause
