"""Causal speech-start and conservative utterance-end policy for a hybrid probe.

ASR head scores alone are not complete-turn decisions. Require quiet input,
settled decoded text and terminal punctuation. This baseline is not a learned
micro-turn policy and has not qualified unfinished thoughts or backchannels.
"""


class TurnEndpoint:
    def __init__(self,decode,energy_floor=.003,quiet_frames=3,text_idle_frames=2):
        if energy_floor<=0 or quiet_frames<1 or text_idle_frames<1:
            raise ValueError('Invalid causal endpoint policy')
        self.decode=decode;self.energy_floor=energy_floor;self.required_quiet=quiet_frames
        self.text_idle=text_idle_frames;self.frame=-1;self.tokens=[];self.last_word=-1
        self.quiet=0;self.active_frames=0;self.speech_active=False;self.last_time=-1.

    def update(self,token,score,rms,available_s):
        import math
        if (not math.isfinite(rms) or rms<0 or not math.isfinite(available_s) or available_s<=self.last_time
                or (score is not None and (not math.isfinite(score) or not 0<=score<=1))):
            raise ValueError('Invalid arrived-audio endpoint observation')
        self.frame+=1;self.last_time=available_s
        if rms>self.energy_floor:self.active_frames+=1;self.quiet=0
        else:self.active_frames=0;self.quiet+=1
        started=not self.speech_active and self.active_frames>=2
        if started:self.speech_active=True
        ended=self.speech_active and self.quiet>=self.required_quiet
        if ended:self.speech_active=False
        if token is not None and token>=4:
            self.tokens.append(token);self.last_word=self.frame
        text=self.decode(self.tokens).strip() if self.tokens else ''
        terminal=text.rstrip(' \t\r\n"\'”’').endswith(('.', '?', '!'))
        settled=bool(text and self.frame-self.last_word>=self.text_idle)
        raw_candidate=settled and score is not None and score>.5
        complete=raw_candidate and terminal and self.quiet>=self.required_quiet and not self.speech_active
        utterance=None
        if complete:
            utterance=dict(text=text,input_available_s=available_s,publisher_eot_score=score,
                quiet_frames=self.quiet,text_idle_frames=self.frame-self.last_word)
            self.tokens=[]
        return dict(speech_started=started,speech_ended=ended,speech_active=self.speech_active,
            utterance=utterance,raw_endpoint_candidate=raw_candidate,
            candidate_deferred=raw_candidate and not complete,
            completed_text_cue=terminal,quiet_frames=self.quiet)
