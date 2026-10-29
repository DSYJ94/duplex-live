"""Opt-in final-utterance TTS budget experiment; native control stays unchanged.

The pinned upstream duplex path caps even terminal speech at26 audio tokens,
then clears its TTS cache. This wrapper permits natural TTS EOS within a finite
larger budget on final chunks only. Native audio qualification remains required.
"""


def install(duplex, maximum_tokens=128):
    if type(maximum_tokens) is not int or not 27<=maximum_tokens<=200:
        raise ValueError('Explicit finite terminal TTS budget27..200 required')
    original=duplex.model.tts.generate_chunk
    observations=[]
    def generate(*args,**kwargs):
        final=bool(duplex.current_turn_ended)
        old=kwargs.get('max_new_token')
        changed=final and old==26 and kwargs.get('min_new_tokens')==0
        if changed:kwargs=dict(kwargs,max_new_token=maximum_tokens)
        tokens,cache=original(*args,**kwargs)
        observations.append(dict(terminal=final,original_maximum=old,
            maximum=kwargs.get('max_new_token'),changed=changed,
            returned_tokens=int(tokens.shape[1]),
            possibly_budget_exhausted=bool(changed and tokens.shape[1]>=maximum_tokens-1),
            native_audio_qualified=False))
        return tokens,cache
    duplex.model.tts.generate_chunk=generate
    return observations
