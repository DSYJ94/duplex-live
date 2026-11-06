"""Keep explicit native delegate actions out of speech conditioning.

Only the existing special action delimiters control this channel. Ordinary text,
listen/speak policy, model history, worker requests and cue words are untouched.
Experimental until paired native audio/behavior checks pass.
"""

class ActionChannel:
    def __init__(self,open_id,close_id,special_ids):
        self.open_id,self.close_id=open_id,close_id
        self.special_ids=set(special_ids);self.reset()

    def reset(self):
        self.inside=False;self.hidden_tokens=0;self.action_units=0

    def visible(self,results):
        kept=[];action=self.inside
        for row in results:
            token=int(row[0])
            if token==self.open_id:self.inside=True;action=True;self.hidden_tokens+=1
            elif token==self.close_id:self.inside=False;action=True;self.hidden_tokens+=1
            elif self.inside:self.hidden_tokens+=1
            else:kept.append(row)
        if action:self.action_units+=1
        has_words=any(int(row[0]) not in self.special_ids for row in kept)
        return kept if action else results,action and not has_words

def install(duplex):
    import torch
    if getattr(duplex,'_action_channel',None) is not None:
        raise ValueError('Action channel already installed')
    tokenizer=duplex.tokenizer
    open_id=tokenizer.convert_tokens_to_ids('<delegate>');close_id=tokenizer.convert_tokens_to_ids('</delegate>')
    assert open_id!=close_id and tokenizer.decode([open_id],skip_special_tokens=False)=='<delegate>' and tokenizer.decode([close_id],skip_special_tokens=False)=='</delegate>'
    # Native turn_eos is an added special token, but the pinned tokenizer's
    # all_special_ids omits it. It must not make a delegate-only unit appear
    # to contain words and seed acoustic sampling from an end-of-turn marker.
    control_ids=set(tokenizer.all_special_ids)|set(duplex.turn_terminator_token_ids)
    channel=ActionChannel(open_id,close_id,control_ids)
    convert=duplex._convert_results_to_tts_input;generate=duplex.model.tts.generate_chunk
    def speech_condition(results):
        visible,action_only=channel.visible(results)
        if action_only:
            # Zero new conditioning tokens: no phantom speech and no TTS cache
            # advancement. Native end-of-turn handling still flushes any prior
            # genuine spoken tail. The LLM/action history remains exact.
            return convert([])[:,:0]
        return convert(visible)
    def speech_generate(*args,**kwargs):
        embeddings=kwargs.get('inputs_embeds')
        if embeddings is not None and embeddings.shape[1]==0:
            return torch.empty((embeddings.shape[0],0),dtype=torch.long,device=embeddings.device),kwargs.get('past_key_values')
        return generate(*args,**kwargs)
    duplex._convert_results_to_tts_input=speech_condition
    duplex.model.tts.generate_chunk=speech_generate
    # Completion owns a pinned sampler captured before this wrapper. Publish
    # the channel so that its direct sampler path can honor empty conditioning
    # too. Install completion first; reverse order fails its source check.
    duplex._action_channel=channel
    return channel
