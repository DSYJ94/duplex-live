"""Released native token scheduling, shared by CPU data checks and GPU replay."""

def schedule(duplex, timeline):
    scheduled={}
    for action in timeline['actions']:
        if action['kind']=='end_turn':
            chunks=[[duplex.speak_token_id,duplex.turn_eos_token_id,duplex.chunk_eos_token_id]]
        elif action['kind']=='speak' and 'native_speech_units' in action:
            from data.venus.repair import words
            units=action['native_speech_units'];assert units and [u['offset'] for u in units]==list(range(len(units)))
            assert words(' '.join(u['text'] for u in units))==words(action['text'])
            assert [u['final'] for u in units]==[False]*(len(units)-1)+[True]
            chunks=[]
            for unit in units:
                ids=duplex.tokenizer.encode(unit['text'],add_special_tokens=False)
                if len(ids)>16:raise ValueError('Measured speech unit exceeds native decode capacity; repair source pacing, not thresholds')
                chunks.append([duplex.speak_token_id]+ids+([duplex.turn_eos_token_id] if unit['final'] and action['final'] else [])+[duplex.chunk_eos_token_id])
        else:
            text=action['text']
            if action['kind']=='delegate':text='<delegate>'+text+'</delegate>'
            ids=duplex.tokenizer.encode(text,add_special_tokens=False)
            chunks=[]
            for start in range(0,len(ids),16):
                last=start+16>=len(ids)
                chunks.append([duplex.speak_token_id]+ids[start:start+16]+
                    ([duplex.turn_eos_token_id] if last and action['final'] else [])+
                    [duplex.chunk_eos_token_id])
        for offset,ids in enumerate(chunks):
            frame=action['frame']+offset
            if frame in scheduled or frame>=timeline['frames'] or len(ids)>19:
                raise ValueError('Teacher speech/action does not fit original causal timeline')
            scheduled[frame]=ids
    return scheduled

