"""Exact unheard PCM at a causal pause; no word alignment or new synthesis."""
import math


def unheard_packets(packets,event_id,at_s,rate=24000,buffer_s=.16):
    import numpy as np
    if not math.isfinite(at_s) or at_s<0 or rate<=0 or not 0<=buffer_s<=.5:
        raise ValueError('Invalid causal speech pause')
    cursor=0;first=True;previous=-1.;tail=[];cut=math.ceil(at_s*rate)
    for packet in packets:
        if packet['event_id']!=event_id:continue
        available=packet['available_s'];pcm=np.asarray(packet['pcm'])
        if (not math.isfinite(available) or available<previous or available<0
                or pcm.ndim!=1 or not len(pcm) or not np.isfinite(pcm).all()):
            raise ValueError('Invalid generated pause packet')
        start=max(cursor,math.ceil(available*rate)+(math.ceil(buffer_s*rate) if first else 0))
        heard=max(0,min(len(pcm),cut-start))
        if heard<len(pcm):tail.append(pcm[heard:].copy())
        cursor=start+len(pcm);first=False;previous=available
    if first:raise ValueError('Cannot pause an unknown speech event')
    return tail


def played_positions(group,rate=24000):
    cap=math.ceil(group['actual_stop_s']*rate) if group['actual_stop_s'] is not None else None
    positions=[]
    for packet in group['packets']:
        first=packet['playback_start_sample'];last=first+packet['samples']
        if cap is not None:last=min(last,cap)
        if last>first:positions.append((first,last))
    return positions


def grouped_completions(case,playout,starts,rate=24000):
    """Full same-generation spans, including their real pause, never other speech."""
    result=[];claimed=set()
    for completed in case['fully_played_responses']:
        identities=completed.get('playback_event_ids')
        if identities is None:continue
        if (not isinstance(identities,list) or len(identities)<2 or any(not isinstance(i,str) for i in identities)
                or len(set(identities))!=len(identities) or set(identities)&claimed
                or identities[-1]!=completed['event_id']):raise ValueError('Invalid completed pause identity chain')
        for index,identity in enumerate(identities):
            start=starts[identity]
            if start['task_id']!=completed['task_id'] or start['text']!=completed['text']:
                raise ValueError('Pause chain changes its actual response')
            if index and (start.get('same_generator') is not True or start.get('resumes_event_id')!=identities[index-1]):
                raise ValueError('Pause completion lacks a retained generation')
        positions=[p for i in identities if i in playout['events'] for p in played_positions(playout['events'][i],rate)]
        if not positions:raise ValueError('Paused completion has no actual heard PCM')
        first=min(p[0] for p in positions);last=max(p[1] for p in positions)
        for identity,group in playout['events'].items():
            if identity in identities:continue
            if any(a<last and b>first for a,b in played_positions(group,rate)):
                raise ValueError('Another answer is inside a claimed same-generation completion')
        claimed.update(identities)
        result.append(dict(event_id=identities[-1]+'-complete',task_id=completed['task_id'],first=first,last=last,
            expected=completed['text'],fully=True,playback_event_ids=identities,pause_aggregate=True,intentional_pause_included=True))
    return result
