"""Causal TTS playout: buffer each new utterance and honour actual interruptions."""
import math
from collections import OrderedDict
from evals.audio_playout import render_packets


def render_utterances(packets, rate=24000, buffer_s=.24, stops=None):
    """Preserve PCM except samples cancelled before their scheduled playback."""
    import numpy as np
    if rate<=0 or not 0<=buffer_s<=.5 or not packets:
        raise ValueError('Invalid bounded utterance playout')
    stops=stops or {};groups=OrderedDict();previous=-1.
    for packet in packets:
        identity,available,pcm=packet['event_id'],packet['available_s'],np.asarray(packet['pcm'])
        if not identity or not math.isfinite(available) or available<previous:
            raise ValueError('Invalid utterance identity or PCM arrival order')
        groups.setdefault(identity,[]).append((available,pcm));previous=available
    if any(not math.isfinite(t) or t<0 for t in stops.values()):
        raise ValueError('Invalid actual interruption clock')
    segments=[];events={};source_samples=0;cancelled=0;within=0
    for identity,values in groups.items():
        _,ledger=render_packets(values,rate,buffer_s)
        cap=math.ceil(stops[identity]*rate) if identity in stops else None
        kept=0;prior_end=None;gaps=0
        for row,(_,pcm) in zip(ledger['packets'],values):
            first=row['playback_start_sample'];last=first+len(pcm);source_samples+=len(pcm)
            end=last if cap is None else min(last,cap)
            keep=max(0,end-first)
            if prior_end is not None:
                gaps+=max(0,(first if cap is None else min(first,cap))-prior_end)
            if keep:segments.append((first,pcm[:keep],identity));kept+=keep
            cancelled+=len(pcm)-keep
            prior_end=last if cap is None else min(last,cap)
        within+=gaps
        events[identity]={'startup_buffer_s':buffer_s,'actual_stop_s':stops.get(identity),
            'source_samples':ledger['source_samples'],'played_samples':kept,
            'within_utterance_underrun_samples':gaps,'packets':ledger['packets']}
    segments.sort(key=lambda row:row[0]);cursor=0
    for first,pcm,identity in segments:
        if first<cursor:raise ValueError('Buffered own utterances overlap without a causal stop')
        cursor=first+len(pcm)
    output=np.zeros(cursor,dtype=np.result_type(*(np.asarray(p['pcm']).dtype for p in packets)))
    for first,pcm,_ in segments:output[first:first+len(pcm)]=pcm
    return output,{'schema':'dl-utterance-playout-v1','sample_rate':rate,'startup_buffer_s':buffer_s,
        'source_samples':source_samples,'played_samples':sum(len(pcm) for _,pcm,_ in segments),
        'cancelled_before_playback_samples':cancelled,'within_utterance_underrun_samples':within,
        'events':events,'sample_accounting_exact':source_samples==cancelled+sum(len(pcm) for _,pcm,_ in segments)}
