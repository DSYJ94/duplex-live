"""Private-recording policy diagnosis: export only counts, times and hashes.

Replays saved causal ASR outputs, never reruns a model or exports user words/PCM.
Counterfactual saved-input replay is not a new live acoustic-loop acceptance test.
"""
import argparse,json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.duplex_endpoint import TurnEndpoint
from evals.duplex_pause_confirmation import PauseConfirmation


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,action='append',required=True);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    import numpy as np,soundfile as sf,sentencepiece
    assets=json.loads(a.assets.read_text());tokenizer=next(x for x in assets['assets'] if x['role']=='tokenizer');p=Path(tokenizer['path'])
    if sha256(p)!=tokenizer['sha256']:raise ValueError('Changed pinned tokenizer')
    spm=sentencepiece.SentencePieceProcessor(str(p));rows=[]
    for p in a.source:
        pcm,rate=sf.read(p/'input-user.wav',dtype='float32');asr=json.loads((p/'asr-events.json').read_text());events=json.loads((p/'events.json').read_text())
        if pcm.ndim!=1 or rate!=24000:raise ValueError('Unexpected actual capture format')
        endpoint=TurnEndpoint(spm.decode);gate=PauseConfirmation();last_voice=-float('inf');ends=[];confirmations=[]
        for i,e in enumerate(asr):
            frame=pcm[i*1920:(i+1)*1920];frame=np.pad(frame,(0,1920-len(frame)));rms=float(np.sqrt(np.mean(frame**2)));t=e['input_available_s']
            if rms>endpoint.energy_floor:last_voice=t
            score=e['heads'][2][0] if e['heads'] is not None else None;r=endpoint.update(e['token'],score,rms,t)
            fragment=spm.decode([e['token']]) if e['token'] is not None and e['token']>=4 else ''
            if gate.update(fragment,t-last_voice<=1.,r['utterance'] is not None):confirmations.append(e['available_wall_s'])
            if r['utterance']:ends.append(r['utterance']['text'])
        actual_ends=[e['text'] for e in events if e['kind']=='user_utterance_complete']
        if ends!=actual_ends:raise ValueError('Original endpoint replay differs; candidate comparison invalid')
        windows=[];starts={}
        for e in events:
            if e['kind']=='speech_started' and not e.get('same_generator'):starts[e['task_id']]=e['at_s']
            if e['kind']=='speech_fully_played':windows.append((starts.pop(e['task_id']),e['at_s']))
        windows.extend((v,asr[-1]['available_wall_s']) for v in starts.values())
        new_pauses=[t for t in confirmations if any(start<t<end for start,end in windows)]
        pauses=[e['at_s'] for e in events if e['kind']=='speech_paused'];packets=json.loads((p/'packets.json').read_text());groups={}
        for e in packets:
            if e.get('wire_transmitted',True):groups.setdefault(e['event_id'],[]).append(e['available_s'])
        gaps=[b-a for ts in groups.values() for a,b in zip(ts,ts[1:])]
        receipts=json.loads((p/'capture-arrivals.json').read_text());play=[e['client_playback_receipt'] for e in receipts if 'client_playback_receipt' in e]
        rows.append(dict(source_session=p.name,original_endpoint_replay_exact=True,completed_user_utterances=len(ends),original_acoustic_pauses=len(pauses),lexically_confirmed_pauses_during_existing_output=len(new_pauses),confirmed_pause_times_s=new_pauses,server_max_within_event_packet_gap_s=max(gaps,default=None),browser_max_underruns=max((x.get('underruns',0) for x in play),default=0),sources={f:sha256(p/f) for f in ('input-user.wav','asr-events.json','events.json','packets.json','capture-arrivals.json')}))
    report=dict(schema='dl-private-live-pause-numeric-audit-v1',rows=rows,recordings_exported=False,words_exported=False,model_rerun=False,optimizer_steps=0,do_not_train=True,counterfactual_saved_input_only=True,live_acoustic_loop_acceptance=False,code_sha256=sha256(Path(__file__)))
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
