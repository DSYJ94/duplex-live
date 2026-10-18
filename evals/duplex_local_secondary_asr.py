"""Local-only second recognizer for synthetic audio audit discrepancies.

No network transcription or API calls. Original Whisper gates remain unchanged.
Kyutai STT is also the prototype listener, so this is a second recognizer relative
to Whisper, not a fully independent end-to-end acceptance system.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.duplex_secondary_asr import inventory


def transcribe(listener,pcm):
    import numpy as np
    if pcm.ndim!=1 or not np.isfinite(pcm).all():raise ValueError('Invalid diagnostic mono audio')
    listener.reset();frames=math.ceil(len(pcm)/1920);flush=20;padded=np.pad(pcm,(0,(frames+flush)*1920-len(pcm)));tokens=[];events=[]
    for i in range(frames+flush):
        value=listener.step(padded[i*1920:(i+1)*1920])
        if value['token'] is not None and value['token']>=4:tokens.append(value['token'])
        events.append(dict(frame=i,token=value['token'],synthetic_flush=i>=frames))
    return listener.spm.decode(tokens),events


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('runs','out','assets','mimi-load-approval'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--inventory',type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import soundfile as sf
    from data.synthetic.filter import norm_words,wer
    from evals.kyutai_stt_probe import StreamingListener
    if a.inventory is None:clips=inventory(a.runs)
    else:
        source=json.loads(a.inventory.read_text())
        if source['training_approved'] or not source['do_not_train']:raise ValueError('Secondary inventory is evaluation-only')
        clips=[]
        for row in source['clips']:
            path=a.runs/row['path']
            if not path.resolve().is_relative_to(a.runs.resolve()) or sha256(path)!=row['wav_sha256']:raise ValueError('Secondary audio source changed')
            clips.append(dict(row,path=path))
    listener=StreamingListener(a.assets,a.mimi_load_approval);results=[]
    with listener.session():
        listener.reset()
        for _ in range(20):listener.step(np.zeros(1920,dtype=np.float32),verify_input=False)
        for clip in clips:
            path=clip['path'];pcm,rate=sf.read(path,dtype='float32');assert rate==24000
            before=time.monotonic();text,events=transcribe(listener,pcm)
            row={k:v for k,v in clip.items() if k not in ('path','expected')}
            row.update(path=str(path),wav_sha256=sha256(path),input_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),input_samples=len(pcm),transcript=text,seconds=time.monotonic()-before,expected=clip['expected'],diagnostic_wer=wer(norm_words(clip['expected']),norm_words(text)),reference_supplied_to_recognizer=False,actual_frames_reached_temporal_forward=True)
            (a.out/(clip['id']+'-events.json')).write_text(json.dumps(events,indent=2)+'\n');results.append(row)
    report=dict(schema='dl-local-secondary-asr-diagnostic-v1',model='kyutai/stt-1b-en_fr-candle',results=results,rows=len(results),component_execution_completed=len(results)==len(clips),actual_external_requests=0,local_only=True,also_the_prototype_input_listener=True,all_predeclared_clips_retained=True,primary_gates_unchanged=True,training_approved=False,qualification_replacement=False,assets_sha256=listener.assets_sha256,strict_loads=listener.loads,strict_mimi_loads=listener.mimi_loads,code_sha256=sha256(Path(__file__)),inventory_code_sha256=sha256(Path(__file__).with_name('duplex_secondary_asr.py')),explicit_inventory_sha256=sha256(a.inventory) if a.inventory else None)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(rows=len(results),actual_external_requests=0)),flush=True)

if __name__=='__main__':main()
