"""Matched digit/word speech diagnostic with fixed RNG, no controller or training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from evals.moshi_model_download import sha256

TEXTS=[('15','fifteen'),('52','fifty two')]


def render_speech(a,listener):
    import numpy as np
    import soundfile as sf
    import torch
    from evals.duplex_speaker import StreamingSpeaker
    a.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42424242);torch.cuda.manual_seed_all(42424242)
    speaker=StreamingSpeaker(a.assets,a.mimi_load_approval);speaker.open();warmup=speaker.warmup()
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all();rows=[]
    from evals.moshi_training_smoke import tensor_hashes
    def parameter_hashes():return {role:tensor_hashes(model.named_parameters()) for role,model in [('lm',speaker.tts.lm),('mimi',speaker.tts.mimi)]}
    before=parameter_hashes()
    try:
        for seed_offset in (0,1):
            for digit,word in TEXTS:
                for mode,text in [('digits',digit),('words',word)]:
                    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
                    if seed_offset:torch.manual_seed(42424242+seed_offset);torch.cuda.manual_seed_all(42424242+seed_offset)
                    identity=f'{digit}-{mode}-{seed_offset}';speaker.start(text);packets=[];steps=[];start=time.monotonic()
                    while speaker.active:
                        result=speaker.step();steps.append(result['compute_s']);packets.extend(result['pcm'])
                    pcm=np.concatenate(packets);path=a.out/(identity+'.wav');sf.write(path,pcm,24000,subtype='FLOAT')
                    rows.append(dict(id=identity,input_text=text,expected_number=digit,mode=mode,seed_offset=seed_offset,wav=path.name,wav_sha256=sha256(path),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),samples=len(pcm),all_steps=steps,seconds=time.monotonic()-start,synthetic_model_speech=True))
    finally:speaker.close()
    assert before==parameter_hashes()
    # Same pinned recognizer used as the live input listener; diagnostic only.
    from evals.duplex_local_secondary_asr import transcribe
    listener.reset()
    for _ in range(20):listener.step(np.zeros(1920,dtype=np.float32),verify_input=False)
    for row in rows:
        wave,rate=sf.read(a.out/row['wav'],dtype='float32');assert rate==24000
        text,events=transcribe(listener,wave)
        (a.out/(row['id']+'-asr-events.json')).write_text(json.dumps(events,indent=2)+'\n')
        row['local_transcript']=text;row['reference_supplied_to_recognizer']=False
    report=dict(schema='dl-short-number-speech-probe-v1',results=rows,rows=8,component_execution_completed=True,speaker_evidence=speaker.evidence,listener_loads=listener.loads,warmup=warmup,all_speaker_parameter_hashes_unchanged=True,rng_reset_before_each_render=True,actual_external_requests=0,optimizer_steps=0,owner_audio_used=False,owner_demo_changed=False,training_approved=False,do_not_train=True,independent_whisper_required=True,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(rows=8,transcripts=[(r['id'],r['local_transcript']) for r in rows])))


def audit(a):
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words,wer
    report=json.loads((a.run/'report.json').read_text())
    if report['schema']!='dl-short-number-speech-probe-v1' or not report['do_not_train'] or report['owner_audio_used'] or len(report['results'])!=8:raise ValueError('Expected bounded synthetic render source')
    a.out.mkdir(parents=True,exist_ok=False);model=whisper.load_model('turbo',device='cuda');rows=[]
    for row in report['results']:
        path=a.run/row['wav']
        if path.resolve().parent!=a.run.resolve() or sha256(path)!=row['wav_sha256']:raise ValueError('Speech source changed')
        wave,rate=sf.read(path,dtype='float32')
        if rate!=24000 or hashlib.sha256(wave.tobytes()).hexdigest()!=row['pcm_sha256']:raise ValueError('Speech PCM changed')
        result=whisper.transcribe(model,resample_poly(wave,2,3),language='en',vad=False,verbose=False)
        # The expected text is used only after transcription.
        text=result['text'];rows.append(dict(id=row['id'],transcript=text,wer=wer(norm_words(row['expected_number']),norm_words(text)),wav_sha256=row['wav_sha256'],segments=result['segments'],reference_supplied_to_recognizer=False))
    value=dict(schema='dl-short-number-independent-asr-v1',results=rows,source_report_sha256=sha256(a.run/'report.json'),actual_external_requests=0,owner_audio_used=False,training_approved=False,qualification_replacement=False,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(rows=len(rows),transcripts=[(r['id'],r['transcript']) for r in rows])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',action='store_true')
    for name in ('assets','stt-assets','mimi-load-approval','run'):p.add_argument('--'+name,type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.audit:
        if a.run is None:p.error('Saved render required')
        audit(a)
    else:
        if any(getattr(a,n) is None for n in ('assets','stt_assets','mimi_load_approval')):p.error('Pinned speech assets required')
        from evals.kyutai_stt_probe import StreamingListener
        listener=StreamingListener(a.stt_assets,a.mimi_load_approval)
        with listener.session():render_speech(a,listener)


if __name__=='__main__':main()
