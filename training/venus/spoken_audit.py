"""Independent complete36 spoken fidelity audit, after native generation exits.

No teacher answers or transcript hints reach the recognizer. Both the literal
generated text and intended user-facing text are retained, to expose tool-call
words entering speech. Raw native WAVs and peaks remain unmodified.
"""
import argparse,hashlib,json,re,time
from pathlib import Path
from data.venus.repair import wer

REVISION='41f01f3fe87f28c78e2fbf8b568835947dd65ed9'


def references(traces):
    text=''.join(t.get('raw_text',t['text']) for t in traces)
    text=text.replace('<|turn_eos|>',' ')
    text=re.sub(r'<(?!/?delegate>)[^>]*>','',text)
    user=re.sub(r'<delegate>.*?</delegate>','',text,flags=re.S)
    unclosed='<delegate>' in user
    if unclosed:user=user.split('<delegate>',1)[0]
    clean=lambda s:re.sub(r'<[^>]*>','',s).strip()
    return dict(literal_generated=clean(text),intended_user_facing=clean(user),unclosed_tool_call=unclosed,
        complete_tool_calls=len(re.findall(r'<delegate>.*?</delegate>',text,flags=re.S)))


def main(a):
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
    from portable.venus.native_app import ensure_guard
    ensure_guard();root=Path(a.collection);terminal=json.loads((root/'collection-terminal.json').read_text())
    results=json.loads((root/'results.json').read_text());cases=results['cases']
    assert terminal['complete'] and terminal['completed_cases']==36 and len(cases)==36 and not terminal['qualified']
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    stop=json.loads(Path(a.lease).read_text())['stop_unix'];assert time.time()<stop-240
    out=Path(a.output);out.mkdir(exist_ok=False);rows=[]
    name='openai/whisper-large-v3-turbo'
    model=AutoModelForSpeechSeq2Seq.from_pretrained(name,revision=REVISION,local_files_only=True,torch_dtype=torch.float16).to('cuda')
    processor=AutoProcessor.from_pretrained(name,revision=REVISION,local_files_only=True)
    recognize=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,torch_dtype=torch.float16,device=0,chunk_length_s=30,batch_size=1)
    for case in cases:
        ensure_guard()
        if time.time()>=stop-210:raise TimeoutError('Keep full backup/STOP reserve; no passing subset')
        folder=root/case['id'];traces=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()];ref=references(traces)
        waves=[];sources=[]
        for row in traces:
            p=folder/row['chunk_file'];data,sr=sf.read(p,dtype='float32');assert sr==24000 and data.ndim==1 and np.isfinite(data).all()
            waves.append(data);sources.append(dict(file=row['chunk_file'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        audio=np.concatenate(waves);peak=float(np.max(np.abs(audio))) if len(audio) else 0
        original=out/(case['id']+'-emitted.wav');sf.write(original,audio,24000,subtype='FLOAT')
        if not len(audio) or not np.any(audio):transcript='';zero_audio=True
        else:
            converted=resample_poly(audio,2,3).astype(np.float32)
            transcript=recognize(dict(array=converted,sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'];zero_audio=False
        row=dict(id=case['id'],family=case['family'],asr_text=transcript,references=ref,
            literal_text_wer=wer(ref['literal_generated'],transcript),user_facing_text_wer=wer(ref['intended_user_facing'],transcript),
            raw_peak=peak,raw_peak_pass=peak<=.995,exact_zero_audio=zero_audio,
            source_chunks=sources,emitted_wav_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            source_collection=hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),qualified=False)
        rows.append(row);(out/(case['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(dict(event='independent_spoken_case',completed=len(rows),required=36,id=case['id'],literal_wer=row['literal_text_wer'],user_facing_wer=row['user_facing_text_wer'],peak=peak)),flush=True)
    result=dict(complete=len(rows)==36,required_cases=36,cases=rows,asr_model=name,asr_revision=REVISION,producer_source=a.source,
        word_limit=.05,peak_limit=.995,all_literal_word_checks=all(r['literal_text_wer']<=.05 for r in rows),
        all_user_facing_word_checks=all(r['user_facing_text_wer']<=.05 for r in rows),all_peak_checks=all(r['raw_peak_pass'] for r in rows),
        spoken_fidelity_is_not_behavioral_correctness=True,independent_semantic_review_required=True,qualified=False)
    (out/'spoken-audit.json').write_text(json.dumps(result,indent=2)+'\n');assert len(rows)==36


if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('collection','output','source','lease')];main(p.parse_args())
