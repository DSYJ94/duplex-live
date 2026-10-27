"""Paired native voice comparison. Never adopts a voice or updates weights."""
import argparse,asyncio,base64,gc,hashlib,json,time
from pathlib import Path
from .behavior import preflight,run_case
from .spoken_audit import REVISION,references
from .lora import disabled
from data.venus.repair import wer
from portable.venus.native_app import NativeEngine,ensure_guard
from portable.venus.voice import reference

CONDITIONS=('base-released-reference','step190-released-reference','step190-eleven-reference')

def recognizer():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq,AutoProcessor,pipeline
    name='openai/whisper-large-v3-turbo'
    model=AutoModelForSpeechSeq2Seq.from_pretrained(name,revision=REVISION,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
    p=AutoProcessor.from_pretrained(name,revision=REVISION,local_files_only=True)
    return pipeline('automatic-speech-recognition',model=model,tokenizer=p.tokenizer,feature_extractor=p.feature_extractor,torch_dtype=torch.float16,device=0,chunk_length_s=30,batch_size=1)

async def main(a):
    import torch,numpy as np,soundfile as sf
    from scipy.signal import resample_poly
    from transformers import set_seed
    stop=json.loads(Path(a.lease).read_text())['stop_unix'];assert stop-time.time()>600
    plan=json.loads(Path(a.execution_plan).read_text());ref=plan['voice_reference'];wav=Path(a.reference)
    assert hashlib.sha256(wav.read_bytes()).hexdigest()==ref['sha256']
    out=Path(a.output);out.mkdir(exist_ok=False);ensure_guard()
    pipe=recognizer();data,sr=sf.read(wav,dtype='float32');assert sr==16000
    recognized=pipe(dict(array=data,sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text']
    source=dict(source_sha256=ref['sha256'],asr_text=recognized,expected_text=ref['authored_text'],wer=wer(ref['authored_text'],recognized),peak=float(np.abs(data).max()),asr_revision=REVISION)
    source['passed']=source['wer']<=.05 and source['peak']<=.995
    (out/'reference-source-audit.json').write_text(json.dumps(source,indent=2)+'\n')
    assert source['passed'],'No native custom-reference run from failed reference audio'
    del pipe;gc.collect();torch.cuda.empty_cache()
    cases=[c for c in preflight(a.plan,a.timelines,a.approval) if c['family']=='ordinary'];assert len(cases)==8 and sum(c['frames'] for c in cases)==81
    assert hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()==a.checkpoint_sha256
    engine=NativeEngine(a.model,adapter_checkpoint=a.checkpoint)
    released=engine.voice_reference;custom=reference(a.model,wav,ref['sha256']);results=[]
    for condition in CONDITIONS:
        root=out/condition;root.mkdir();engine.voice_reference=custom if condition==CONDITIONS[2] else released
        (root/'profile.json').write_text(json.dumps(dict(engine.profile,voice_reference=engine.voice_reference,condition=condition,adapter_enabled=condition!=CONDITIONS[0],input_set='all8ordinary-from-inspected36',qualified=False),indent=2)+'\n')
        async def collect():
            for case in cases:
                set_seed(42);ensure_guard()
                if stop-time.time()<case['frames']*1.2+240:raise TimeoutError('Full comparison/backup reserve required')
                row=await run_case(engine,case,root,stop);results.append(dict(condition=condition,**row))
                print(json.dumps(dict(event='native_voice_case_complete',condition=condition,id=case['id'],completed=len(results),required=24)),flush=True)
        if condition==CONDITIONS[0]:
            with disabled(engine.model.llm):await collect()
        else:await collect()
    del engine;gc.collect();torch.cuda.empty_cache();pipe=recognizer();audit=[]
    for row in results:
        ensure_guard();assert stop-time.time()>180
        folder=out/row['condition']/row['id'];traces=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
        waves=[sf.read(folder/t['chunk_file'],dtype='float32')[0] for t in traces];audio=np.concatenate(waves)
        spoken=pipe(dict(array=resample_poly(audio,2,3).astype(np.float32),sampling_rate=16000),generate_kwargs=dict(language='en',task='transcribe',do_sample=False))['text'] if np.any(audio) else ''
        targets=references(traces);p=folder/'joined-emitted.wav';sf.write(p,audio,24000,subtype='FLOAT')
        audit.append(dict(id=row['id'],condition=row['condition'],asr_text=spoken,references=targets,word_error_rate=wer(targets['intended_user_facing'],spoken),peak=float(np.abs(audio).max()) if len(audio) else 0,joined_wav_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_chunk_hashes=[hashlib.sha256((folder/t['chunk_file']).read_bytes()).hexdigest() for t in traces],qualified=False))
    result=dict(complete=len(results)==len(audit)==24,conditions=list(CONDITIONS),cases_per_condition=8,input_frames_per_condition=81,
        cases=audit,source=a.source,reference_audit=source,asr_revision=REVISION,optimizer_updates=0,seed=42,qualified=False,
        naturalness_requires_listening=True,source_set_already_inspected=True,not_a_fresh_holdout_score=True)
    (out/'voice-comparison.json').write_text(json.dumps(result,indent=2)+'\n');assert result['complete']

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('execution-plan','reference','model','checkpoint','checkpoint-sha256','plan','timelines','approval','output','source','lease')];asyncio.run(main(p.parse_args()))
