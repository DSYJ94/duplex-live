"""Paced streaming recognition on selected approved TRAIN user PCM only.

Model input contains no reference words, assistant audio or future chunk.
Reference words are read after recognition for evaluation. No task or mic claim.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from evals.moshi_model_download import sha256
from evals.kyutai_stt_assets import STT_REVISION
from evals.kyutai_tts_compat import verified_overlay
from evals.kyutai_tts_mimi_load import MIMI_SHA256,strict_mimi_loader


def bootstrap(gen):
    """Advance the initial-only LM slot without consuming actual user audio."""
    state=gen._streaming_state;lm=gen.lm_model
    if state.offset_cpu!=0 or bool((state.offsets!=0).any()) or lm.dep_q!=0 or any(lm.delays):
        raise RuntimeError('ASR bootstrap requires a reset zero-delay text-only generator')
    gen.step(lm._get_initial_token()[:,1:])


def checked_step(gen,codes):
    """Require every supplied user code to reach the actual temporal forward."""
    import torch
    state=gen._streaming_state;original=state.graphed_main;calls=0
    def forward(value,*args):
        nonlocal calls
        if not torch.equal(value[:,1:],codes):
            raise RuntimeError('Actual user frame was replaced before the ASR temporal forward')
        calls+=1;return original(value,*args)
    state.graphed_main=forward
    try:result=gen.step_with_extra_heads(codes)
    finally:state.graphed_main=original
    if calls!=1:raise RuntimeError('Actual ASR frame did not execute exactly one temporal forward')
    return result


class StreamingListener:
    """One reusable frozen ASR runtime; returns raw causal tokens and head scores."""
    def __init__(self,assets_path,mimi_approval):
        assets=json.loads(assets_path.read_text());gate=json.loads(mimi_approval.read_text())
        if (assets['schema']!='dl-kyutai-stt-assets-v1' or not assets['passed']
                or assets['revision']!=STT_REVISION
                or assets['code_sha256']!=sha256(Path(__file__).with_name('kyutai_stt_assets.py'))
                or gate['schema']!='dl-kyutai-tts-mimi-load-cpu-v1' or not gate['passed']
                or gate['code_sha256']!=sha256(Path(__file__).with_name('kyutai_tts_mimi_load.py'))
                or len(gate['records'])!=1 or not gate['records'][0]['all_tensors_exact']
                or gate['records'][0]['checkpoint_sha256']!=MIMI_SHA256):
            raise RuntimeError('Streaming ASR lacks matching assets and exact shared-codec load proof')
        compat_folder=Path(assets['compat_folder']);compat,overlay=verified_overlay(compat_folder)
        if sha256(compat_folder/'report.json')!=assets['compat_report_sha256']:
            raise RuntimeError('Changed ASR package compatibility source')
        rows=assets['assets'];roles={'config','model','mimi','tokenizer','model_card'}
        if len(rows)!=len(roles) or {r['role'] for r in rows}!=roles:raise RuntimeError('Incomplete ASR asset inventory')
        paths={}
        for row in rows:
            p=Path(row['path'])
            if row['revision']!=STT_REVISION or p.stat().st_size!=row['bytes'] or sha256(p)!=row['sha256']:
                raise RuntimeError('Changed streaming ASR asset')
            paths[row['role']]=p
        if next(r['sha256'] for r in rows if r['role']=='mimi')!=MIMI_SHA256:
            raise RuntimeError('ASR did not use the independently verified identical Mimi')
        sys.path.insert(0,str(overlay))
        import torch
        from moshi.models import loaders,LMGen
        from moshi.utils.compile import no_compile
        from evals.context_spanning_probe import audited_loads
        from evals.context_execution import math_attention,snapshot
        self.torch=torch;self.no_compile=no_compile;self.loads=[];self.mimi_loads=[]
        math_attention();start=time.perf_counter()
        with audited_loads(self.loads),strict_mimi_loader(paths['mimi'],self.mimi_loads):
            info=loaders.CheckpointInfo.from_hf_repo('kyutai/stt-1b-en_fr-candle',
                moshi_weights=paths['model'],mimi_weights=paths['mimi'],
                tokenizer=paths['tokenizer'],config_path=paths['config'])
            self.mimi=info.get_mimi(device='cuda');self.spm=info.get_text_tokenizer()
            self.lm=info.get_moshi(device='cuda',dtype=torch.bfloat16)
        for model in (self.mimi,self.lm):
            checked=[r for r in self.loads if r['module_identity']==id(model)]
            if not checked or any(not r['strict'] or r['missing'] or r['unexpected'] for r in checked):
                raise RuntimeError('Streaming ASR did not load completely and strictly')
            model.eval().requires_grad_(False)
        self.gen=LMGen(self.lm,temp=0.,temp_text=0.)
        self.load_seconds=time.perf_counter()-start;self.execution=snapshot(self.lm)
        self.assets_sha256=sha256(assets_path);self.mimi_approval_sha256=sha256(mimi_approval)

    @contextmanager
    def session(self):
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with self.torch.inference_mode(),self.no_compile(),sdpa_kernel(SDPBackend.MATH),self.mimi.streaming(1),self.gen.streaming(1):
            yield self

    def reset(self):
        self.mimi.reset_streaming();self.gen.reset_streaming()
        bootstrap(self.gen)

    def step(self,pcm,verify_input=True):
        import numpy as np
        t=self.torch
        if pcm.shape!=(1920,) or pcm.dtype!=np.float32 or not np.isfinite(pcm).all():
            raise RuntimeError('ASR accepts one finite arrived 24kHz mono 80ms frame')
        codes=self.mimi.encode(t.from_numpy(pcm.copy()).view(1,1,-1).to('cuda'))
        if codes.shape!=(1,32,1):raise RuntimeError('Streaming codec changed its one-frame input clock')
        result=checked_step(self.gen,codes) if verify_input else self.gen.step_with_extra_heads(codes)
        if result is None:return dict(token=None,heads=None,codes=codes.detach().cpu())
        tokens,heads=result
        if len(heads)!=4 or any(v.shape!=(1,1,6) for v in heads):raise RuntimeError('Unexpected ASR end-of-turn head layout')
        values=t.stack([v[0,0].float() for v in heads]).cpu()
        if not t.isfinite(values).all() or (values<0).any() or (values>1).any():
            raise RuntimeError('Invalid streaming ASR head probabilities')
        token=int(tokens[0,0,0].item())
        return dict(token=token,heads=values.tolist(),codes=codes.detach().cpu())


def selected_inputs(data,ids):
    from data.synthetic.quality import checked_rows
    receipt=json.loads((data/'ingestion.json').read_text());batch=Path(receipt['source_batch'])
    if sha256(batch/'quality.jsonl')!=receipt['source_quality_sha256']:
        raise RuntimeError('Changed original input quality approval')
    if len(set(ids))!=len(ids):raise RuntimeError('Duplicate ASR probe input')
    rows=[]
    for identity in ids:
        row=next(r for r in receipt['accepted'] if r['scenario_id']==identity)
        if row['split']!='train':raise RuntimeError('This bounded ASR probe reads TRAIN inputs only')
        checked_rows(batch,[row],role='streaming ASR TRAIN component input')
        for path,key in [(Path(row['path']),'wav_sha256'),(Path(row['path']).with_suffix('.json'),'sidecar_sha256')]:
            if sha256(path)!=row[key]:raise RuntimeError('Changed approved ASR probe source')
        rows.append(row)
    return rows


def run_case(listener,row,out):
    import numpy as np
    import soundfile as sf
    from data.synthetic.filter import norm_words,wer
    pcm,rate=sf.read(row['path'],dtype='float32',always_2d=True)
    if rate!=24000 or pcm.shape[1]!=2 or not np.isfinite(pcm).all():raise RuntimeError('Invalid approved stereo source')
    user=pcm[:,1].copy();out.mkdir();sf.write(out/'input-user.wav',user,rate,subtype='FLOAT')
    warm=time.perf_counter()
    for _ in range(20):listener.step(np.zeros(1920,dtype=np.float32),verify_input=False)
    listener.reset();warm_seconds=time.perf_counter()-warm
    if listener.gen._streaming_state.graphed_main._graph is None:raise RuntimeError('Streaming ASR CUDA graph was not captured')
    real_frames=math.ceil(len(user)/1920);flush_frames=math.ceil(.5*12.5)
    padded=np.pad(user,(0,(real_frames+flush_frames)*1920-len(user)))
    events=[];codes=[];tokens=[];all_words=[];current=[];endpoints=[];last_word=-1;steps=[];backlog=[]
    start=time.perf_counter()
    for i in range(real_frames+flush_frames):
        available=(i+1)*.08;wait=start+available-time.perf_counter()
        if wait>0:time.sleep(wait)
        backlog.append(max(0.,time.perf_counter()-start-available));before=time.perf_counter()
        result=listener.step(padded[i*1920:(i+1)*1920]);steps.append(time.perf_counter()-before)
        token=result['token'];heads=result['heads'];codes.append(result['codes']);tokens.append(-2 if token is None else token)
        if token is not None and token>=4:
            all_words.append(token);current.append(token);last_word=i
        score=heads[2][0] if heads is not None else None
        if current and token in (0,3) and i-last_word>=2 and score is not None and score>.5:
            endpoints.append(dict(text=listener.spm.decode(current),input_available_s=available,
                available_wall_s=time.perf_counter()-start,publisher_eot_score=score));current=[]
        events.append(dict(frame=i,input_available_s=available,available_wall_s=time.perf_counter()-start,
            real_samples_seen=min((i+1)*1920,len(user)),token=token,heads=heads,
            compute_s=steps[-1],arrival_backlog_s=backlog[-1],synthetic_flush=i>=real_frames))
    wall=time.perf_counter()-start
    recognized=listener.spm.decode(all_words)
    # Reference content is first parsed after the complete causal recognition run.
    side=json.loads(Path(row['path']).with_suffix('.json').read_text())
    reference=' '.join(w['text'] for w in side['user_words'])
    expected=[dict(start_s=e['actual_start_ms']/1000,end_s=e['actual_end_ms']/1000,text=e.get('text',''))
        for e in side['timeline']['events'] if e['ch']=='user' and e['kind']!='marker']
    file=out/'stream.pt';listener.torch.save(dict(encoded_user_codes=listener.torch.cat(codes,dim=2),
        predicted_tokens=listener.torch.tensor(tokens)),file)
    (out/'events.json').write_text(json.dumps(events,indent=2)+'\n')
    record=dict(scenario=row['scenario_id'],source_split='train',source_wav_sha256=row['wav_sha256'],
        user_pcm_sha256=hashlib.sha256(user.tobytes()).hexdigest(),actual_user_samples=len(user),
        real_user_samples_processed=events[-1]['real_samples_seen'],all_user_audio_processed=True,
        input_seconds=len(user)/rate,wall_seconds=wall,warmup_seconds=warm_seconds,
        padded_samples=real_frames*1920-len(user),synthetic_flush_frames=flush_frames,
        compute_p95_s=float(np.percentile(steps,95)),maximum_input_backlog_s=max(backlog),
        recognized_text=recognized,reference_user_text=reference,word_error_rate=wer(norm_words(reference),norm_words(recognized)),
        end_of_turn_events=endpoints,unended_text=listener.spm.decode(current),reference_user_events=expected,
        stream_tensor_sha256=sha256(file),reference_transcript_supplied=False,reference_read_after_generation=True,
        assistant_audio_supplied=False,causal_input_frame_end_clock=True,timestamps_are_availability_not_inferred_word_times=True,
        initial_lm_bootstrap_virtual_frames=1,bootstrap_actual_audio_samples=0,all_live_user_rows_exact_at_temporal_forward=True,
        microphone_acceptance=False,duplex_system_qualified=False,training_approved=False,optimizer_steps=0)
    (out/'report.json').write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','mimi-load-approval','data','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--scenario',action='append',required=True);args=ap.parse_args()
    rows=selected_inputs(args.data,args.scenario);args.out.mkdir(parents=True,exist_ok=False)
    listener=StreamingListener(args.assets,args.mimi_load_approval);results=[]
    with listener.session():
        for row in rows:results.append(run_case(listener,row,args.out/row['scenario_id']))
    report=dict(schema='dl-kyutai-streaming-asr-probe-v1',passed=True,results=results,
        model_load_seconds=listener.load_seconds,loads=listener.loads,strict_mimi_loads=listener.mimi_loads,
        execution=listener.execution,assets_sha256=listener.assets_sha256,mimi_approval_sha256=listener.mimi_approval_sha256,
        cuda_graph_main_captured_in_every_case=True,torch_compile_enabled=False,
        endpoint_rule='publisher head2/class0 >0.5 after two nonlexical frames; raw scores retained',
        mimi_cpu_proof_reused_for_identical_codec=True,reference_transcript_supplied=False,
        training_approved=False,optimizer_steps=0,microphone_acceptance=False,duplex_system_qualified=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
