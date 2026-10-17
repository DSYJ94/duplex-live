"""One actual backend receipt spoken by pinned streaming TTS on one GPU.

Typed input, externally chosen tools/words, complete response text before speech.
This is a component compatibility proof, not native duplex or microphone acceptance.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from evals.moshi_model_download import sha256
from evals.kyutai_tts_compat import MODEL_REVISION,VOICE_REVISION,WHEEL_SHA256,verified_overlay
from evals.kyutai_tts_assets import DSM_COMMIT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--mimi-load-approval',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();manifest=json.loads(args.assets.read_text())
    mimi_gate=json.loads(args.mimi_load_approval.read_text())
    if (mimi_gate['schema']!='dl-kyutai-tts-mimi-load-cpu-v1' or not mimi_gate['passed']
            or not mimi_gate['original_default_loader_failed'] or mimi_gate['gpu_used'] is not False
            or mimi_gate['assets_manifest_sha256']!=sha256(args.assets)
            or mimi_gate['code_sha256']!=sha256(Path(__file__).with_name('kyutai_tts_mimi_load.py'))
            or len(mimi_gate['negative_controls_rejected'])!=3):
        raise RuntimeError('Missing matching independent strict Mimi load proof')
    if (not manifest['passed'] or manifest['schema']!='dl-kyutai-tts-assets-v1'
            or manifest['model_revision']!=MODEL_REVISION or manifest['voice_revision']!=VOICE_REVISION
            or manifest['wheel_sha256']!=WHEEL_SHA256
            or manifest['code_sha256']!=sha256(Path(__file__).with_name('kyutai_tts_assets.py'))):
        raise RuntimeError('Missing matching pinned speech assets')
    compat_folder=Path(manifest['compat_folder']);compat,overlay=verified_overlay(compat_folder)
    if sha256(compat_folder/'report.json')!=manifest['compat_report_sha256']:
        raise RuntimeError('Changed CPU import report')
    paths={}
    roles={'config','model','mimi','tokenizer','model_card','voice','voice_card'}
    if len(manifest['assets'])!=len(roles) or {r['role'] for r in manifest['assets']}!=roles:
        raise RuntimeError('Speech asset inventory is incomplete or duplicated')
    for row in manifest['assets']:
        voice_asset=row['role'] in ('voice','voice_card')
        if row['revision']!=(VOICE_REVISION if voice_asset else MODEL_REVISION):
            raise RuntimeError('Speech asset belongs to a different revision')
        path=Path(row['path'])
        if path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:
            raise RuntimeError('Changed speech asset: '+row['role'])
        paths[row['role']]=path
    source=Path(manifest['streaming_source']);example=source/'scripts/tts_pytorch_streaming.py'
    if (manifest['streaming_source_commit']!=DSM_COMMIT
            or subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=DSM_COMMIT
            or subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
            or sha256(example)!=manifest['streaming_example_sha256']):
        raise RuntimeError('Changed upstream streaming example')
    sys.path.insert(0,str(overlay))
    import numpy as np
    import soundfile as sf
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import TTSModel
    from moshi.utils.compile import no_compile,no_cuda_graph
    from evals.context_spanning_probe import audited_loads
    from evals.context_execution import math_attention,snapshot
    from evals.recovery_worker import RecordWorld,run_worker
    from evals.audio_playout import render_packets
    from evals.kyutai_tts_mimi_load import strict_mimi_loader
    spec=importlib.util.spec_from_file_location('pinned_kyutai_streaming_example',example)
    upstream=importlib.util.module_from_spec(spec);sys.modules[spec.name]=upstream;spec.loader.exec_module(upstream)
    args.out.mkdir(parents=True,exist_ok=False)
    math_attention();torch.manual_seed(42424242);torch.cuda.manual_seed_all(42424242)
    loads=[];mimi_loads=[];load_start=time.perf_counter()
    with torch.inference_mode(),no_compile(),no_cuda_graph(),sdpa_kernel(SDPBackend.MATH):
        with audited_loads(loads),strict_mimi_loader(paths['mimi'],mimi_loads):
            info=CheckpointInfo.from_hf_repo('kyutai/tts-1.6b-en_fr',
                moshi_weights=paths['model'],mimi_weights=paths['mimi'],
                tokenizer=paths['tokenizer'],config_path=paths['config'])
            tts=TTSModel.from_checkpoint_info(info,n_q=32,temp=.6,device='cuda')
        for model in (tts.lm,tts.mimi):
            checked=[r for r in loads if r['module_identity']==id(model)]
            if not checked or any(not r['strict'] or r['missing'] or r['unexpected'] for r in checked):
                raise RuntimeError('Speech model did not load completely and strictly')
            model.eval().requires_grad_(False)
        torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
        attributes=tts.make_condition_attributes([paths['voice']],cfg_coef=2.)
        world=RecordWorld({'Willow North':{'room marker':'teal hexagon'}})
        context=[dict(role='user',text='Please check the room marker for Project Willow North.')]
        task='tts-component-lookup';revision=world.reserve(task,context)
        start=time.perf_counter();backend=run_worker(context,world,task,expected_revision=revision)
        backend_ready=time.perf_counter()-start
        receipts=[r for r in backend['receipts'] if r['tool']=='lookup_record' and r.get('executed')]
        if not receipts or any(not world.deliverable(r) for r in receipts):
            raise RuntimeError('Speech trial has no actual deliverable backend lookup')
        text=backend['content'];(args.out/'backend.json').write_text(json.dumps(dict(context=context,result=backend),indent=2)+'\n')
        packets=[];frame_steps=[]
        def on_frame(frame):
            if not bool((frame!=-1).all()):return
            pcm=tts.mimi.decode(frame[:,1:,:]).detach().cpu().numpy()[0,0].copy()
            if pcm.shape!=(1920,) or not np.isfinite(pcm).all():
                raise RuntimeError('Invalid actual streaming TTS PCM')
            packets.append((time.perf_counter()-start,pcm))
        gen=upstream.TTSGen(tts,[attributes],on_frame=on_frame)
        original=gen._step
        def measured_step():
            t=time.perf_counter();original();torch.cuda.synchronize();frame_steps.append(time.perf_counter()-t)
        gen._step=measured_step
        with tts.mimi.streaming(1):
            for entry in upstream.prepare_script(tts,text,first_turn=True):
                gen.append_entry(entry);gen.process()
            gen.process_last()
        wall=time.perf_counter()-start
    if not packets:raise RuntimeError('Streaming speech produced no decoded packets')
    raw=np.concatenate([p for _,p in packets]);rate=tts.mimi.sample_rate
    if float(np.max(np.abs(raw)))<.001:raise RuntimeError('Generated speech is below the diagnostic energy floor')
    audio,playout=render_packets(packets,rate,.16)
    sf.write(args.out/'model.wav',audio,rate,subtype='FLOAT')
    sf.write(args.out/'packet-stream.wav',raw,rate,subtype='FLOAT')
    ledger=[dict(packet=i,available_s=t,samples=len(p)) for i,(t,p) in enumerate(packets)]
    (args.out/'packets.json').write_text(json.dumps(ledger,indent=2)+'\n')
    report=dict(schema='dl-external-controller-tts-probe-v1',passed=True,
        text=text,backend=backend,backend_ready_s=backend_ready,model_load_seconds=load_seconds,
        wall_seconds=wall,audio_seconds=len(raw)/rate,sample_rate=rate,
        frame_compute_p95_s=float(np.percentile(frame_steps,95)),
        first_pcm_packet_available_s=packets[0][0],playout=playout,
        raw_peak=float(np.max(np.abs(raw))),raw_samples_above_one=int(np.count_nonzero(np.abs(raw)>1)),
        audio_sha256=sha256(args.out/'model.wav'),packet_stream_sha256=sha256(args.out/'packet-stream.wav'),
        assets_manifest_sha256=sha256(args.assets),loads=loads,execution=snapshot(tts.lm),
        mimi_load_approval_sha256=sha256(args.mimi_load_approval),strict_mimi_loads=mimi_loads,
        native_duplex_qualified=False,hybrid_system_qualified=False,microphone_acceptance=False,
        speech_model_routing=False,text_chosen_by='external gpt-6-luna controller',
        input_is_typed=True,input_text_complete_before_render=True,concurrent_listening=False,
        voice_is_published_embedding=True,voice_identity_is_accepted_reference=False,
        playout_is_causal_simulation=True,packet_stream_is_quality_diagnostic_only=True,
        upstream_dependency_versions=compat['package_versions'],torch_within_upstream_requirement=compat['torch_within_upstream_requirement'],
        gpu_compatibility_scope='this pinned single response on the recorded environment; not supported-version certification',
        training_approved=False,independent_speech_review='pending',human_listening='pending')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
