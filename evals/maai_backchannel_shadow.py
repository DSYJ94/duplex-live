"""Exploratory acoustic-only detector scores on completed causal hybrid captures.

No speech policy changes. Uses the verified Moshi Mimi continuous encoder as an
explicit alternative to the publisher's HF/resampling path; parity is unproved.
Every frame contains only arrived user and actually played assistant samples.
This is unpaced isolated inference, not concurrent-system or deployment approval.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from evals.moshi_model_download import sha256
from evals.maai_backchannel_head import load_head
from evals.kyutai_tts_compat import verified_overlay
from evals.kyutai_stt_assets import STT_REVISION
from evals.kyutai_tts_mimi_load import MIMI_SHA256,strict_mimi_loader


def load_mimi(assets_path,approval):
    assets=json.loads(assets_path.read_text());gate=json.loads(approval.read_text())
    if (assets['schema']!='dl-kyutai-stt-assets-v1' or not assets['passed'] or assets['revision']!=STT_REVISION
            or assets['code_sha256']!=sha256(Path(__file__).with_name('kyutai_stt_assets.py'))
            or gate['schema']!='dl-kyutai-tts-mimi-load-cpu-v1' or not gate['passed']
            or gate['code_sha256']!=sha256(Path(__file__).with_name('kyutai_tts_mimi_load.py'))
            or len(gate['records'])!=1 or not gate['records'][0]['all_tensors_exact']
            or gate['records'][0]['checkpoint_sha256']!=MIMI_SHA256):raise RuntimeError('Missing exact frozen codec proof')
    _,overlay=verified_overlay(Path(assets['compat_folder']));sys.path.insert(0,str(overlay));paths={}
    for row in assets['assets']:
        path=Path(row['path'])
        if sha256(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:raise RuntimeError('Changed frozen source asset')
        paths[row['role']]=path
    if sha256(paths['mimi'])!=MIMI_SHA256:raise RuntimeError('Unexpected acoustic encoder weights')
    from moshi.models.loaders import CheckpointInfo
    from evals.context_spanning_probe import audited_loads
    loads=[];exact=[]
    with audited_loads(loads),strict_mimi_loader(paths['mimi'],exact):
        info=CheckpointInfo.from_hf_repo('kyutai/stt-1b-en_fr-candle',moshi_weights=paths['model'],mimi_weights=paths['mimi'],
            tokenizer=paths['tokenizer'],config_path=paths['config'])
        mimi=info.get_mimi(device='cuda')
    checked=[r for r in loads if r['module_identity']==id(mimi)]
    if not checked or any(not r['strict'] or r['missing'] or r['unexpected'] for r in checked):raise RuntimeError('Codec did not load strictly')
    return mimi.eval().requires_grad_(False),dict(loads=loads,exact=exact,only_codec_used=True,asr_lm_loaded=False)


def run_case(head,mimi,folder,case,out):
    import numpy as np
    import soundfile as sf
    import torch
    user,rate=sf.read(folder/'input-user.wav',dtype='float32');own,sr=sf.read(folder/'model.wav',dtype='float32')
    if (rate!=24000 or sr!=rate or user.ndim!=1 or own.ndim!=1 or not np.isfinite(user).all() or not np.isfinite(own).all()
            or hashlib.sha256(user.tobytes()).hexdigest()!=case['user_pcm_sha256']
            or sha256(folder/'model.wav')!=case['audio_sha256']):raise RuntimeError('Changed actual causal capture')
    out.mkdir();frames=math.ceil(max(len(user),len(own))/1920);audio=np.zeros((2,frames*1920),dtype=np.float32)
    audio[0,:len(user)]=user;audio[1,:len(own)]=own
    events=[];latents=[];cache=None;times=[]
    with mimi.streaming(2):
        for _ in range(20):mimi.encode_to_latent(torch.zeros(2,1,1920,device='cuda'),quantize=False)
        mimi.reset_streaming()
        for i in range(frames):
            # No resampling or whole-file encoder: the current frame ends at
            # available_s. The transformer/codec caches contain past frames only.
            pcm=audio[:,i*1920:(i+1)*1920].copy();before=time.perf_counter()
            latent=mimi.encode_to_latent(torch.from_numpy(pcm).unsqueeze(1).to('cuda'),quantize=False)
            if latent.shape!=(2,512,1) or not torch.isfinite(latent).all():raise RuntimeError('Unexpected causal encoder frame')
            projected=torch.relu(head.decrease_dimension(latent.transpose(1,2)))
            result,cache=head(projected[:1],projected[1:],cache=cache)
            elapsed=time.perf_counter()-before;times.append(elapsed);latents.append(latent.detach().cpu())
            events.append(dict(frame=i,available_s=(i+1)*.08,p_user_backchannel=result['p_bc_det'][0],p_agent_backchannel=result['p_bc_det'][1],
                user_rms=float(np.sqrt(np.mean(pcm[0]**2))),agent_rms=float(np.sqrt(np.mean(pcm[1]**2))),compute_s=elapsed,
                user_samples_seen=min((i+1)*1920,len(user)),agent_playout_samples_seen=min((i+1)*1920,len(own))))
    # Event annotations enter only AFTER every acoustic model score is computed.
    source_events=json.loads((folder/'events.json').read_text());observations=[]
    for event in source_events:
        if event['kind']!='user_speech_started':continue
        frame=round(event['input_available_s']/.08)-1
        observations.append(dict(source_acoustic_start=event,first_scores=events[max(0,frame-1):min(frames,frame+6)]))
    tensor=out/'continuous-features.pt';torch.save(torch.cat(latents,dim=2),tensor)
    (out/'scores.json').write_text(json.dumps(events,indent=2)+'\n')
    report=dict(scenario=case['scenario'],source_user_pcm_sha256=case['user_pcm_sha256'],source_agent_audio_sha256=case['audio_sha256'],
        source_case_report_sha256=sha256(folder/'report.json'),source_events_sha256=sha256(folder/'events.json'),
        frames=frames,user_samples=len(user),agent_samples=len(own),every_real_sample_processed=True,
        features_sha256=sha256(tensor),scores_sha256=sha256(out/'scores.json'),compute_p95_s=float(np.percentile(times,95)),
        acoustic_start_observations=observations,ground_truth_transcript_supplied=False,
        only_arrived_user_and_played_agent_audio=True,annotations_read_after_all_scores=True,
        publisher_encoder_parity_proven=False,encoder_is_explicit_moshi_variant=True,
        threshold_selected=False,controls_assistant=False,backchannel_behavior_qualified=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','head-approval','stt-assets','mimi-approval','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--source-run',type=Path,action='append',required=True);args=ap.parse_args()
    gate=json.loads(args.head_approval.read_text())
    if (gate['schema']!='dl-maai-bcdet-head-cpu-v1' or not gate['passed']
            or gate['inventory_sha256']!=sha256(args.inventory)
            or gate['code_sha256']!=sha256(Path(__file__).with_name('maai_backchannel_head.py'))
            or not gate['load']['all_tensors_exact']):raise RuntimeError('Missing exact strict/causal CPU head proof')
    import torch
    head,headload=load_head(args.inventory,device='cuda');mimi,codecload=load_mimi(args.stt_assets,args.mimi_approval)
    from moshi.utils.compile import no_compile
    from torch.nn.attention import sdpa_kernel,SDPBackend
    args.out.mkdir(parents=True,exist_ok=False);results=[]
    with torch.inference_mode(),no_compile(),sdpa_kernel(SDPBackend.MATH):
        for source in args.source_run:
            parent=json.loads((source/'report.json').read_text())
            if parent['schema']!='dl-concurrent-hybrid-development-v1' or not parent['component_execution_completed']:
                raise RuntimeError('Shadow input requires a completed actual concurrent capture')
            for case in parent['results']:
                results.append(run_case(head,mimi,source/case['scenario'],case,args.out/(source.name+'-'+case['scenario'])))
    report=dict(schema='dl-maai-bcdet-moshi-shadow-v1',inference_executed=True,head_load=headload,codec_load=codecload,results=results,
        code_sha256=sha256(Path(__file__)),head_approval_sha256=sha256(args.head_approval),
        inventory_sha256=sha256(args.inventory),encoder_assets_sha256=sha256(args.stt_assets),
        encoder_variant='official Moshi continuous Mimi, raw24kHz; publisher HF/resampling parity NOT established',
        input_is_completed_causal_playout_simulation=True,unpaced_isolated_diagnostic=True,
        concurrent_system_qualified=False,controls_assistant=False,threshold_selected=False,
        production_approved=False,weight_license_approval=False,training_approved=False,microphone_acceptance=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
