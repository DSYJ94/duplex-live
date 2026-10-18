"""Pinned single-step streaming TTS service for concurrent hybrid probes.

Uses the same strict, fully verified load and public streaming reset as the
successful matched renderer. No controller, tool choice or microphone claim.
The upstream TTSGen is an example-local pinned API, not a stable library API.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from evals.moshi_model_download import sha256
from evals.kyutai_tts_compat import MODEL_REVISION,VOICE_REVISION,WHEEL_SHA256,verified_overlay
from evals.kyutai_tts_assets import DSM_COMMIT


def load_verified_tts(assets_path,mimi_approval):
    manifest=json.loads(assets_path.read_text())
    mimi_gate=json.loads(mimi_approval.read_text())
    if (mimi_gate['schema']!='dl-kyutai-tts-mimi-load-cpu-v1' or not mimi_gate['passed']
            or not mimi_gate['original_default_loader_failed'] or mimi_gate['gpu_used'] is not False
            or mimi_gate['assets_manifest_sha256']!=sha256(assets_path)
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
    from evals.kyutai_tts_mimi_load import strict_mimi_loader
    spec=importlib.util.spec_from_file_location('pinned_kyutai_streaming_example',example)
    upstream=importlib.util.module_from_spec(spec);sys.modules[spec.name]=upstream;spec.loader.exec_module(upstream)
    loads=[];mimi_loads=[];load_start=time.perf_counter()
    math_attention()
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
    return tts,upstream,attributes,dict(loads=loads,strict_mimi_loads=mimi_loads,
        load_seconds=load_seconds,assets_sha256=sha256(assets_path),
        mimi_approval_sha256=sha256(mimi_approval),execution=snapshot(tts.lm))


class StreamingSpeaker:
    """Main-thread GPU actor. One step never renders an entire utterance."""
    def __init__(self,assets_path,mimi_approval):
        self.tts,self.upstream,self.attributes,self.evidence=load_verified_tts(assets_path,mimi_approval)
        self.gen=None;self.active=False;self.frames=[]

    def open(self):
        self.codec_context=self.tts.mimi.streaming(1);self.codec_context.__enter__()
        self.gen=self.upstream.TTSGen(self.tts,[self.attributes],on_frame=self._on_frame)

    def _on_frame(self,frame):
        import numpy as np
        if not bool((frame!=-1).all()):return
        pcm=self.tts.mimi.decode(frame[:,1:,:]).detach().cpu().numpy()[0,0].copy()
        if pcm.shape!=(1920,) or not np.isfinite(pcm).all():
            raise RuntimeError('Invalid actual single-step TTS frame')
        self.frames.append(pcm)

    def start(self,text):
        if self.active:raise RuntimeError('Cannot overlap independent speech generations')
        if not isinstance(text,str) or not text.strip():raise ValueError('Empty speech')
        self.gen.lm_gen.reset_streaming();self.tts.mimi.reset_streaming()
        self.gen.offset=0;self.gen.state=self.tts.machine.new_state([])
        self.frames=[];self.flush_remaining=None
        for entry in self.upstream.prepare_script(self.tts,text,first_turn=True):
            self.gen.append_entry(entry)
        self.active=True

    def step(self):
        if not self.active:raise RuntimeError('No active speech generation')
        before=time.perf_counter();self.frames=[];self.gen._step()
        # Match the pinned upstream process_last termination exactly.
        if self.flush_remaining is None:
            if not self.gen.state.entries and self.gen.state.end_step is None:
                self.flush_remaining=self.tts.delay_steps+max(self.tts.lm.delays)+8
        else:
            self.flush_remaining-=1
            if self.flush_remaining==0:self.active=False
        if self.gen.offset>2500:raise RuntimeError('TTS exceeded bounded 200s utterance generation')
        return dict(pcm=self.frames.copy(),done=not self.active,compute_s=time.perf_counter()-before)

    def stop(self):
        self.active=False;self.frames=[]

    def warmup(self):
        self.start('Preparing the streaming speech system.')
        count=0;before=time.perf_counter()
        while self.active:count+=len(self.step()['pcm'])
        state=self.gen.lm_gen._streaming_state
        if not all(g._graph is not None for g in (state.graphed_main,state.graphed_depth)):
            raise RuntimeError('TTS warmup did not capture both CUDA graphs')
        return dict(seconds=time.perf_counter()-before,discarded_packets=count)

    def close(self):
        if self.gen is not None:
            self.gen.lm_gen._stop_streaming()
            self.codec_context.__exit__(None,None,None)
            self.gen=None
