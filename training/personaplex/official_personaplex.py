"""Official NVIDIA inference contract, separate from the historical CS wrapper.

No adapter transplant, optimizer, output normalization or implicit downloads.
Published state must match every official inference tensor before strict load.
"""
import json,sys
from pathlib import Path
from .expanded_guard import sha

BASE=Path('/projects/u6ada/duplex-live')
UPSTREAM=BASE/'vendor/personaplex-nvidia-16419a2'
REVISION='16419a2ef7b98611f58c9df07477d00e5ad61d5f'
ASSETS=BASE/'public-models/personaplex-fdaf-v1'
PERSONA='You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.'


def official_imports():
    receipt=json.loads((UPSTREAM/'source-receipt.json').read_text())
    if receipt['revision']!=REVISION:raise ValueError('Exact NVIDIA runtime source required')
    for name,digest in receipt['files'].items():
        if sha(UPSTREAM/name)!=digest:raise ValueError('Official source changed: '+name)
    import sphn  # Reuse the existing compiled ARM extension from the completed official32 run.
    sys.path.insert(0,str(UPSTREAM/'moshi'))
    from moshi.models import LMGen,loaders
    from moshi.offline import wrap_with_system_tags,warmup,seed_all
    return LMGen,loaders,wrap_with_system_tags,warmup,seed_all


def model_schema():
    import torch
    from safetensors import safe_open
    _,loaders,_,_,_=official_imports()
    from moshi.models.lm import LMModel
    kwargs=dict(loaders._lm_kwargs,dep_q=16)
    model=LMModel(device='meta',dtype=torch.bfloat16,**kwargs)
    expected={k:list(v.shape)for k,v in model.state_dict().items()}
    with safe_open(str(ASSETS/'model.safetensors'),framework='pt',device='cpu')as f:
        published={k:f.get_slice(k).get_shape()for k in f.keys()}
    if expected!=published:raise ValueError('Complete official LM tensor membership/shape mismatch; no copied/random weights allowed')
    io=sys.modules['sphn']
    return dict(passed=True,source=REVISION,tensors=len(expected),dep_q=model.dep_q,delays=list(model.delays),no_missing_or_extra_or_expanded_weights=True,meta_only=True,audio_IO_module=io.__name__,audio_IO_sha256=sha(io.__file__),official_binary_environment_claimed=False,compiled_sphn_required=True),model


class OfficialEngine:
    frame_size=1920
    frame_rate=12.5

    def __init__(self,guard,voice='NATF2'):
        import torch
        import sentencepiece
        from safetensors.torch import load_file
        from evals.context_execution import math_attention
        from .expanded_capture import REFERENCE,REFERENCE_SHA
        self.guard=guard;guard(600);self.device=torch.device('cuda:0')
        if not torch.cuda.is_available()or torch.cuda.device_count()!=1:raise ValueError('One actual allocated GPU required')
        math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        LMGen,loaders,wrap,warmup,seed_all=official_imports();self.seed_all=seed_all
        self.schema,self.lm=model_schema();state=load_file(str(ASSETS/'model.safetensors'),device='cuda:0')
        self.lm.load_state_dict(state,strict=True,assign=True);del state
        self.lm.eval().requires_grad_(False)
        self.mimi=loaders.get_mimi(str(ASSETS/loaders.MIMI_NAME),device=self.device).eval()
        self.other_mimi=loaders.get_mimi(str(ASSETS/loaders.MIMI_NAME),device=self.device).eval()
        self.mimi.requires_grad_(False);self.other_mimi.requires_grad_(False)
        if self.mimi.sample_rate!=24000 or float(self.mimi.frame_rate)!=12.5:raise ValueError('Exact official native PCM clock required')
        self.spm=sentencepiece.SentencePieceProcessor(str(ASSETS/loaders.TEXT_TOKENIZER_NAME))
        self.lm_gen=LMGen(self.lm,device=self.device,temp=.8,temp_text=.7,audio_silence_frame_cnt=6,sample_rate=24000,frame_rate=12.5,prefill_system_prompt=True,pin_system_prompt=True,max_pinned_frames=512)
        for module in (self.mimi,self.other_mimi,self.lm_gen):module.streaming_forever(1)
        with torch.no_grad():warmup(self.mimi,self.other_mimi,self.lm_gen,self.device,self.frame_size)
        self.wrap=wrap;self.select_voice(voice)

    def select_voice(self,voice):
        import torch
        from .expanded_capture import REFERENCE,REFERENCE_SHA
        if voice=='NATF2':
            path=ASSETS/'NATF2.pt';value=torch.load(path,map_location='cpu',weights_only=True)
            if set(value)!={'embeddings','cache'}or not all(torch.is_tensor(v)and torch.isfinite(v).all()for v in value.values()):raise ValueError('Official packaged embedding/cache prompt required')
            self.lm_gen.load_voice_prompt_embeddings(str(path))
        elif voice=='RPBen-normalized':
            path=REFERENCE
            if sha(path)!=REFERENCE_SHA:raise ValueError('Unchanged original British source required')
            self.lm_gen.load_voice_prompt(str(path))
        else:raise ValueError('Explicit official NAT or normalized same-RPBen condition required')
        self.lm_gen.text_prompt_tokens=self.spm.encode(self.wrap(PERSONA));self.voice=voice;self.voice_sha256=sha(path)

    def reset(self):
        # Exact upstream server order; codec state after prompt is discarded.
        for module in (self.mimi,self.other_mimi,self.lm_gen):module.reset_streaming()
        self.seed_all(42);self.lm_gen.step_system_prompts(self.mimi);self.mimi.reset_streaming()

    def step(self,frame):
        import numpy as np,torch
        self.guard();pcm=np.asarray(frame,dtype=np.float32)
        if pcm.shape!=(1920,)or not np.isfinite(pcm).all():raise ValueError('One finite actual audio frame required')
        with torch.no_grad():
            x=torch.as_tensor(pcm,device=self.device)[None,None];codes=self.mimi.encode(x);self.other_mimi.encode(x)
            if codes.shape[-1]!=1:raise ValueError('Exact official80ms Mimi clock required')
            value=self.lm_gen.step(codes)
            if value is None:return dict(agent_pcm=None,text_token=None,is_ret=False)
            if value.shape[1]!=self.lm.dep_q+1:raise ValueError('All native output channels required')
            audio=self.mimi.decode(value[:,1:9]);self.other_mimi.decode(value[:,1:9]);token=int(value[0,0,0])
            return dict(agent_pcm=audio[0,0].cpu().numpy(),text_token=token,is_ret=token==4)

    def decode_text(self,tokens):return self.spm.decode([int(t)for t in tokens if t is not None and int(t)not in (0,1,2,3,4)])
