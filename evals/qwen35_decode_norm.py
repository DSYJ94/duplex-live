"""Process-local, decode-only normalization compilation diagnostic.

Only stateless decoder input/post/final RMS norms are eligible. No recurrent or
attention cache is compiled/replaced. Prefill and all model parameters stay as
loaded. No package files or source methods are edited on disk.
"""
import re,time
from evals.moshi_model_download import sha256
from pathlib import Path


def rms_norm(x,weight,eps):
    import torch
    output=x.float()
    output=output*torch.rsqrt(output.pow(2).mean(-1,keepdim=True)+eps)
    output=output*(1.0+weight.float())
    return output.type_as(x)


def install(model):
    import torch
    from torch._dynamo.utils import counters
    expected=2*model.config.text_config.num_hidden_layers+1
    pattern=r'model\.language_model\.(?:layers\.[0-9]+\.(?:input_layernorm|post_attention_layernorm)|norm)'
    selected=[(name,m) for name,m in model.named_modules() if re.fullmatch(pattern,name)]
    if len(selected)!=expected or any(type(m).__name__!='Qwen3_5RMSNorm' for _,m in selected):raise ValueError('Unexpected decoder normalization inventory')
    if any(p.requires_grad for p in model.parameters()):raise ValueError('Normalization diagnostic requires frozen weights')
    compiled=torch.compile(rms_norm,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    report=dict(schema='dl-decode-normalization-compile-v1',modules=[],decode_calls=0,prefill_calls=0,whole_model_compiled=False,cache_changed=False,cuda_graphs=False,code_sha256=sha256(Path(__file__)))
    original_graphs=counters['stats']['unique_graphs']
    before=time.monotonic()
    with torch.inference_mode():
        for name,module in selected:
            x=torch.linspace(-1.,1.,module.weight.numel(),device=module.weight.device,dtype=torch.float32).to(module.weight.dtype).reshape(1,1,-1)
            reference=module(x);actual=compiled(x,module.weight,module.eps)
            torch.cuda.synchronize(module.weight.device);error=float((actual.float()-reference.float()).abs().max())
            if not torch.isfinite(actual).all() or error>.03125:raise ValueError('Compiled normalization differs beyond predeclared BF16 bound')
            report['modules'].append(dict(name=name,maximum_absolute_error=error,bitwise_equal=torch.equal(actual,reference)))
    report['warmup_seconds']=time.monotonic()-before
    report['compiler_counters_after_warmup']={k:dict(v) for k,v in counters.items()}
    report['compiled_graphs_created']=counters['stats']['unique_graphs']-original_graphs
    if report['compiled_graphs_created']!=1:raise ValueError('Expected exactly one shared compiled normalization graph; no eager fallback accepted')
    def wrapper(module,original):
        def forward(x):
            if x.ndim==3 and x.shape[0]==1 and x.shape[1]==1:
                report['decode_calls']+=1;return compiled(x,module.weight,module.eps)
            report['prefill_calls']+=1;return original(x)
        return forward
    for _,module in selected:module.forward=wrapper(module,module.forward)
    return report
