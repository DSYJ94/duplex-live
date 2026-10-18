"""Strict newer9B local controller component, text use only, no vision claim."""
import json
from pathlib import Path
import re
import threading
import time
from evals.moshi_model_download import sha256
from evals.qwen_response_executor import QwenResponseExecutor,validate
from evals.qwen_controller_probe import Generator


MTP_KEYS={
    'mtp.fc.weight','mtp.norm.weight','mtp.pre_fc_norm_embedding.weight','mtp.pre_fc_norm_hidden.weight',
    *('mtp.layers.0.'+name+'.weight' for name in (
        'mlp.down_proj','mlp.gate_proj','mlp.up_proj','input_layernorm','post_attention_layernorm',
        'self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','self_attn.q_norm','self_attn.k_norm')),
}


def retain_published_auxiliary(model,weights):
    """Retain all15 publisher MTP tensors as frozen storage, without running MTP.

    The official conditional-generation implementation explicitly ignores ^mtp.*
    during its normal load. We instead account for and retain these exact extra
    tensors, then strict-load and compare the entire source inventory. This adds
    no random inference weights and does not claim speculative MTP execution.
    """
    import torch
    extra=set(weights)-set(model.state_dict())
    if not extra:return []
    if (type(model).__name__!='Qwen3_5ForConditionalGeneration' or extra!=MTP_KEYS
            or '^mtp.*' not in model._keys_to_ignore_on_load_unexpected):
        raise RuntimeError('Unaccounted non-core source tensors: '+str(extra))
    for key in sorted(extra):
        parts=key.split('.');parent=model
        for name in parts[:-1]:
            if name not in parent._modules:parent.add_module(name,torch.nn.Module())
            parent=parent._modules[name]
        parent.register_parameter(parts[-1],torch.nn.Parameter(weights[key],requires_grad=False))
    return sorted(extra)


def load_model(assets_path,device='cuda:0',kernel_assets=None):
    import sys
    assets=json.loads(assets_path.read_text())
    from evals.qwen35_runtime import verified_runtime
    runtime=Path(assets['runtime_folder'])
    if sha256(runtime/'report.json')!=assets['runtime_report_sha256']:raise RuntimeError('Changed runtime proof')
    package,_=verified_runtime(runtime);sys.path.insert(0,str(package));kernel_package=None
    if kernel_assets is not None:
        from evals.qwen35_kernels import verified_kernels
        kernel_package,_=verified_kernels(kernel_assets,runtime);sys.path.insert(0,str(kernel_package))
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig,AutoTokenizer,AutoModelForImageTextToText
    from transformers.initialization import no_init_weights
    assets=json.loads(assets_path.read_text());pin=Path(__file__).with_name('assets')/'qwen35-controller-pins.json'
    if (assets['schema']!='dl-qwen35-controller-assets-v1' or not assets['passed'] or assets['pins_sha256']!=sha256(pin)
            or assets['code_sha256']!=sha256(Path(__file__).with_name('qwen35_controller_assets.py'))):raise RuntimeError('Missing exact text-model asset proof')
    folder=Path(assets['folder'])
    for row in assets['assets']:
        path=Path(row['path'])
        if path.parent!=folder or path.stat().st_size!=row['bytes'] or sha256(path)!=row['sha256']:raise RuntimeError('Changed local text-model asset')
    config=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    started=time.perf_counter()
    with no_init_weights():
        model=AutoModelForImageTextToText.from_config(config,torch_dtype=torch.bfloat16,attn_implementation='sdpa',trust_remote_code=False)
    model.tie_weights();weights={};index=json.loads((folder/'model.safetensors.index.json').read_text())['weight_map']
    for file in sorted(set(index.values())):
        loaded=load_file(folder/file,device='cpu')
        if set(weights)&set(loaded):raise RuntimeError('Duplicated source tensor')
        weights.update(loaded)
    if set(weights)!=set(index):raise RuntimeError('Model tensor inventory changed')
    auxiliary=retain_published_auxiliary(model,weights)
    aliases={};expected=set(model.state_dict());extra=expected-set(weights)
    if extra:
        if (extra!={'lm_head.weight'} or not config.tie_word_embeddings or 'model.embed_tokens.weight' not in weights
                or model.get_input_embeddings().weight is not model.get_output_embeddings().weight):
            raise RuntimeError('Unaccounted model tensors: '+str(extra))
        weights['lm_head.weight']=weights['model.embed_tokens.weight'];aliases['lm_head.weight']='model.embed_tokens.weight'
    result=model.load_state_dict(weights,strict=True,assign=True);model.tie_weights();state=model.state_dict()
    if result.missing_keys or result.unexpected_keys or set(state)!=set(weights):raise RuntimeError('Text model did not load completely')
    for key,value in state.items():
        if value.shape!=weights[key].shape or value.dtype!=weights[key].dtype or not torch.equal(value,weights[key]):
            raise RuntimeError('Loaded text-model tensor differs from source: '+key)
    if any(p.device.type=='meta' for p in model.parameters()):raise RuntimeError('Unmaterialized model parameter')
    model.eval().requires_grad_(False).to(device);model.tie_weights();torch.cuda.synchronize(device)
    selected=[]
    for name,module in model.named_modules():
        if type(module).__name__!='Qwen3_5GatedDeltaNet':continue
        functions={key:getattr(module,key).__module__ for key in ('causal_conv1d_update','chunk_gated_delta_rule','recurrent_gated_delta_rule')}
        functions['gated_norm']=type(module.norm).__module__
        if kernel_package is not None:
            for key in ('chunk_gated_delta_rule','recurrent_gated_delta_rule','gated_norm'):
                source=sys.modules[functions[key]].__file__
                if not functions[key].startswith('fla.') or not Path(source).resolve().is_relative_to(kernel_package.resolve()):
                    raise RuntimeError('Requested FLA operator not actually selected from verified package')
        selected.append(dict(module=name,**functions))
    if len(selected)!=24:raise RuntimeError('Unexpected Gated DeltaNet layer inventory')
    driver_discovery=None
    if kernel_package is not None:
        from evals.qwen35_cuda_discovery import configure_loaded_driver
        driver_discovery=configure_loaded_driver()
    proof=dict(strict=True,all_source_tensors_exact_on_cpu_before_device_copy=True,source_tensor_count=len(index),
        model_state_tensor_count=len(state),explicit_true_tied_aliases=aliases,missing=[],unexpected=[],
        declared_retained_unused_mtp_tensors=auxiliary,mtp_speculative_inference_executed=False,
        core_model_tensor_count=len(state)-len(auxiliary),vision_weights_loaded=True,vision_inference_executed=False,
        model_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        load_seconds=time.perf_counter()-started,device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype),
        assets_sha256=sha256(assets_path),repo=assets['repo'],revision=assets['revision'],
        triton_driver_discovery=driver_discovery,selected_inference_functions=selected,optional_kernel_assets_sha256=sha256(kernel_assets/'report.json') if kernel_assets else None,
        depthwise_convolution_functions=sorted({r['causal_conv1d_update'] for r in selected}),
        gpu_name=torch.cuda.get_device_name(device),gpu_total_memory=torch.cuda.get_device_properties(device).total_memory,
        compute_capability=list(torch.cuda.get_device_capability(device)))
    return model,tokenizer,proof



def parse_action(text,definitions):
    match=re.fullmatch(r'<tool_call>\s*<function=([a-z_]+)>(.*?)</function>\s*</tool_call>',text.strip(),re.S)
    if not match:raise ValueError('Expected exactly one published Qwen3.5 XML-style function call')
    name,body=match.groups();definition=next((d for d in definitions if d['name']==name),None)
    if definition is None:raise ValueError('Unavailable function')
    properties=definition['parameters']['properties'];arguments={};offset=0
    for item in re.finditer(r'<parameter=([a-z_]+)>(.*?)</parameter>',body,re.S):
        if body[offset:item.start()].strip():raise ValueError('Unexpected text between parameters')
        key,value=item.groups();offset=item.end()
        if key not in properties or key in arguments:raise ValueError('Unknown or duplicate parameter')
        value=value.strip();kind=properties[key]['type']
        if kind=='string':arguments[key]=value
        elif kind in ('array','object'):arguments[key]=json.loads(value)
        else:raise ValueError('Unsupported parameter type')
    if body[offset:].strip():raise ValueError('Unexpected parameter tail')
    validate(arguments,definition['parameters'])
    return dict(name=name,arguments=arguments)


class NonThinkingTokenizer:
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def __getattr__(self,name):return getattr(self.tokenizer,name)
    def __call__(self,*args,**kwargs):return self.tokenizer(*args,**kwargs)
    def apply_chat_template(self,*args,**kwargs):return self.tokenizer.apply_chat_template(*args,**dict(kwargs,enable_thinking=False))


class Qwen35ResponseExecutor(QwenResponseExecutor):
    provider_name='local-transformers-qwen35-9b'
    def __init__(self,assets,device='cuda:0',prompt_profile='contract-examples',attention_backend='math',pending_speech_protocol='array',kernel_assets=None,spoken_output_contract=False,compact_replies=False):
        if attention_backend!='math':raise ValueError('First newer9B proof requires explicit math attention')
        if prompt_profile not in ('base','contract-examples'):raise ValueError('Unknown prompt profile')
        if pending_speech_protocol not in ('array','explicit-actions'):raise ValueError('Unknown speech action protocol')
        if compact_replies and pending_speech_protocol!='explicit-actions':raise ValueError('Compact replies require explicit speech actions')
        self.compact_replies=compact_replies
        self.spoken_output_contract=spoken_output_contract
        self.pending_speech_protocol=pending_speech_protocol
        self.attention_backend=attention_backend
        self.model,self.tokenizer,self.proof=load_model(assets,device,kernel_assets)
        self.generator=Generator(self.model,NonThinkingTokenizer(self.tokenizer));self.model_id=self.proof['repo']
        self.prompt_profile=prompt_profile;self.lock=threading.RLock();self.history={};self.counter=0
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with torch.inference_mode(),torch.cuda.device(self.model.device),sdpa_kernel(SDPBackend.MATH):
            self.warmup=self.generator.request([dict(role='user',content='Say ready.')],[])
    def parse_generated_action(self,text,definitions):return parse_action(text,definitions)
    def tool_history_arguments(self,arguments):return dict(arguments)
