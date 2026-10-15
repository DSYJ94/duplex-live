"""Capture actual fresh native prefills for an independent HF graph comparison.

Diagnostic only: synchronous copies/projections invalidate performance timings.
No sampling setting, output token, or parameter is changed.
"""
from contextlib import contextmanager


@contextmanager
def capture_prefills(engine, out):
    import torch
    from lychee_fd.vllm_integration.model_lychee import LycheeDuplexState as state
    model=engine.engine.model_executor.driver_worker.model_runner.model
    captured=[];active=[None];last_length=[0]
    out.mkdir(exist_ok=False)
    def first_layer(module,args):
        active[0]=None
        positions,embeddings=args[:2]
        if len(captured)>=3 or positions.ndim!=1 or positions.numel()<2:return
        if int(positions[0])!=0 or int(positions[-1])!=positions.numel()-1:return
        body=state.stoken_ids.reshape(-1)
        has_speech=bool(((body>=151696)&(body<158257)).any())
        if len(captured)==1 and not has_speech:return
        if len(captured)==2 and positions.numel()<last_length[0]+100:return
        active[0]={'embeddings':embeddings.detach().cpu().clone(),
            'input_ids':state.text_input_ids.detach().cpu().clone(),
            'prefix_len':int(state.prefix_input_len),'positions':positions.detach().cpu().clone(),
            'speech_in_history':has_speech}
    hook=model.main_layers[0].register_forward_pre_hook(first_layer)
    original_side=model._project_side_last_logits
    original_text=model._sample_text_token_for_merge
    def raw(hidden):
        return model._project_logits(hidden[-1:]).detach().float().cpu().reshape(-1)
    def side(hidden,*args,**kwargs):
        if active[0] is not None:
            branch=kwargs['branch']
            active[0][branch+'_logits']=raw(hidden)
        result=original_side(hidden,*args,**kwargs)
        if active[0] is not None and kwargs['branch']=='stoken':
            value=active[0]
            if not {'text_logits','control_logits','next_text','stoken_logits'}<=value.keys():
                raise RuntimeError('Incomplete three-head native forward capture')
            path=out/f'prefill-{len(captured):03d}.pt'
            torch.save(value,path);captured.append(str(path));last_length[0]=len(value['positions']);active[0]=None
        return result
    def text(hidden):
        if active[0] is not None:active[0]['text_logits']=raw(hidden)
        result=original_text(hidden)
        if active[0] is not None:active[0]['next_text']=int(result)
        return result
    model._project_side_last_logits=side;model._sample_text_token_for_merge=text
    try:yield captured
    finally:
        hook.remove();model._project_side_last_logits=original_side;model._sample_text_token_for_merge=original_text
