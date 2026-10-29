"""Explicit experimental TTS attention backend; no weights/precision changes.

Pinned Venus init_tts_module forces eager unless the global backend is Flash2.
The pinned Transformers LlamaAttention dispatches on its own config at forward.
CPU parity is not native GPU/audio qualification; released remains default.
"""


def configure_tts_attention(model,implementation='released'):
    if implementation not in ('released','sdpa'):
        raise ValueError('Only released or explicit experimental SDPA TTS attention')
    backbone=model.tts.model
    previous=backbone.config._attn_implementation
    if implementation=='released':
        return dict(requested='released',actual=previous,experimental=False,native_audio_qualified=False)
    import transformers
    from transformers.models.llama.modeling_llama import LlamaAttention,LlamaModel
    if transformers.__version__!='4.51.3' or not isinstance(backbone,LlamaModel):
        raise ValueError('TTS backend opt-in requires the exact pinned Transformers Llama implementation')
    if backbone.training or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('TTS runtime experiment requires frozen eval weights')
    if any(type(layer.self_attn) is not LlamaAttention or layer.self_attn.config is not backbone.config for layer in backbone.layers):
        raise ValueError('All TTS attention layers must use the same audited dynamic dispatch config')
    if previous not in ('eager','sdpa') or backbone.config.attention_dropout!=0:
        raise ValueError('Expected the original zero-dropout eager TTS path')
    language_backend=model.llm.config._attn_implementation
    versions={name:p._version for name,p in backbone.named_parameters()}
    backbone.config._attn_implementation='sdpa'
    if model.llm.config._attn_implementation!=language_backend:
        raise ValueError('TTS experiment must not change the language model backend')
    assert all(p._version==versions[name] for name,p in backbone.named_parameters())
    return dict(requested='sdpa',previous=previous,actual=backbone.config._attn_implementation,
        experimental=True,transformers=transformers.__version__,weights_unchanged=True,
        precision_unchanged=True,native_audio_qualified=False)
