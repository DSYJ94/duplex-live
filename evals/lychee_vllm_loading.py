"""Require full released branch/shard loading; disclose unused merge embeddings."""
from contextlib import contextmanager
import json


@contextmanager
def strict_loading(model_path):
    from lychee_fd.vllm_integration import model_lychee, engine
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    index = json.loads((model_path/'model.safetensors.index.json').read_text())['weight_map']
    receipt = {'model':None,'audio':None,'checkpoint_tensor_count':len(index)}
    original_model = model_lychee.LycheeDuplexForVLLM.load_weights
    original_audio = engine.AudioPreprocessor._load_weights

    def load_model(self, weights):
        if receipt['model'] is not None:
            raise RuntimeError('Strict single-GPU probe loaded multiple model instances')
        params = dict(self.named_parameters())
        loads = {name:set() for name in params}
        source_loaded, sources, active = set(),set(),[None]
        originals = []
        for name,param in params.items():
            had_loader = hasattr(param,'weight_loader')
            old = getattr(param,'weight_loader',default_weight_loader)
            def tracked(p,value,*args,_name=name,_old=old,**kwargs):
                if active[0] is None:
                    raise RuntimeError('Parameter initialized without an actual checkpoint tensor')
                shard = args[0] if args else kwargs.get('shard_id')
                _old(p,value,*args,**kwargs)
                loads[_name].add(shard)
                source_loaded.add(active[0])
            originals.append((param,had_loader,old))
            param.weight_loader = tracked
        def tracked_weights():
            for name,value in weights:
                if name not in index or name in sources:
                    raise RuntimeError('Unexpected or duplicate checkpoint tensor: '+name)
                sources.add(name);active[0]=name
                yield name,value
            active[0]=None
        try:
            result = original_model(self,tracked_weights())
        finally:
            for param,had_loader,old in originals:
                if had_loader: param.weight_loader=old
                else: delattr(param,'weight_loader')
        if sources != set(index):
            raise RuntimeError('Released engine did not visit the complete checkpoint')
        expected_aux = {name for name in index if name.startswith(('model.encoder.','encoder.',
            'model.adapter.','adapter.','model.merge_model.model.embed_tokens.',
            'model.merge_model.embed_tokens.','merge_model.model.embed_tokens.','merge_model.embed_tokens.'))}
        if sources-source_loaded != expected_aux:
            raise RuntimeError('Unaccounted or unexpectedly skipped checkpoint tensors: '+
                repr(sorted((sources-source_loaded)^expected_aux)))
        for name,actual in loads.items():
            expected = {'q','k','v'} if 'qkv_proj' in name else {0,1} if 'gate_up_proj' in name else {None}
            if actual != expected:
                raise RuntimeError('Missing/extra native branch shards: '+repr((name,actual,expected)))
        receipt['model'] = {'parameters':len(params),'loaded_source_tensors':len(source_loaded),
            'all_parameter_shards_loaded':True,'separate_audio_source_tensors':sorted(n for n in expected_aux
                if 'merge_model' not in n),
            'unused_merge_token_embeddings':sorted(n for n in expected_aux if 'merge_model' in n),
            'unused_reason':'released merge branch consumes shared hidden states, not token IDs',
            'side_head_fallback_used':False}
        return result

    def load_audio(self, path):
        if receipt['audio'] is not None:
            raise RuntimeError('Unexpected repeated audio preprocessor loading')
        counts, originals = {},[]
        for name in ('encoder','adapter'):
            module = getattr(self,name);original = module.load_state_dict
            def load(state,*args,_name=name,_original=original,**kwargs):
                if not state or _name in counts:
                    raise RuntimeError('Missing or duplicate actual audio state')
                # The official loader requests strict=False. This probe requires
                # exact keys and shapes instead; never initialize missing heads.
                output = _original(state,strict=True)
                counts[_name] = len(state)
                return output
            originals.append((module,original));module.load_state_dict=load
        try:
            result = original_audio(self,path)
        finally:
            for module,original in originals: module.load_state_dict=original
        if set(counts) != {'encoder','adapter'}:
            raise RuntimeError('Audio encoder or adapter was not actually loaded')
        expected = sum(name.startswith(('model.encoder.','encoder.','model.adapter.','adapter.')) for name in index)
        if sum(counts.values()) != expected:
            raise RuntimeError('Separate audio tensors do not cover the checkpoint index')
        receipt['audio'] = {**counts,'strict_keys_and_shapes':True}
        return result

    model_lychee.LycheeDuplexForVLLM.load_weights=load_model
    engine.AudioPreprocessor._load_weights=load_audio
    try:
        yield receipt
    finally:
        model_lychee.LycheeDuplexForVLLM.load_weights=original_model
        engine.AudioPreprocessor._load_weights=original_audio
