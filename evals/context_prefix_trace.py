"""Cloned first-prefix layer evidence, outside CUDA graphs; no model changes."""


class PrefixTrace:
    def __init__(self, lm, enabled):
        self.values, self.handles = {}, []
        def save(name, value):
            if enabled(): self.values[name] = value[:, -1:].detach().cpu().clone()
        def output(name):
            def hook(module, args, value): save(name, value)
            return hook
        modules = {'text_emb': lm.text_emb, 'out_norm': lm.out_norm,
            **{'emb.'+str(i): m for i,m in enumerate(lm.emb)},
            **{'layer.'+str(i): m for i,m in enumerate(lm.transformer.layers)}}
        for name,module in modules.items():
            self.handles.append(module.register_forward_hook(output(name)))
        def before(module, args): save('transformer_input', args[0])
        self.handles.append(lm.transformer.register_forward_pre_hook(before))

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()
