"""Development ablation: learned correction only while naming one function.

A public, request-scoped PyTorch pre-forward hook observes generated native
syntax. It never reads the user text, selects a function, repairs an argument or
modifies library files. Frozen projection handles all content/argument tokens.
"""
from training.control.head_adapter import ResidualHead
from training.control.function_boundary import name_phase


class FunctionNameHead(ResidualHead):
    def __init__(self,base,delta):super().__init__(base,delta);self.active=False;self.enabled=True
    def forward(self,hidden):return super().forward(hidden) if self.enabled and self.active else self.base(hidden)


class FunctionNameGenerator:
    def __init__(self,original,head):self.original=original;self.head=head
    def request(self,messages,tools,assistant_prefill=''):
        if assistant_prefill!='<tool_call>\n':raise ValueError('Scoped correction requires the known native function-call prefix')
        tokenizer=self.original.tokenizer
        prompt=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True)+assistant_prefill
        expected=tokenizer(prompt)['input_ids'];generated=[];first=True;positions=[]
        def observe(module,args,kwargs):
            nonlocal first
            ids=kwargs.get('input_ids')
            if ids is None or ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('Scoped correction requires batch-one token-ID generation')
            values=ids[0].tolist()
            if first:
                if values!=expected:raise ValueError('Scoped correction saw a different actual model prompt')
                first=False;self.head.active=False
            else:
                if len(values)!=1:raise ValueError('Scoped correction requires verified single-token cached decode')
                generated.extend(values);text=tokenizer.decode(generated,skip_special_tokens=False)
                self.head.active=name_phase(text)
                if self.head.active:positions.append(dict(after_generated_tokens=len(generated),native_prefix=text))
        self.head.active=False;hook=self.original.model.register_forward_pre_hook(observe,with_kwargs=True)
        try:
            generation=self.original.request(messages,tools,assistant_prefill)
            if not positions:raise ValueError('No actual function-name token boundary was observed')
            generation.update(output_correction_scope='function-name-only',function_name_positions=positions,content_projection_is_original=True,
                function_name_correction_enabled=self.head.enabled)
            return generation
        finally:
            hook.remove();self.head.active=False
