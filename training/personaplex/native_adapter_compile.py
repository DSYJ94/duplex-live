"""Keep the published native gating compiler policy after adding adapters.

The native kernel fuses activation/product in BF16. An eager replacement changes
rounding even when B is zero. Use the SAME vendor lazy compiler, never disable it.
"""
import sys
from types import MethodType
from evals.personaplex_adapters import AdapterLinear


def compile_gates(lm):
    changed=[]
    for name,module in lm.transformer.named_modules():
        if not (isinstance(getattr(module,'linear_in',None),AdapterLinear)and isinstance(getattr(module,'linear_out',None),AdapterLinear)):continue
        implementation=sys.modules.get(type(module).__module__)
        compiler=getattr(implementation,'torch_compile_lazy',None)
        if compiler is None:continue
        original=module.forward.__func__
        module.forward=MethodType(compiler(original),module);changed.append(name)
    return changed
