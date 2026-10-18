"""Declared measured limits supplied as context, not a user-text router."""
import copy
import hashlib
import json

PROFILE=(
 'Measured reliability profile for this small nonthinking controller: in the CURRENT latest user request, '
 'multiplication involving a factor in the hundreds or larger is outside verified reliable direct calculation. '
 'When the user permits external help and delegate_reasoning is available, delegate such multiplication rather than guessing. '
 'Do not apply this rule to earlier questions or already running tasks. Always honor a prohibition on tools or another model; '
 'then answer locally only if reliable, or honestly say you cannot verify it. For a local numerical question asking only for the result, '
 'return just that number without restating operands or adding a preamble. This does not change which existing tasks may be cancelled or which speech remains owed.'
)
PROFILE_SHA256=hashlib.sha256(PROFILE.encode()).hexdigest()


class CapabilityGenerator:
    def __init__(self,original):self.original=original
    def request(self,messages,tools,assistant_prefill=''):
        if not messages or messages[0]['role']!='system':raise ValueError('Capability profile requires the explicit system context')
        actual=copy.deepcopy(messages);actual[0]['content']+='\nDECLARED CONTROLLER CAPABILITY PROFILE: '+PROFILE
        generation=self.original.request(actual,tools,assistant_prefill)
        generation.update(capability_profile_sha256=PROFILE_SHA256,capability_profile_is_declared_not_learned=True,
            actual_profiled_messages_sha256=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest())
        return generation
