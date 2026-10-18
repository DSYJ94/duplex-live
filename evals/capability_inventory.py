"""Opt-in truthful tool-inventory clarification; no output rewriting or routing."""
import copy
import hashlib
import json
from pathlib import Path


class CapabilityInventoryGenerator:
    def __init__(self,base):self.base=base;self.requests=[]
    def __getattr__(self,name):return getattr(self.base,name)
    def request(self,messages,tools,assistant_prefill=''):
        names=sorted(t['function']['name'] for t in tools)
        if len(names)!=len(set(names)):raise ValueError('Duplicate actual function inventory')
        proposed=copy.deepcopy(messages)
        if not proposed or proposed[0]['role']!='system':raise ValueError('Expected actual controller system instructions')
        note=('\nCOMPLETE CURRENT OPERATION INVENTORY: '+json.dumps(names)+'. '
              'Only these functions are available for this request. Never select an absent function or claim an absent operation succeeded. '
              'Asking for more identifiers cannot enable a missing capability. Clarification is for missing information needed by an AVAILABLE operation; '
              'if an operation is unavailable, explain that limitation honestly rather than asking for the private value you were asked to retrieve. '
              'An absent cancellation operation cannot cancel a finished result. Address any remaining feasible part of the latest request through an available function. '
              'These restrictions do not remove any listed operation or authorize unrelated work.')
        proposed[0]['content']+=note
        digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
        self.requests.append(dict(functions=names,original_messages_sha256=digest(messages),proposed_messages_sha256=digest(proposed),tools_sha256=digest(tools),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        return self.base.request(proposed,tools,assistant_prefill=assistant_prefill)
