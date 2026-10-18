"""Opt-in schema alignment for actions whose speech must be nonempty."""
import copy
import hashlib
import json
from pathlib import Path


def constrain_tools(tools):
    proposal = copy.deepcopy(tools)
    speech_actions = {'answer', 'clarify', 'respond_to_user', 'resolve_pending_speech', 'cancel_reasoning'}
    for tool in proposal:
        function = tool['function']
        if function['name'] not in speech_actions:
            continue
        content = function['parameters']['properties'].get('content')
        if content is None or content.get('type') != 'string':
            raise ValueError('Expected the existing user-facing speech field')
        # These constraints describe the worker's existing .strip() guard.
        # Silent withdrawal and bound resume expose no content field at all.
        content.update(minLength=1, pattern=r'\S')
    return proposal


class NonemptySpeechGenerator:
    def __init__(self, base):
        self.base = base
        self.requests = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def request(self, messages, tools, assistant_prefill=''):
        proposal = constrain_tools(tools)
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        self.requests.append(dict(original_tools_sha256=digest(tools), proposed_tools_sha256=digest(proposal),
                                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        return self.base.request(messages, proposal, assistant_prefill=assistant_prefill)
