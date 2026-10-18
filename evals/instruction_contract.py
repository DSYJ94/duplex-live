"""Inference-only instruction comparisons; no action or generated-text repair."""
import copy
import hashlib
import json
from pathlib import Path

RESUME_RULE = 'For stop, a requested revision, another question or a cancellation, use the corresponding provided action instead. '
RESUME_CORRECTED = (
    'For stop, a requested revision or another question, use the corresponding provided action instead. '
    'A cancellation requires a matching ACTIVE task: cancel that task through its provided cancellation action, preserving unrelated work. '
    'If the user also asks to continue the paused explanation, use the combined cancellation-and-resume action when that matching task is active. '
    'If the requested task is already finished or absent, there is nothing to cancel: never claim cancellation happened and never cancel a different active task. '
    'In that case, a simultaneous request to continue the unchanged explanation may use resume_speech without new words. ')
RESUME_DESCRIPTION = 'Continue the identified PAUSED speech generator from its retained audio cursor, without any new words. Use when the user asks to carry on or acknowledges the explanation without requesting a change. Do not use for stop, a shorter version, a new answer or cancellation.'
RESUME_DESCRIPTION_CORRECTED = (
    'Continue the identified PAUSED speech generator from its retained audio cursor, without any new words. '
    'Use when the user asks to carry on or acknowledges the unchanged explanation. '
    'Do not use for stop, a shorter version, a new answer, or when a matching active task must be cancelled. '
    'If the requested cancellation target is already finished or absent and the user also asks to continue, this may resume the explanation; it does not execute or claim cancellation. Never cancel unrelated work.')
RECEIPT_RULE = 'Report the actual tool receipt using exactly one provided reply-format function. Return exactly one function call, never plain prose. Include every required field and no fields outside the selected function schema. Do not execute another external operation.'
RECEIPT_CORRECTED = (
    'Answer the original latest user request using facts from the actual tool receipt. '
    'Honor the requested output form, including a request for only a value or for a full sentence. '
    'Do not add internal task or lookup metadata unless the user requested it. '
    'Return exactly one provided reply-format function call, never plain prose. '
    'Include every required field and no fields outside the selected function schema. Do not execute another external operation.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class InstructionContractGenerator:
    def __init__(self, base, mode, receipts=None):
        if mode not in ('resume', 'receipt'):
            raise ValueError('Expected one independent instruction comparison')
        self.base, self.mode, self.receipts = base, mode, receipts
        self.requests = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def request(self, messages, tools, assistant_prefill=''):
        proposed_messages, proposed_tools = messages, tools
        names = [t['function']['name'] for t in tools]
        changed = False
        if self.mode == 'resume' and 'resume_speech' in names:
            if messages[0]['role'] != 'system' or messages[0]['content'].count(RESUME_RULE) != 1:
                raise ValueError('Actual resumable instruction source changed')
            if names.count('resume_speech') != 1:
                raise ValueError('Ambiguous retained generator')
            proposed_messages, proposed_tools = copy.deepcopy(messages), copy.deepcopy(tools)
            proposed_messages[0]['content'] = proposed_messages[0]['content'].replace(RESUME_RULE, RESUME_CORRECTED)
            definition = next(t['function'] for t in proposed_tools if t['function']['name'] == 'resume_speech')
            if definition['description'] != RESUME_DESCRIPTION:
                raise ValueError('Actual retained-generator description changed')
            definition['description'] = RESUME_DESCRIPTION_CORRECTED
            changed = True
        if self.mode == 'receipt' and self.receipts is not None:
            if messages[-1] != dict(role='user', content=RECEIPT_RULE):
                raise ValueError('Actual post-receipt instruction source changed')
            actual = [dict(role='tool', tool_call_id=r['call_id'], content=r['output']) for r in self.receipts]
            if messages[-1-len(actual):-1] != actual:
                raise ValueError('Post-receipt instruction lacks the actual matching tool outputs')
            proposed_messages = copy.deepcopy(messages)
            proposed_messages[-1]['content'] = RECEIPT_CORRECTED
            changed = True
        self.requests.append(dict(mode=self.mode, changed=changed,
            original_messages_sha256=digest(messages), proposed_messages_sha256=digest(proposed_messages),
            original_tools_sha256=digest(tools), proposed_tools_sha256=digest(proposed_tools),
            original_messages=copy.deepcopy(messages), proposed_messages=copy.deepcopy(proposed_messages),
            original_tools=copy.deepcopy(tools), proposed_tools=copy.deepcopy(proposed_tools),
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        return self.base.request(proposed_messages, proposed_tools, assistant_prefill=assistant_prefill)


class InstructionContractExecutor:
    """Bind receipt phase from the actual protocol, never from user text."""
    def __init__(self, base, mode):
        self.base, self.mode, self.requests = base, mode, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, payload):
        receipts = None
        if payload.get('tool_choice') == 'none':
            if payload.get('previous_response_id') not in self.base.history:
                raise ValueError('Receipt phase requires an actual prior local response')
            receipts = payload['input']
            if not receipts or any(r.get('type') != 'function_call_output' for r in receipts):
                raise ValueError('Receipt phase requires actual function outputs')
        with self.base.lock:
            original = self.base.generator
            proposal = InstructionContractGenerator(original, self.mode, receipts)
            self.base.generator = proposal
            try:
                return self.base(payload)
            finally:
                self.base.generator = original
                self.requests.extend(proposal.requests)
