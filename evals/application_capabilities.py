"""Opt-in actual deployment capability descriptions, distinct from tool gating.

The two fixed worker implementations define these profiles. User words never
select a profile or an action; all native schemas and generated outputs survive.
"""
import copy
import hashlib
import json
from pathlib import Path

PROFILES = {
    'reasoning-development': dict(
        supported_external_operations=['delegate the unchanged user question to the stronger reasoning service', 'cancel delivery of identified active reasoning results'],
        private_record_reader_supported=False,
        messaging_or_email_sender_supported=False,
        calendar_or_booking_supported=False,
        browser_or_computer_actions_supported=False),
    'records': dict(
        supported_external_operations=['read user-identified project fields from the actual supplied record store', 'cancel identified active record lookups', 'replace identified active record lookups'],
        private_record_reader_supported=True,
        record_read_requires_user_project_identification=True,
        record_execution_tool_can_be_withheld_until_identified=True,
        messaging_or_email_sender_supported=False,
        calendar_or_booking_supported=False,
        browser_or_computer_actions_supported=False),
}
ALLOWED = {
    'reasoning-development': {'respond_to_user', 'delegate_reasoning', 'withdraw_speech', 'resume_speech', 'cancel_reasoning', 'cancel_reasoning_and_resume_speech'},
    'records': {'respond_to_user', 'lookup_record', 'cancel_lookup', 'replace_lookup'},
}


class ApplicationCapabilityGenerator:
    def __init__(self, base, domain):
        self.base, self.domain, self.requests = base, domain, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def request(self, messages, tools, assistant_prefill=''):
        if not messages or messages[0]['role'] != 'system':
            raise ValueError('Expected actual application system instructions')
        proposed = copy.deepcopy(messages)
        proposed[0]['content'] += (
            '\nTRUSTED CURRENT APPLICATION CAPABILITIES: '+json.dumps(PROFILES[self.domain], sort_keys=True)+'. '
            'This describes supported operations of this deployment, not just the functions exposed in the current step. '
            'An operation absent from the supported external operations cannot be executed here. '
            'A spoken answer is never an external execution receipt. Do not claim an unsupported operation happened. '
            'For an unsupported operation, state the limitation honestly through an ordinary answer; asking for passwords, access permission or the requested private value cannot add a missing capability. '
            'For a SUPPORTED record reader with missing project identification, ask which project; do not claim the reader is unavailable. '
            'Use only the currently provided native functions, whose argument and state guards remain authoritative.')
        digest = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
        self.requests.append(dict(domain=self.domain, capabilities=PROFILES[self.domain],
            original_messages_sha256=digest(messages), proposed_messages_sha256=digest(proposed),
            original_messages=copy.deepcopy(messages), proposed_messages=copy.deepcopy(proposed),
            tools_sha256=digest(tools), code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        return self.base.request(proposed, tools, assistant_prefill=assistant_prefill)


class ApplicationCapabilityExecutor:
    def __init__(self, base, *, receipt_contract=False):
        self.base, self.requests = base, []
        self.receipt_contract = receipt_contract

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, payload):
        state = payload['local_worker_state']; domain = state.get('domain', 'records')
        if domain not in PROFILES or state['dispatch_protocol'] != 'state-aware':
            raise ValueError('Unsupported actual application profile')
        names = {t['name'] for t in payload.get('tools', [])}
        if not names <= ALLOWED[domain]:
            raise ValueError('Actual operation registry differs from the declared profile')
        with self.base.lock:
            original = self.base.generator
            generation_base = original
            if self.receipt_contract:
                from evals.instruction_contract import InstructionContractGenerator
                receipts = None
                if payload.get('tool_choice') == 'none':
                    if payload.get('previous_response_id') not in self.base.history:
                        raise ValueError('Receipt correction requires the actual prior response')
                    receipts = payload['input']
                    if not receipts or any(r.get('type') != 'function_call_output' for r in receipts):
                        raise ValueError('Receipt correction requires actual function outputs')
                generation_base = InstructionContractGenerator(original, 'receipt', receipts)
            proposal = ApplicationCapabilityGenerator(generation_base, domain)
            self.base.generator = proposal
            try:
                return self.base(payload)
            finally:
                self.base.generator = original
                self.requests.extend(proposal.requests)
