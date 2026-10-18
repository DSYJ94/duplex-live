"""Actual worker/compiler seams; fixture actions test guards, not model ability."""
import copy
import threading
import unittest
from evals.instruction_contract import InstructionContractExecutor, InstructionContractGenerator, RESUME_RULE, RECEIPT_RULE
from evals.instruction_contract_probe import resume_fixture, record_fixture
from training.control.curriculum import xml_action


class Capture:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.inputs = []
    def request(self, messages, tools, assistant_prefill=''):
        self.inputs.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        text = xml_action(next(self.actions))
        return dict(raw_text=text, cleaned_text=text, input_tokens=0, generated_tokens=0)


def cpu_executor(generator):
    from evals.qwen35_controller import Qwen35ResponseExecutor
    base = object.__new__(Qwen35ResponseExecutor)
    base.model_id = 'fixture'; base.compact_replies = True; base.spoken_output_contract = False
    base.pending_speech_protocol = 'explicit-actions'; base.prompt_profile = 'contract-examples'
    base.attention_backend = 'math'; base.generator = generator; base.history = {}; base.counter = 0; base.lock = threading.RLock()
    class CPU:
        def __getattr__(self, name):
            return getattr(base, name)
        @property
        def generator(self):
            return base.generator
        @generator.setter
        def generator(self, value):
            base.generator = value
        def __call__(self, payload):
            return base._request(payload)
    return CPU()


class InstructionContractTest(unittest.TestCase):
    def test_actual_resume_compiler_preserves_functions_schemas_and_real_cancellation(self):
        for status in ('active', 'finished', 'unrelated-active'):
            action = dict(name='cancel_reasoning_and_resume_speech', arguments=dict(task_ids=['blue-count'])) if status == 'active' else dict(name='resume_speech', arguments={})
            capture = Capture([action]); base = cpu_executor(capture); proposed = InstructionContractExecutor(base, 'resume')
            scene, run = resume_fixture(status, 0, base.model_id)
            result, checks, tasks = run(proposed)
            self.assertTrue(all(checks.values()), checks)
            audit = proposed.requests[0]
            self.assertNotIn(RESUME_RULE, audit['proposed_messages'][0]['content'])
            self.assertEqual(audit['original_messages'][1:], audit['proposed_messages'][1:])
            original, changed = copy.deepcopy(audit['original_tools']), copy.deepcopy(audit['proposed_tools'])
            next(t['function'] for t in changed if t['function']['name']=='resume_speech')['description'] = next(t['function'] for t in original if t['function']['name']=='resume_speech')['description']
            self.assertEqual(original, changed)
            self.assertIs(base.generator, capture)

    def test_actual_post_read_phase_only_changes_instruction_and_retains_receipt(self):
        scene, run = record_fixture(0, 'only-value', 'fixture')
        capture = Capture([dict(name='lookup_record', arguments=dict(project='Silver Orchard', field='dispatch emblem')),
                           dict(name='answer', arguments=dict(content='violet canoe'))])
        base = cpu_executor(capture); proposed = InstructionContractExecutor(base, 'receipt')
        result, checks, tasks = run(proposed)
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual([r['changed'] for r in proposed.requests], [False, True])
        audit = proposed.requests[1]
        self.assertEqual(audit['original_messages'][-1], dict(role='user', content=RECEIPT_RULE))
        self.assertEqual(audit['original_messages'][:-1], audit['proposed_messages'][:-1])
        self.assertEqual(audit['original_tools'], audit['proposed_tools'])
        self.assertEqual(result['receipts'][0]['value'], 'violet canoe')
        self.assertIs(base.generator, capture)

    def test_unaffected_request_uses_original_objects_and_changed_sources_fail_closed(self):
        class Identity:
            def request(self, messages, tools, assistant_prefill=''):
                self.messages, self.tools = messages, tools
                return 'unchanged'
        base = Identity(); wrapper = InstructionContractGenerator(base, 'resume')
        messages = [dict(role='system', content='Original contract.'), dict(role='user', content=RECEIPT_RULE)]
        tools = []
        self.assertEqual(wrapper.request(messages, tools), 'unchanged')
        self.assertIs(base.messages, messages); self.assertIs(base.tools, tools)
        self.assertFalse(wrapper.requests[-1]['changed'])
        with self.assertRaisesRegex(ValueError, 'instruction source changed'):
            wrapper.request(messages, [dict(function=dict(name='resume_speech'))])
        with self.assertRaisesRegex(ValueError, 'matching tool outputs'):
            InstructionContractGenerator(base, 'receipt', [dict(call_id='actual', output='actual receipt')]).request(messages, tools)


if __name__ == '__main__':
    unittest.main()
