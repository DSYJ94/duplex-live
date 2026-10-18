import copy
import unittest
from evals.capability_inventory import CapabilityInventoryGenerator


class Capture:
    def request(self,messages,tools,assistant_prefill=''):
        self.messages=messages;self.tools=tools;self.prefill=assistant_prefill
        return dict(raw_text='unmodified actual generation')


class InventoryContract(unittest.TestCase):
    def test_actual_tools_only_no_original_message_or_output_change(self):
        base=Capture();wrapped=CapabilityInventoryGenerator(base)
        messages=[dict(role='system',content='Existing contract.'),dict(role='user',content='Actual latest request.')]
        tools=[dict(type='function',function=dict(name='answer')),dict(type='function',function=dict(name='resume_speech'))]
        saved=copy.deepcopy(messages)
        self.assertEqual(wrapped.request(messages,tools,assistant_prefill='<tool_call>'),dict(raw_text='unmodified actual generation'))
        self.assertEqual(messages,saved);self.assertIs(base.tools,tools);self.assertEqual(base.messages[1],messages[1])
        self.assertIn('["answer", "resume_speech"]',base.messages[0]['content']);self.assertEqual(base.prefill,'<tool_call>')
        self.assertEqual(wrapped.requests[0]['functions'],['answer','resume_speech'])
    def test_missing_system_or_duplicate_inventory_is_rejected(self):
        wrapped=CapabilityInventoryGenerator(Capture())
        with self.assertRaises(ValueError):wrapped.request([dict(role='user',content='Actual')],[])
        item=dict(type='function',function=dict(name='answer'))
        with self.assertRaises(ValueError):wrapped.request([dict(role='system',content='Actual')],[item,item])


if __name__=='__main__':unittest.main()
