import unittest
from evals.qwen_controller_probe import parse_action


class LocalActionFormat(unittest.TestCase):
    def test_exact_supported_serializations_preserve_arguments(self):
        expected={'name':'lookup_record','arguments':{'project':'Oak Central','field':'meeting room'}}
        import json
        encoded=json.dumps(expected)
        self.assertEqual(parse_action(encoded),expected)
        self.assertEqual(parse_action('<tool_call>'+encoded+'</tool_call>'),expected)

    def test_prose_multiple_calls_and_nonobject_arguments_do_not_become_actions(self):
        for text in ['I will check now.','<tool_call>{"name":"lookup_record","arguments":{}}</tool_call><tool_call>{"name":"cancel_lookup","arguments":{}}</tool_call>',
                     '{"name":"lookup_record","arguments":"Oak"}', '{"route":"delegate","content":"Done"}']:
            with self.subTest(text=text),self.assertRaises(ValueError):parse_action(text)


if __name__=='__main__':unittest.main()
