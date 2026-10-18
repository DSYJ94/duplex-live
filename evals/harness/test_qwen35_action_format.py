import unittest
from evals.qwen35_controller import parse_action
from evals.recovery_worker import tool


class PublishedActionFormat(unittest.TestCase):
    def setUp(self):
        self.tools=[tool('respond_to_user','Reply without external side effects',dict(
            route=dict(type='string',enum=['answer','clarify']),content=dict(type='string'),
            consume_pending_speech_ids=dict(type='array',items=dict(type='string'))))]
    def text(self,body):return '<tool_call>\n<function=respond_to_user>\n'+body+'\n</function>\n</tool_call>'
    def test_native_xml_string_and_array_arguments_remain_model_selected(self):
        value=parse_action(self.text('<parameter=route>answer</parameter><parameter=content>Four.</parameter><parameter=consume_pending_speech_ids>["old"]</parameter>'),self.tools)
        self.assertEqual(value,dict(name='respond_to_user',arguments=dict(route='answer',content='Four.',consume_pending_speech_ids=['old'])))
    def test_malformed_multiple_unknown_missing_duplicate_and_wrong_types_reject(self):
        good=self.text('<parameter=route>answer</parameter><parameter=content>Four.</parameter><parameter=consume_pending_speech_ids>[]</parameter>')
        values=[good+good,'Let me check. '+good,good.replace('function=respond_to_user','function=unknown'),
            good.replace('<parameter=route>answer</parameter>',''),good.replace('>[]</parameter>','>"old"</parameter>'),
            good.replace('<parameter=route>answer</parameter>','<parameter=route>answer</parameter><parameter=route>clarify</parameter>')]
        for value in values:
            with self.subTest(value=value),self.assertRaises(ValueError):parse_action(value,self.tools)


if __name__=='__main__':unittest.main()
