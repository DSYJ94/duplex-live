import copy
import unittest
from evals.qwen_response_executor import QwenResponseExecutor
from evals.qwen35_controller import parse_action
from evals.recovery_worker import tool


class CompactSpeechActions(unittest.TestCase):
    def test_native_function_choice_retains_semantic_route_and_preserves_actual_ids(self):
        executor=object.__new__(QwenResponseExecutor);executor.compact_replies=True;executor.pending_speech_protocol='explicit-actions'
        definitions=[tool('respond_to_user','Reply',dict(route=dict(type='string',enum=['answer','clarify']),content=dict(type='string'),consume_pending_speech_ids=dict(type='array',items=dict(type='string')))),
            tool('lookup_record','Read an actual record',dict(project=dict(type='string'),field=dict(type='string')))]
        original=copy.deepcopy(definitions);state=dict(dispatch_protocol='state-aware',pending_speech=[dict(id='actual-speech-7')])
        native=executor.model_functions(definitions,state);self.assertEqual(definitions,original)
        self.assertEqual([x['name'] for x in native],['answer','clarify','lookup_record','resolve_pending_speech'])
        for route in ('answer','clarify'):
            text='<tool_call><function='+route+'><parameter=content>Actual model words.</parameter></function></tool_call>'
            action=parse_action(text,native);self.assertEqual(executor.canonical_action(action,state),dict(name='respond_to_user',arguments=dict(route=route,content='Actual model words.',consume_pending_speech_ids=[])))
        action=dict(name='resolve_pending_speech',arguments=dict(route='answer',content='New answer.',speech_ids=['actual-speech-7']))
        self.assertEqual(executor.canonical_action(action,state)['arguments']['consume_pending_speech_ids'],['actual-speech-7'])
        action=dict(name='lookup_record',arguments=dict(project='Real Cedar',field='label'));self.assertEqual(executor.canonical_action(action,state),action)
        bad='<tool_call><function=clarify><parameter=route>answer</parameter><parameter=content>Words.</parameter></function></tool_call>'
        with self.assertRaises(ValueError):parse_action(bad,native)


if __name__=='__main__':unittest.main()
