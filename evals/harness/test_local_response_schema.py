import unittest
from evals.qwen_response_executor import validate


class LocalSchema(unittest.TestCase):
    def test_explicit_reply_action_preserves_speech_and_resolve_keeps_selected_ids(self):
        from evals.qwen_response_executor import QwenResponseExecutor
        from evals.recovery_worker import tool
        executor=object.__new__(QwenResponseExecutor);executor.pending_speech_protocol='explicit-actions'
        original=tool('respond_to_user','Reply',dict(route=dict(type='string',enum=['answer','clarify']),
            content=dict(type='string'),consume_pending_speech_ids=dict(type='array',items=dict(type='string'))))
        state=dict(dispatch_protocol='state-aware',pending_speech=[dict(id='old-a'),dict(id='old-b')])
        functions=executor.model_functions([original],state);definitions={d['name']:d for d in functions}
        self.assertIn('consume_pending_speech_ids',original['parameters']['properties'])
        ordinary=dict(name='respond_to_user',arguments=dict(route='answer',content='Lookup cancelled.'))
        validate(ordinary['arguments'],definitions['respond_to_user']['parameters'])
        self.assertEqual(executor.canonical_action(ordinary,state)['arguments']['consume_pending_speech_ids'],[])
        resolved=dict(name='resolve_pending_speech',arguments=dict(route='answer',content='Revised explanation.',speech_ids=['old-b']))
        validate(resolved['arguments'],definitions['resolve_pending_speech']['parameters'])
        converted=executor.canonical_action(resolved,state)
        self.assertEqual(converted['arguments'],dict(route='answer',content='Revised explanation.',consume_pending_speech_ids=['old-b']))
        for bad in ([],['unknown']):
            with self.assertRaises(ValueError):validate(dict(resolved['arguments'],speech_ids=bad),definitions['resolve_pending_speech']['parameters'])

    def test_missing_extra_wrong_type_and_wrong_enum_are_rejected(self):
        schema=dict(type='object',properties=dict(route=dict(type='string',enum=['answer','clarify']),content=dict(type='string'),
            consume=dict(type='array',items=dict(type='string'))),required=['route','content','consume'],additionalProperties=False)
        valid=dict(route='answer',content='Three.',consume=[]);validate(valid,schema)
        for value in [dict(route='answer',content='Three.'),dict(valid,extra='unavailable'),dict(valid,route='delegate'),dict(valid,consume='other'),dict(valid,content=None)]:
            with self.subTest(value=value),self.assertRaises(ValueError):validate(value,schema)


if __name__=='__main__':unittest.main()
