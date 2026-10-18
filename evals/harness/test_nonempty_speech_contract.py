"""Exercise the actual compiler and worker, including silence and invalid outputs."""
import json
import unittest
from training.control.curriculum import specifications
from training.control.paired_curriculum import compile_scene, assess, xml_action
from evals.nonempty_speech_contract import NonemptySpeechGenerator
from evals.recovery_worker import WorkerContractError


class NativeCapture:
    model_id = 'synthetic-schema-contract-test'

    def __init__(self, action):
        self.action = action
        self.tools = None

    def request(self, messages, tools, assistant_prefill=''):
        self.tools = tools
        text = xml_action(self.action)
        return dict(raw_text=text, cleaned_text=text, assistant_prefill=assistant_prefill, input_tokens=0, generated_tokens=0)


class NonemptySpeechContractTest(unittest.TestCase):
    def test_cancel_only_scope_does_not_change_other_native_tools(self):
        from evals.nonempty_speech_contract import constrain_tools
        tools = [dict(type='function', function=dict(name=name, parameters=dict(properties=dict(content=dict(type='string'))))) for name in ('answer', 'clarify', 'resolve_pending_speech', 'cancel_reasoning')]
        copied = json.loads(json.dumps(tools))
        result = constrain_tools(tools, 'cancel-only')
        self.assertEqual(tools, copied)
        self.assertEqual(result[:3], tools[:3])
        self.assertEqual(result[3]['function']['parameters']['properties']['content']['minLength'], 1)

    def test_actual_native_cancel_schema_excludes_worker_rejected_empty_speech(self):
        scene = next(s for s in specifications() if s['kind'] == 'cancel-preserve' and s['split'] == 'heldout')
        action = dict(name='cancel_reasoning', arguments=dict(task_ids=scene['expected']['cancelled'], content='Selected result delivery cancelled.', consume_pending_speech_ids=[]))
        capture = NativeCapture(action)
        # This compiles the actual worker tools through the production native aliases.
        from training.control.curriculum import label_scene
        result, quality = label_scene(scene, NonemptySpeechGenerator(capture))
        self.assertTrue(quality['passed'])
        schema = next(t['function'] for t in capture.tools if t['function']['name'] == 'cancel_reasoning')['parameters']['properties']['content']
        self.assertEqual(schema.get('minLength'), 1)
        self.assertEqual(schema.get('pattern'), r'\S')
        self.assertTrue(result['receipts'][0]['delivery_cancelled'])

    def test_silent_resume_stays_zero_argument_and_no_words_are_added(self):
        from training.control.paired_curriculum import specifications as paired
        scene = next(s for s in paired() if s['kind'] == 'resume')
        capture = NativeCapture(dict(name='resume_speech', arguments={}))
        result = compile_scene(scene, NonemptySpeechGenerator(capture))
        self.assertTrue(assess(scene, result)['passed'])
        schema = next(t['function'] for t in capture.tools if t['function']['name'] == 'resume_speech')['parameters']
        self.assertEqual(schema['properties'], {})
        self.assertEqual(json.loads(result['output'][0]['arguments']), {})

    def test_invalid_empty_generation_is_never_filled_or_repaired(self):
        scene = next(s for s in specifications() if s['kind'] == 'cancel-preserve' and s['split'] == 'heldout')
        capture = NativeCapture(dict(name='cancel_reasoning', arguments=dict(task_ids=scene['expected']['cancelled'], content='', consume_pending_speech_ids=[])))
        from training.control.curriculum import label_scene
        with self.assertRaisesRegex(WorkerContractError, 'empty speech'):
            label_scene(scene, NonemptySpeechGenerator(capture))


if __name__ == '__main__':
    unittest.main()
