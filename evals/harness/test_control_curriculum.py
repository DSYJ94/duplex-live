import copy
import json
import unittest
from training.control.curriculum import specifications,label_scene,xml_action,assess
class AuthoredGenerator:
 model_id='procedural-fixture-not-neural'
 def __init__(self,scene):self.scene=scene
 def request(self,messages,tools,assistant_prefill=''):
  self.messages=messages;self.tools=tools;e=self.scene['expected'];kind=self.scene['kind']
  content=e.get('exact_content',e.get('normalized_content','Which calculation do you mean?' if kind=='clarify' else 'Delivery of the selected result is cancelled.'))
  if e['route']=='silence':action=dict(name='withdraw_speech',arguments=dict(speech_ids=e['consumed']))
  elif e['route']=='cancel':action=dict(name='cancel_reasoning',arguments=dict(task_ids=e['cancelled'],content=content,consume_pending_speech_ids=e['consumed']))
  elif e['consumed']:action=dict(name='resolve_pending_speech',arguments=dict(route=e['route'],content=content,speech_ids=e['consumed']))
  else:action=dict(name=e['route'],arguments=dict(content=content))
  text=xml_action(action)
  return dict(raw_text=text,cleaned_text=text,assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0,fixture_only=True)
class ControlCurriculum(unittest.TestCase):
 def test_all_authored_labels_use_real_schema_and_worker_preservation_guards(self):
  rows=specifications();self.assertEqual(len({r['id'] for r in rows}),len(rows))
  families={s:{r['family'] for r in rows if r['split']==s} for s in ('train','heldout')};self.assertFalse(families['train']&families['heldout'])
  signatures=[]
  for scene in rows:
   compiler=AuthoredGenerator(scene);result,quality=label_scene(scene,compiler)
   self.assertTrue(quality['passed'],scene['id']);self.assertNotIn('"expected":',json.dumps(compiler.messages))
   signatures.append(json.dumps(dict(messages=compiler.messages,tools=compiler.tools),sort_keys=True))
  self.assertEqual(len(set(signatures)),len(rows),'Duplicated full student prompt')
 def test_correct_new_words_without_retirement_and_unrelated_cancellation_do_not_pass(self):
  from evals.duplex_reasoning_worker import ReasoningWorld
  scene=next(r for r in specifications(True) if r['kind']=='replace');result,quality=label_scene(scene,AuthoredGenerator(scene));self.assertTrue(quality['passed'])
  result['consume_pending_speech_ids']=[];self.assertFalse(assess(scene,result,ReasoningWorld({}))['passed'])
  scene=next(r for r in specifications(True) if r['kind']=='cancel-preserve');result,_=label_scene(scene,AuthoredGenerator(scene))
  world=ReasoningWorld({})
  for task in scene['active_calculations']:world.reserve(task);world.tasks[task]['cancelled']=True
  result['consume_pending_speech_ids']=[scene['pending_speech'][0]['id']]
  self.assertFalse(assess(scene,result,world)['passed'])
