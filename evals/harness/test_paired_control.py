import copy,json,unittest
from training.control.paired_curriculum import specifications,compile_scene,assess,xml_action


def authored(scene):
    e=scene['expected'];kind=scene['kind'];sid=scene['pending_speech'][0]['id'] if scene['pending_speech'] else None
    if kind=='delegate':return dict(name='delegate_reasoning',arguments={})
    if kind=='resume':return dict(name='resume_speech',arguments={})
    if kind=='cancel-resume':return dict(name='cancel_reasoning_and_resume_speech',arguments=dict(task_ids=scene['active_calculations']))
    if kind=='withdraw':return dict(name='withdraw_speech',arguments=dict(speech_ids=[sid]))
    if kind=='clarify':return dict(name='clarify',arguments=dict(content='Which calculation should I cancel?'))
    content=e.get('exact_content',e.get('number'))
    if kind in ('replace','fulfill'):return dict(name='resolve_pending_speech',arguments=dict(route='answer',content=content,speech_ids=[sid]))
    return dict(name='answer',arguments=dict(content=content))


class Generator:
    model_id='fixture-only'
    def __init__(self,action):self.action=action
    def request(self,messages,tools,assistant_prefill=''):
        text=xml_action(self.action)
        return dict(raw_text=text,cleaned_text=text,assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0)


class PairedTest(unittest.TestCase):
    def test_every_authoring_contract_survives_actual_compiler_and_worker(self):
        scenes=specifications();self.assertEqual(len(scenes),162)
        for s in scenes:
            with self.subTest(scene=s['id']):self.assertTrue(assess(s,compile_scene(s,Generator(authored(s))))['passed'])

    def test_correct_new_number_does_not_hide_lost_old_answer(self):
        s=next(s for s in specifications() if s['kind']=='preserve')
        a=dict(name='resolve_pending_speech',arguments=dict(route='answer',content=s['expected']['number'],speech_ids=[s['pending_speech'][0]['id']]))
        q=assess(s,compile_scene(s,Generator(a)))
        self.assertTrue(q['checks']['number']);self.assertFalse(q['checks']['speech_retirement']);self.assertFalse(q['passed'])

    def test_correct_new_number_does_not_hide_stale_replaced_answer(self):
        s=next(s for s in specifications() if s['kind']=='replace')
        q=assess(s,compile_scene(s,Generator(dict(name='answer',arguments=dict(content=s['expected']['number'])))))
        self.assertTrue(q['checks']['number']);self.assertFalse(q['checks']['speech_retirement'])

    def test_hard_problem_direct_answer_is_not_a_delegation_label(self):
        s=next(s for s in specifications() if s['kind']=='delegate')
        self.assertFalse(assess(s,compile_scene(s,Generator(dict(name='answer',arguments=dict(content='Seventeen.')))))['passed'])

    def test_paired_families_stay_together_and_expected_labels_never_enter_prompt(self):
        scenes=specifications();train=[s for s in scenes if s['split']=='train'];held=[s for s in scenes if s['split']=='heldout']
        self.assertFalse({s['pair_id'] for s in train}&{s['pair_id'] for s in held});self.assertFalse({s['family'] for s in train}&{s['family'] for s in held})
        for s in scenes:
            from training.control.paired_curriculum import payload
            p,_,_,_=payload(s,'fixture-only');self.assertNotIn('expected',p['input']);self.assertNotIn('expected',p['local_worker_state'])

if __name__=='__main__':unittest.main()
