import json
from pathlib import Path
import unittest
from types import SimpleNamespace

from tokenizers import Tokenizer
from data.venus.context_timeline import BASE, DRAFT, PLAN, TOKENIZER, compile_scene, prerequisites
from training.venus.native_schedule import schedule


class ContextTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.users, cls.assistant = prerequisites()
        cls.plan = json.loads(PLAN.read_text())['scenes']
        cls.drafts = {s['id']: s for s in json.loads(DRAFT.read_text())['scenes']}
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def build(self, family):
        s = next(s for s in self.plan if s['family'] == family)
        r = json.loads((BASE/'context-teacher-work-v1'/(s['id']+'.json')).read_text())
        return compile_scene(s, self.drafts[s['id']], r, self.users, self.assistant, self.tokenizer), r

    def test_quiet_has_no_ack_or_action_before_release_end(self):
        t, _ = self.build('quiet_scope')
        cue = t['user_turns'][2]
        self.assertTrue(all(a['frame'] >= cue['start_seconds']+cue['seconds'] for a in t['actions']))

    def test_pause_resumes_exact_unsaid_tail_without_restart(self):
        t, _ = self.build('pause_resume')
        first, stop, resumed = t['actions']
        self.assertLess(stop['frame'], t['user_turns'][1]['start_seconds']+t['user_turns'][1]['seconds'])
        self.assertEqual(first['frame']+len(first['native_speech_units']), stop['frame'])
        self.assertFalse(first['final'])
        self.assertEqual([u['final'] for u in first['native_speech_units']], [False,True])
        self.assertTrue(resumed['resumes_unsaid_tail'])
        original = first['original_teacher_text'] + ' Next, cover the furniture. Finally, mask the edges.'
        self.assertEqual((first['text']+' '+resumed['text']).split(), original.split())

    def test_cancelled_travel_receipt_is_never_delivered(self):
        t, r = self.build('selective_cancel')
        retired = {e['work_id'] for e in r['events'] if e['kind']=='delivery_retired'}
        result_ids = {f['work_id'] for f in t['backend_feedback'] if 'receipt_sha256' in f}
        self.assertFalse(retired & result_ids)
        self.assertEqual(len(result_ids),1)

    def test_correction_delivers_only_current_revision_after_receipt(self):
        t, _ = self.build('correction')
        results = [f for f in t['backend_feedback'] if 'receipt_sha256' in f]
        self.assertEqual(len(results),1)
        self.assertEqual(json.loads(results[0]['text'])['revision'],2)
        self.assertGreater(t['actions'][-1]['frame'],results[0]['frame'])

    def test_all36_token_capacity_and_complete_words(self):
        config=json.loads(TOKENIZER.with_name('tokenizer_config.json').read_text())
        ids={v['content']:int(k) for k,v in config['added_tokens_decoder'].items()}
        encoder=SimpleNamespace(encode=lambda text,**kw:self.tokenizer.encode(text,add_special_tokens=False).ids)
        duplex=SimpleNamespace(tokenizer=encoder,listen_token_id=ids['<|listen|>'],speak_token_id=ids['<|speak|>'],turn_eos_token_id=ids['<|turn_eos|>'],chunk_eos_token_id=ids['<|chunk_eos|>'])
        for s in self.plan:
            r=json.loads((BASE/'context-teacher-work-v1'/(s['id']+'.json')).read_text())
            t=compile_scene(s,self.drafts[s['id']],r,self.users,self.assistant,self.tokenizer)
            scheduled=schedule(duplex,t)
            self.assertTrue(scheduled)
            self.assertTrue(all(len(tokens)<=19 for tokens in scheduled.values()))
            occupied=set()
            for a in t['actions']:
                if a['kind']=='speak':
                    self.assertEqual(' '.join(u['text'] for u in a['native_speech_units']).split(),a['text'].split())
                    n=len(a['native_speech_units'])
                elif a['kind']=='delegate':n=(len(self.tokenizer.encode('<delegate>'+a['text']+'</delegate>',add_special_tokens=False).ids)+15)//16
                else:n=1
                interval=set(range(a['frame'],a['frame']+n))
                self.assertFalse(occupied & interval)
                occupied |= interval
            self.assertLess(max(occupied),t['frames'])

    def test_preserved_v2_pause_reproduces_native_boundary_rejection(self):
        config=json.loads(TOKENIZER.with_name('tokenizer_config.json').read_text())
        ids={v['content']:int(k) for k,v in config['added_tokens_decoder'].items()}
        encoder=SimpleNamespace(encode=lambda text,**kw:self.tokenizer.encode(text,add_special_tokens=False).ids)
        duplex=SimpleNamespace(tokenizer=encoder,listen_token_id=ids['<|listen|>'],speak_token_id=ids['<|speak|>'],turn_eos_token_id=ids['<|turn_eos|>'],chunk_eos_token_id=ids['<|chunk_eos|>'])
        path=BASE/'context-timelines-v2/venus-context-repair-draft-train-pause_resume-00/timeline.json'
        original=json.loads(path.read_text())
        with self.assertRaises(AssertionError):schedule(duplex,original)
