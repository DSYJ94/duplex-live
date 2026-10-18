"""New native labels must preserve quiet intervals and answer after task changes."""
import copy
import unittest
from evals.recovery_teacher_quality import assess
from evals.teacher_protocol import NATIVE_PROTOCOL, NATIVE_INSTRUCTIONS, validate_protocol


def speech(text, starts):
    return dict(model_text=text, wer_against_model_text=0., audio_duration_s=30.,
        asr_segments=[dict(words=[dict(text=w, start=t, end=t+.2) for w,t in zip(text.split(), starts)])])


class NativeTeacherTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(curriculum_version=2, required_facts=[['elapsed'], ['time']],
            quiet_until_user_ordinal=1, source_events=[
                dict(ch='user', text='Stay quiet until ready.', actual_start_ms=0, actual_end_ms=1000),
                dict(ch='user', text='Ready. What does a timer measure?', actual_start_ms=7000, actual_end_ms=8000)])
        self.events = [dict(type='session.closed')]
        self.session = dict(session_finalized=True, errors=[])

    def result(self, audio, kind='simple'):
        return assess(audio, self.events, self.meta, kind, self.session)

    def test_quiet_request_forbids_acknowledgement(self):
        self.assertTrue(self.result(speech('Elapsed time.', [9., 9.3]))['speech_and_fact_checks_pass'])
        bad = self.result(speech('Okay. Elapsed time.', [2., 9., 9.3]))
        self.assertIn('Assistant speech intrudes into an explicitly requested quiet interval', bad['reasons'])

    def test_word_tail_cannot_cross_into_quiet(self):
        self.meta.pop('quiet_until_user_ordinal')
        self.meta.update(quiet_between_user_ordinals=[0, 1], no_restart_after_cue=['book'])
        self.meta['source_events'][1]['actual_start_ms'] = 7000
        bad = self.result(speech('Elapsed time.', [1.3, 9.]))
        self.assertIn('Assistant speech intrudes into an explicitly requested quiet interval', bad['reasons'])

    def test_resume_must_not_restart_completed_step(self):
        self.meta.pop('quiet_until_user_ordinal')
        self.meta.update(quiet_between_user_ordinals=[0, 1], no_restart_after_cue=['book'])
        bad = self.result(speech('Elapsed time. Book.', [9., 9.3, 9.6]))
        self.assertIn('Paused task restarts an already completed step after the cue', bad['reasons'])

    def test_cancel_new_answer_must_follow_actual_new_question(self):
        self.meta = dict(curriculum_version=2, required_facts=[['temperature']], phrase='blue circle',
            source_events=[dict(ch='user', text='Check the code.', actual_start_ms=0, actual_end_ms=1000),
                dict(ch='user', text='Cancel that lookup. Instead, what does a thermometer measure?',
                    actual_start_ms=7000, actual_end_ms=8000)])
        self.events = [dict(type='session.delegation.created', received_at_s=2., delegation=dict(id='lookup')),
            dict(type='session.delegation.created', received_at_s=8.1, delegation=dict(id='cancel')),
            dict(type='local.worker.tool_result', received_at_s=9., task_id='cancel', result=dict(content='Cancelled.',
                receipts=[dict(tool='cancel_lookup', cancelled=True, task_ids=['lookup'])])), dict(type='session.closed')]
        good = self.result(speech('Cancelled. Temperature.', [10., 10.3]), 'cancel')
        self.assertTrue(good['speech_and_fact_checks_pass'], good['reasons'])
        bad = self.result(speech('Temperature. Cancelled.', [5., 10.]), 'cancel')
        self.assertIn('Cancellation omitted the new answer after the actual user request', bad['reasons'])

    def test_native_protocol_does_not_apply_to_comparisons(self):
        for kind in ('simple', 'private', 'clarify', 'cancel'):
            self.assertEqual(validate_protocol(NATIVE_PROTOCOL, 'teacher', kind), NATIVE_INSTRUCTIONS)
            with self.assertRaises(ValueError):
                validate_protocol(NATIVE_PROTOCOL, 'comparison', kind)


if __name__ == '__main__':
    unittest.main()
