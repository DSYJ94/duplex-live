"""Artificial word/PCM fixtures exercise audit guards, never model capability."""
import copy
import math
import unittest

from data.synthetic.native_control_repair import build
from evals.native_control_source_quality import assess_clip
from evals.native_control_teacher_checks import assess


class ControlQuality(unittest.TestCase):
    def fixture(self, scene):
        import numpy as np
        events, end = [], 0
        for i, turn in enumerate(scene['turns']):
            start = 19200 if i == 0 else math.ceil((end + round(turn['gap_before_s'] * 24000)) / 1920) * 1920
            end = start + 2 * 24000
            events.append(dict(ch=turn['role'], text=turn['text'],
                               actual_start_sample=start, actual_end_sample=end))
        frames = math.ceil((end + scene['trailing_observation_s'] * 24000) / 1920) * 1920
        observation = (frames + 15 * 24000) / 24000 + 2
        pcm = np.zeros(round(observation * 24000), dtype='float32')
        words = []
        for i in scene['expected_assistant_turns']:
            start = events[i]['actual_start_sample'] / 24000 + .02
            for j, text in enumerate(scene['turns'][i]['text'].split()):
                a = start + j * .1
                words.append(dict(text=text, start=a, end=a + .06))
                pcm[round(a * 24000):round((a + .06) * 24000)] = .1
        text = ' '.join(w['text'] for w in words)
        speech = dict(model_text=text, asr_text=text, asr_segments=[dict(words=words)])
        quiet_end = frames if scene['release_turn'] is None else events[scene['release_turn']]['actual_end_sample']
        timeline = dict(scenario_id=scene['id'], events=events,
                        quiet_contract=dict(start_sample=events[scene['quiet_start_after_turn']]['actual_end_sample'],
                                            end_sample=quiet_end))
        session = dict(session_finalized=True, errors=[], source_samples=frames,
                       added_silence_samples=15 * 24000, sent_input_samples=frames + 15 * 24000,
                       observed_until_s=observation)
        log = [dict(type='local.input.append', first_sample=i,
                    sample_count=min(1920, session['sent_input_samples'] - i),
                    received_at_s=min(i + 1920, session['sent_input_samples']) / 24000)
               for i in range(0, session['sent_input_samples'], 1920)]
        log += [dict(type='local.input.completed', source_samples=frames,
                     sent_input_samples=session['sent_input_samples'],
                     received_at_s=session['sent_input_samples'] / 24000),
                dict(type='session.closed', received_at_s=observation)]
        checks = []
        for i, turn in enumerate(scene['turns']):
            text = turn['text'].split()
            checks.append(dict(turn_index=i, role=turn['role'], passed=True,
                               measured_words=[dict(text=t, start=.05 + j / len(text),
                                                    end=.05 + (j + .8) / len(text))
                                               for j, t in enumerate(text)]))
        return timeline, speech, pcm, 24000, log, session, checks

    def test_complete40_reference_phases_can_pass_without_training_approval(self):
        for scene in build()['cases']:
            with self.subTest(case=scene['id']):
                result = assess(scene, *self.fixture(scene))
                self.assertTrue(result['passed'], result['reasons'])
                self.assertFalse(result['training_approved'])
                self.assertFalse(result['model_behavior_qualified'])

    def test_acknowledgment_and_untranscribed_audio_both_fail(self):
        scene = next(c for c in build()['cases'] if c['family'] == 'no-release')
        for add_word in (True, False):
            timeline, speech, pcm, rate, log, session, checks = self.fixture(scene)
            start = timeline['quiet_contract']['start_sample'] / rate + .1
            pcm[round(start * rate):round((start + .2) * rate)] = .1
            if add_word:
                speech['asr_segments'][0]['words'].append(dict(text='Okay.', start=start, end=start + .2))
                speech['model_text'] += ' Okay.'
                speech['asr_text'] += ' Okay.'
            result = assess(scene, timeline, speech, pcm, rate, log, session, checks)
            self.assertFalse(result['passed'])
            self.assertTrue(any('non-silent PCM' in r for r in result['reasons']))

    def test_wrong_cue_cannot_release_a_correct_answer_early(self):
        for family in ('distractor-release', 'replace-cue'):
            scene = next(c for c in build()['cases'] if c['family'] == family)
            timeline, speech, pcm, rate, log, session, checks = self.fixture(scene)
            first = speech['asr_segments'][0]['words'][0]
            earlier = timeline['events'][2]['actual_end_sample'] / rate + .1
            shift = first['start'] - earlier
            for word in speech['asr_segments'][0]['words']:
                word['start'] -= shift
                word['end'] -= shift
            result = assess(scene, timeline, speech, pcm, rate, log, session, checks)
            self.assertFalse(result['passed'])
            self.assertIn('Teacher speaks during the requested quiet interval', result['reasons'])

    def test_silence_without_resume_and_restarting_first_step_fail(self):
        scene = next(c for c in build()['cases'] if c['family'] == 'pause-resume')
        for restart in (True, False):
            timeline, speech, pcm, rate, log, session, checks = self.fixture(scene)
            release = timeline['events'][scene['release_turn']]['actual_end_sample'] / rate
            before = [w for w in speech['asr_segments'][0]['words'] if w['end'] < release]
            after = [] if not restart else [dict(text=t, start=release + .6 + i * .1,
                                               end=release + .66 + i * .1)
                                          for i, t in enumerate(scene['no_restart_text'].split())]
            words = before + after
            speech.update(model_text=' '.join(w['text'] for w in words), asr_text=' '.join(w['text'] for w in words),
                          asr_segments=[dict(words=words)])
            result = assess(scene, timeline, speech, pcm, rate, log, session, checks)
            self.assertFalse(result['passed'])
            self.assertTrue(any('restarted answer' in r for r in result['reasons']))

    def test_short_audio_is_not_evidence_of_complete_no_release_observation(self):
        scene = build()['cases'][0]
        timeline, speech, pcm, rate, log, session, checks = self.fixture(scene)
        session['sent_input_samples'] -= rate
        next(e for e in log if e['type'] == 'local.input.completed')['sent_input_samples'] = session['sent_input_samples']
        result = assess(scene, timeline, speech, pcm, rate, log, session, checks)
        self.assertFalse(result['passed'])
        self.assertTrue(any('Incomplete actual source playback' in r for r in result['reasons']))

    def test_nonfinite_times_or_unnecessary_delegation_fail(self):
        scene = build()['cases'][8]
        for corrupt in ('time', 'delegate'):
            timeline, speech, pcm, rate, log, session, checks = self.fixture(scene)
            if corrupt == 'time':
                speech['asr_segments'][0]['words'][0]['end'] = float('nan')
            else:
                log.insert(0, dict(type='session.delegation.created'))
            self.assertFalse(assess(scene, timeline, speech, pcm, rate, log, session, checks)['passed'])

    def test_question_completion_uses_measured_words_not_a_silent_file_tail(self):
        scene = build()['cases'][8]
        timeline, speech, pcm, rate, log, session, checks = self.fixture(scene)
        release = scene['release_turn']
        complete = timeline['events'][release]['actual_start_sample'] / rate + checks[release]['measured_words'][-1]['end']
        old = speech['asr_segments'][0]['words'][0]['start']
        shift = old - complete - .1
        pcm[:] = 0
        for word in speech['asr_segments'][0]['words']:
            word['start'] -= shift
            word['end'] -= shift
            pcm[round(word['start'] * rate):round(word['end'] * rate)] = .1
        self.assertLess(speech['asr_segments'][0]['words'][0]['start'],
                        timeline['events'][release]['actual_end_sample'] / rate)
        judgment = assess(scene, timeline, speech, pcm, rate, log, session, checks)
        self.assertTrue(judgment['passed'], judgment['reasons'])
        checks[release]['measured_words'].pop()
        with self.assertRaisesRegex(ValueError, 'exact independently measured words'):
            assess(scene, timeline, speech, pcm, rate, log, session, checks)

    def test_late_or_missing_input_packets_cannot_use_the_planned_question_clock(self):
        scene = build()['cases'][8]
        for missing in (False, True):
            timeline, speech, pcm, rate, log, session, checks = self.fixture(scene)
            if missing:
                log.pop(1)
            else:
                release = scene['release_turn']
                sample = math.ceil(timeline['events'][release]['actual_start_sample']
                                   + checks[release]['measured_words'][-1]['end'] * rate)
                delayed = False
                for event in log:
                    if event['type'] == 'local.input.append' and event['first_sample'] + event['sample_count'] >= sample:
                        delayed = True
                    if delayed:
                        event['received_at_s'] += 2
                session['observed_until_s'] += 2
            result = assess(scene, timeline, speech, pcm, rate, log, session, checks)
            self.assertFalse(result['passed'])
            if missing:
                self.assertIn('Actual packet availability does not cover measured source words', result['reasons'])
            else:
                self.assertIn('Teacher speaks during the requested quiet interval', result['reasons'])

    def test_source_low_wer_does_not_bless_a_changed_cue_or_guessed_onset(self):
        import numpy as np
        reference = 'Please keep listening without any response at all until you hear me say the final cue beacon again much later in this conversation.'
        words = [dict(text=t, start=.1 + i * .08, end=.15 + i * .08) for i, t in enumerate(reference.split())]
        pcm = np.full(3 * 24000, .1, dtype='float32')
        result = dict(text=reference, segments=[dict(words=words)])
        self.assertTrue(assess_clip(reference, result, pcm, 24000)['passed'])
        wrong = copy.deepcopy(result)
        for word in wrong['segments'][0]['words']:
            if word['text'] == 'beacon':
                word['text'] = 'bacon'
        wrong['text'] = reference.replace('beacon', 'bacon')
        judgment = assess_clip(reference, wrong, pcm, 24000)
        self.assertLessEqual(judgment['raw_wer'], .05)
        self.assertFalse(judgment['passed'])
        pcm[:24000] = 0
        self.assertFalse(assess_clip(reference, result, pcm, 24000)['passed'])


if __name__ == '__main__':
    unittest.main()
