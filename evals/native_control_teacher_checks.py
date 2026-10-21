"""Measured whole-conversation checks for future control3 TRAIN teachers.

These checks never generate or gate speech. Raw output, independent ASR and the
complete input/close event log are required; semantic and numerical approval remain
separate. Historical teacher approvals are untouched.
"""
import math
from data.synthetic.native_control_repair import build


def assess(scene, timeline, speech, pcm, rate, events, session, source_checks):
    import numpy as np
    from data.synthetic.filter import norm_words, wer
    from evals.native_control_teacher_artifacts import input_availability, available_at
    canonical = next(s for s in build()['cases'] if s['id'] == scene['id'])
    if scene != canonical or timeline['scenario_id'] != scene['id']:
        raise ValueError('Exact authored control3 TRAIN case required')
    source = timeline['events']
    if len(source) != len(scene['turns']) or any(
            event['ch'] != turn['role'] or event['text'] != turn['text']
            for event, turn in zip(source, scene['turns'])):
        raise ValueError('Complete source speaker/word inventory required')
    if len(source_checks) != len(source):
        raise ValueError('Complete independent source word checks required')
    completion, onset = {}, {}
    for i, (event, turn, check) in enumerate(zip(source, scene['turns'], source_checks)):
        measured = check['measured_words']
        duration = (event['actual_end_sample'] - event['actual_start_sample']) / rate
        if (check['turn_index'] != i or check['role'] != turn['role'] or check['passed'] is not True
                or not measured or norm_words(turn['text']) != [t for w in measured for t in norm_words(w['text'])]
                or any(not math.isfinite(w['start'] + w['end']) or not 0 <= w['start'] < w['end'] <= duration + .08
                       for w in measured)
                or any(b['start'] < a['end'] - .02 for a, b in zip(measured, measured[1:]))):
            raise ValueError('Source completion needs exact independently measured words')
        completion[i] = event['actual_start_sample'] / rate + min(duration, measured[-1]['end'])
        onset[i] = event['actual_start_sample'] / rate + measured[0]['start']
    reasons = []
    closed = [e for e in events if e['type'] == 'session.closed']
    completed = [e for e in events if e['type'] == 'local.input.completed']
    quiet = timeline['quiet_contract']
    frames = session['source_samples']
    observed = session['observed_until_s']
    if (not session['session_finalized'] or session['errors'] or len(closed) != 1
            or len(completed) != 1 or type(frames) is not int or frames <= 0
            or frames % 1920 or not math.isfinite(observed)
            or session['added_silence_samples'] != 15 * 24000
            or session['sent_input_samples'] != frames + 15 * 24000
            or observed < session['sent_input_samples'] / 24000):
        reasons.append('Incomplete actual source playback, observation or session finalization')
    elif (completed[0]['source_samples'] != frames
          or completed[0]['sent_input_samples'] != session['sent_input_samples']
          or completed[0]['received_at_s'] < session['sent_input_samples'] / 24000):
        reasons.append('Actual input completion event does not cover the whole source')
    try:
        packets = input_availability(events, session['sent_input_samples'])
        for i in completion:
            completion[i] = max(completion[i], available_at(packets, math.ceil(completion[i] * rate)))
            onset[i] = max(onset[i], available_at(packets, math.ceil(onset[i] * rate)))
    except ValueError:
        reasons.append('Actual packet availability does not cover measured source words')
    if any(e['type'] in ('error', 'worker_failure', 'session.delegation.created')
           or e['type'].startswith('local.worker.') for e in events):
        reasons.append('Control-only teacher failed or unnecessarily delegated')
    if (rate != 24000 or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all()
            or float(np.abs(pcm).max()) > .995):
        reasons.append('Invalid or clipped raw teacher waveform')
        return dict(passed=False, reasons=reasons, training_approved=False)
    if len(pcm) / rate > observed + .08:
        reasons.append('Raw teacher waveform exceeds the actual observed session')
    words = [w for segment in speech.get('asr_segments', []) for w in segment.get('words', [])]
    text = speech.get('model_text', '')
    tokens = [t for w in words for t in norm_words(w['text'])]
    error = wer(norm_words(text), norm_words(speech.get('asr_text', '')))
    if (not words or speech.get('audit_failure') or speech.get('word_starts_within_recorded_audio') is False
            or error > .08 or tokens != norm_words(text)
            or tokens != norm_words(speech.get('asr_text', ''))):
        reasons.append('Independent complete teacher speech agreement failed')
    previous = 0.
    valid_times = True
    for w in words:
        a, b = w['start'], w['end']
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a + b) or not 0 <= a < b <= len(pcm) / rate + .04
                or a < previous - .02 or len(norm_words(w['text'])) != 1):
            valid_times = False
            break
        previous = b
    if not valid_times:
        reasons.append('Invalid measured teacher word intervals')
    begin = completion[scene['quiet_start_after_turn']]
    release = scene['release_turn']
    # A no-release target must remain quiet through the *actual observation end*,
    # including the drain. A short WAV cannot establish this by itself.
    end = observed if release is None else completion[release]
    if (not 0 <= begin < end <= observed
            or quiet['start_sample'] != source[scene['quiet_start_after_turn']]['actual_end_sample']
            or quiet['end_sample'] != (frames if release is None else source[release]['actual_end_sample'])
            or frames < max(e['actual_end_sample'] for e in source)):
        reasons.append('Quiet boundary differs from complete source instruction/release')
    if valid_times and any(w['start'] < end and w['end'] > begin for w in words):
        reasons.append('Teacher speaks during the requested quiet interval')
    segment = pcm[round(begin * rate):min(len(pcm), round(end * rate))]
    active = sum(float(np.sqrt(np.mean(segment[i:i + 480] ** 2))) > .001
                 for i in range(0, len(segment), 480))
    if active:
        reasons.append('Teacher emits non-silent PCM while quiet, even if ASR misses it')
    phases, assigned = [], set()
    if valid_times:
        for i in scene['expected_assistant_turns']:
            preceding = max(j for j in range(i) if scene['turns'][j]['role'] == 'user')
            following = next((j for j in range(i + 1, len(source)) if source[j]['ch'] == 'user'), None)
            start = completion[preceding]
            stop = observed if following is None else onset[following]
            selected = [j for j, w in enumerate(words) if start <= w['start'] and w['end'] <= stop]
            actual = [t for j in selected for t in norm_words(words[j]['text'])]
            expected = norm_words(scene['turns'][i]['text'])
            passed = actual == expected
            if not passed:
                reasons.append(f'Missing, early, extra or restarted answer in authored phase{i}')
            assigned.update(selected)
            phases.append(dict(assistant_turn=i, start_s=start, end_s=stop,
                               expected_words=expected, measured_words=actual, passed=passed))
        if assigned != set(range(len(words))):
            reasons.append('Teacher contains speech outside its commanded answer phases')
    return dict(passed=not reasons, reasons=reasons, raw_speech_wer=error,
                quiet_start_s=begin, quiet_end_s=end, quiet_active_20ms_frames=active,
                terminal_observation_included=release is None, answer_phases=phases,
                semantic_review_required=True, native_numerical_review_required=True,
                training_approved=False, model_behavior_qualified=False)
