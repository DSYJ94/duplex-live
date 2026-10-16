"""Fail-closed selection of independently approved future text components.

These helpers do not approve artifacts or run optimization. Callers must first
validate native TRAIN caches and the actual-history restart report. Actual model
replies are conditioning only; they never supply gold labels.
"""
import math
from evals.recovery_prepare_data import spoken_words
from evals.recovery_encode import native_word_timing


def matching_user_stimulus(actual, teacher, actual_words, teacher_words):
    """Require identical user PCM; allow only additional zero tail padding."""
    import numpy as np
    if (actual.ndim != 1 or teacher.ndim != 1 or not len(actual) or not len(teacher)
            or not np.isfinite(actual).all() or not np.isfinite(teacher).all()
            or actual_words != teacher_words):
        return False
    common = min(len(actual), len(teacher))
    return bool(np.array_equal(actual[:common], teacher[:common])
        and not np.count_nonzero(actual[common:]) and not np.count_nonzero(teacher[common:]))


def actual_private_receipt(events, project, field, value, cut_s):
    """A failed backend notice or cancelled task cannot justify a fact target."""
    if not math.isfinite(cut_s) or cut_s < 0 or not spoken_words(value):
        raise ValueError('Invalid actual future boundary or private value')
    candidates = []
    for e in events:
        applied = e.get('applied_at_s')
        result = e.get('result', {})
        if (e.get('event') != 'backend_result_ready' or applied is None
                or not math.isfinite(applied) or applied > cut_s
                or result.get('route') not in ('answer', 'delegate')):
            continue
        words = spoken_words(result.get('content', ''))
        phrase = spoken_words(value)
        contains = any(words[i:i+len(phrase)] == phrase for i in range(len(words)-len(phrase)+1))
        receipts = [r for r in result.get('receipts', []) if r.get('tool') == 'lookup_record'
            and r.get('executed') is True and r.get('read_only') is True
            and (r.get('project'),r.get('field'),r.get('value')) == (project,field,value)]
        if contains and len(receipts) == 1:
            candidates.append(e)
    if len(candidates) != 1:
        raise RuntimeError('Need one actually delivered verified matching private result')
    event = candidates[0]
    if any(e.get('event') in ('cancelled_backend_rejected', 'cancelled_result_rejected')
            and e.get('task_id') == event['task_id'] for e in events):
        raise RuntimeError('Cancelled actual lookup is not eligible for a future fact target')
    return event


def sentence_target(text_codes, alignments, value, encode, ready_s, frame_rate=12.5):
    """Retain a measured complete result sentence in its source token grid.

    Return source coordinates only. Moving it after an actual student history
    is an explicitly new composition, not the original teacher's wall clock.
    """
    if frame_rate != 12.5 or not math.isfinite(ready_s) or ready_s < 0:
        raise ValueError('Invalid source receipt or native clock')
    native_word_timing(text_codes, alignments, encode, frame_rate)
    positions = [i for i,t in enumerate(text_codes) if t >= 5]
    words = [a for a in alignments if a[2] == 'SPEAKER_MAIN']
    slots, cursor = [], 0
    for word, _, _ in words:
        n = len(encode(word.strip()))
        slots.append(positions[cursor:cursor+n]); cursor += n
    groups, start = [], 0
    for i, (word, _, _) in enumerate(words):
        if word.rstrip().endswith(('.', '?', '!')):
            groups.append((start, i+1)); start = i+1
    phrase = spoken_words(value)
    matches = []
    for begin, end in groups:
        flat = spoken_words(' '.join(w[0] for w in words[begin:end]))
        if any(flat[i:i+len(phrase)] == phrase for i in range(len(flat)-len(phrase)+1)):
            matches.append((begin,end))
    if not phrase or len(matches) != 1:
        raise RuntimeError('Private phrase needs one complete independently measured sentence')
    begin, end = matches[0]
    if words[begin][1][0] < ready_s:
        raise RuntimeError('Source result sentence starts before its actual teacher receipt')
    first, last = slots[begin][0], slots[end-1][-1]
    left = first-1
    right = max(last+1, math.ceil(words[end-1][1][1]*frame_rate))+2
    if (left < 0 or right > len(text_codes) or text_codes[left] not in (0,3)
            or any(t == 4 for t in text_codes[left:right])
            or [t for t in text_codes[left:right] if t >= 5] != [text_codes[i]
                for word_slots in slots[begin:end] for i in word_slots]):
        raise RuntimeError('Bounded measured sentence would collide with another word or request')
    return {'source_start_frame': left, 'source_end_frame': right,
        'source_first_word_index': begin, 'source_end_word_index': end,
        'text': ' '.join(w[0] for w in words[begin:end]),
        'source_receipt_s': ready_s, 'native_frame_rate': frame_rate,
        'measured_token_spacing_unchanged': True, 'training_approved': False}


def captured_packets(audio, ledger, rows):
    """Recover packets from hash-verified model.wav; require complete accounting."""
    import numpy as np
    if audio.ndim!=1 or not np.isfinite(audio).all():
        raise RuntimeError('Invalid recorded model PCM')
    packets=[];cursor=0;packet_index=0
    for row in rows:
        count=row['output_samples']
        if not count:
            packets.append(audio[:0]);continue
        if packet_index>=len(ledger):raise RuntimeError('Recorded output packet is absent')
        packet=ledger[packet_index];start=packet['playback_start_sample'];end=start+count
        if (packet['packet']!=packet_index or packet['samples']!=count
                or packet['available_s']!=row['available_s'] or start<cursor or end>len(audio)
                or packet['inserted_gap_samples']!=start-cursor
                or np.count_nonzero(audio[cursor:start])):
            raise RuntimeError('Output ledger does not reproduce the verified recording')
        packets.append(audio[start:end]);cursor=end;packet_index+=1
    if packet_index!=len(ledger) or cursor!=len(audio):
        raise RuntimeError('Unaccounted model output audio')
    return packets
