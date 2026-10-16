"""Derive explicit native pre-lead RET labels without rewriting source clocks.

The published Context preparation places RET one frame before an assistant
lead. Existing worker dispatch timestamps are evidence, not that policy label.
Only approved TRAIN inputs may call this conversion; no spoken word moves.
"""
import math
from evals.recovery_context import causal_dialogue
from evals.recovery_worker import project_identified_by_user


def native_prelead(text_codes, side, encode, frame_rate=12.5):
    source = list(text_codes)
    if frame_rate != 12.5 or 0 in source:
        raise RuntimeError('Native request targets require explicit Context PAD3 conversion')
    meta = side['timeline']['meta']
    if meta.get('split') != 'train':
        raise RuntimeError('Derived request targets are TRAIN-only')
    from evals.recovery_encode import marker_frame
    events = [e for e in side['timeline']['events'] if e['kind'] == 'marker']
    markers = sorted(marker_frame(e, frame_rate) for e in events)
    if markers != [f for f, token in enumerate(source) if token == 4]:
        raise RuntimeError('Source request controls differ from approved events')
    kind = meta['kind']
    expected = 0 if kind == 'simple' else 2 if kind == 'cancel' else 1
    if kind not in ('simple', 'private', 'clarify', 'cancel') or len(markers) != expected:
        raise RuntimeError('Ambiguous source request operations')
    positions = [f for f, token in enumerate(source) if token >= 5]
    firsts, cursor = [], 0
    for word, interval, speaker in side['alignments']:
        if speaker != 'SPEAKER_MAIN':
            continue
        tokens = list(encode(word.strip()))
        slots = positions[cursor:cursor + len(tokens)]
        if not tokens or len(slots) != len(tokens) or [source[f] for f in slots] != tokens:
            raise RuntimeError('Cannot recover every unchanged measured word')
        firsts.append((slots[0], float(interval[0])))
        cursor += len(tokens)
    if cursor != len(positions):
        raise RuntimeError('Unaccounted spoken targets')
    derived = [3 if token == 4 else token for token in source]
    records = []
    for index, old in enumerate(markers):
        limit = markers[index + 1] if index + 1 < len(markers) else len(source)
        word = next(((f, onset) for f, onset in firsts if old <= f < limit), None)
        if word is None:
            raise RuntimeError('Request has no following measured assistant lead')
        first, onset = word
        target = first - 1
        if target < 0 or derived[target] != 3:
            raise RuntimeError('Native pre-lead RET would overwrite a word or another request')
        time_s = target / frame_rate
        context = causal_dialogue(side, time_s, [], time_s)
        operation = 'cancel' if kind == 'cancel' and index == 1 else 'lookup'
        if operation == 'lookup':
            if not project_identified_by_user(context, meta['project']):
                raise RuntimeError('Derived RET predates user project identification')
            arrivals = side['reference_arrivals']
            if arrivals and any(not math.isfinite(a['ready_at_s']) or time_s >= a['ready_at_s']
                                for a in arrivals):
                raise RuntimeError('Derived request would follow an already supplied result')
        else:
            cancels = [e for e in side['timeline']['events'] if e['ch'] == 'user'
                and e['kind'] == 'utterance' and 'cancel' in e['text'].casefold()]
            heard_cancel = any(e['role'] == 'user' and 'cancel' in e['text'].casefold() for e in context)
            if len(cancels) != 1 or cancels[0]['actual_end_ms'] / 1000 > time_s or not heard_cancel:
                raise RuntimeError('Derived cancellation predates its complete user instruction')
        derived[target] = 4
        records.append({'operation': operation, 'source_dispatch_frame': old,
            'derived_request_frame': target, 'derived_request_s': time_s,
            'measured_lead_first_token_frame': first, 'measured_lead_onset_s': onset,
            'source_dispatch_clock_changed': False, 'actual_worker_dispatch_remeasured': False,
            'label_provenance': 'derived native pre-lead policy, not an actual worker timestamp'})
    if [t for t in derived if t >= 5] != [t for t in source if t >= 5]:
        raise RuntimeError('Request conversion changed measured words')
    return derived, records


def native_prelead_epad(text_codes, side, encode, frame_rate=12.5):
    """Retain observed native EPAD0 -> RET4 -> first-word ordering.

    Ordinary end-padding cues stay in place. Where RET occupies the existing
    lead's EPAD slot, move that cue one earlier empty frame; never move a word.
    """
    source = list(text_codes)
    derived, records = native_prelead([3 if t == 0 else t for t in source], side, encode, frame_rate)
    for frame, token in enumerate(source):
        if token == 0 and derived[frame] != 4:
            derived[frame] = 0
    for record in records:
        ret = record['derived_request_frame']
        epad = ret - 1
        if (epad < 0 or derived[epad] not in (0, 3)
                or not (source[ret] == 0 or (source[ret] == 4 and source[epad] == 0))):
            raise RuntimeError('Native EPAD/RET order cannot preserve the measured lead without collision')
        derived[epad] = 0
        original_epad = ret if source[ret] == 0 else epad
        record.update(source_end_padding_frame=original_epad, derived_end_padding_frame=epad,
            end_padding_shift_frames=epad - original_epad,
            label_provenance='derived observed EPAD0 RET4 lead order; actual dispatch clock unchanged')
    if derived.count(0) != source.count(0):
        raise RuntimeError('Request conversion lost an existing end-padding cue')
    return derived, records
