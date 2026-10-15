"""Separate native foreground actions from PersonaPlex's literal byte token 4."""
ACTION_TARGET_SCHEMA = "dl-native-action-targets-v2"


def action_targets(text_codes, timeline):
    """Keep the approved marker clock; distinguish actual cancellation markers."""
    labels = [0] * len(text_codes)
    events = sorted([e for e in timeline["events"] if e["ch"] == "assistant" and e["kind"] == "marker"],
                    key=lambda e:e["start_ms"])
    meta = timeline.get("meta", {})
    legacy_cancel = (meta.get("kind") == "cancel" and meta.get("teacher_audio") is True
        and all(e.get("note") == "delegate actual teacher worker" for e in events))
    if legacy_cancel:
        # Historical approved conversion retained generic notes for both calls.
        # Its verified cancel scene has one lookup then the actual cancel call;
        # immutable source/receipt validation occurs before this derivation.
        arrivals = meta.get("reference_arrivals", [])
        if len(events) != 2 or len(arrivals) != 1 or arrivals[0]["ready_at_s"] < events[-1]["start_ms"]/1000:
            raise ValueError("Ambiguous legacy teacher cancellation chronology")
    frames = [i for i, token in enumerate(text_codes) if token == 4]
    if len(events) != len(frames):
        raise ValueError("Native control count differs from approved timeline")
    expected = []
    for event in events:
        tick = round(event["start_ms"] / 80)
        if abs(tick * 80-event["start_ms"]) > .001 or tick not in frames:
            raise ValueError("Native control frame differs from approved marker clock")
        note = event.get("note", "").lower()
        if "cancel" in note or (legacy_cancel and event is events[-1]):
            label = 2
        elif "delegat" in note:
            label = 1
        else:
            raise ValueError("Unspecified foreground action")
        labels[tick] = label
        expected.append(tick)
    if set(expected) != set(frames):
        raise ValueError("Duplicate/missing approved foreground marker")
    return labels
