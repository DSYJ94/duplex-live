"""Separate native foreground actions from PersonaPlex's literal byte token 4."""


def action_targets(text_codes, timeline):
    """Keep the approved marker clock; distinguish actual cancellation markers."""
    labels = [0] * len(text_codes)
    events = [e for e in timeline["events"] if e["ch"] == "assistant" and e["kind"] == "marker"]
    frames = [i for i, token in enumerate(text_codes) if token == 4]
    if len(events) != len(frames):
        raise ValueError("Native control count differs from approved timeline")
    expected = []
    for event in events:
        tick = round(event["start_ms"] / 80)
        if abs(tick * 80-event["start_ms"]) > .001 or tick not in frames:
            raise ValueError("Native control frame differs from approved marker clock")
        note = event.get("note", "").lower()
        if "cancel" in note:
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
