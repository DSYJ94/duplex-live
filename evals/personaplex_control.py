"""Causal client-state feedback, separate from action labels and spoken words."""
import math

CONTROL_STATES = ("prefix", "ready", "worker_pending", "result_available", "cancel_pending", "cancelled")


def feedback_schedule(actions, arrivals, kind, frame_rate=12.5):
    """A chosen action affects the following frame, never its own prediction."""
    if kind not in ("simple", "private", "clarify", "cancel"):
        raise ValueError("Unknown approved control scene kind")
    selected = [(i, a) for i,a in enumerate(actions) if a]
    expected = [] if kind == "simple" else [1, 2] if kind == "cancel" else [1]
    if [a for _,a in selected] != expected or len(arrivals) != int(bool(expected)):
        raise ValueError("Control feedback lacks the approved bounded action/receipt sequence")
    changes = [(i+1, 0, 2 if a == 1 else 4) for i,a in selected]
    for arrival in arrivals:
        ready = arrival["ready_at_s"]
        if not math.isfinite(ready) or ready < 0:
            raise ValueError("Invalid actual result clock")
        frame = int(ready*frame_rate)
        while frame/frame_rate < ready:
            frame += 1
        # A very fast receipt can be held for the next neural decision tick;
        # it cannot leak the expert action into that action's input state.
        frame = max(frame, selected[-1][0]+1)
        if frame >= len(actions):
            raise ValueError("Actual result falls outside the native conversation")
        changes.append((frame, 1, 5 if kind == "cancel" else 3))
    output, state, cursor = [], 1, 0
    changes.sort()
    for frame in range(len(actions)):
        while cursor < len(changes) and changes[cursor][0] <= frame:
            state = changes[cursor][2]
            cursor += 1
        output.append(state)
    return output


def live_feedback_state(jobs, cancelled, events):
    """Use only dispatched jobs and actually observed results, with no oracle."""
    active = [j for identity,j in jobs.items() if not j["observed"] and identity not in cancelled]
    if any(j["action"] == 2 for j in active):
        return 4
    if active:
        return 2
    for event in reversed(events):
        if event["event"] == "worker_failed":
            return 1
        if event["event"] == "worker_completed":
            receipts = event["result"]["receipts"]
            return 5 if any(r.get("tool") == "cancel_lookup" and r.get("cancelled") for r in receipts) else 3
    return 1
