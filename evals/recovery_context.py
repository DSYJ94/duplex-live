"""Build chronological diagnostic context from already available words only."""


def causal_dialogue(sidecar,input_until_s,packets,output_until_s):
    context = []
    for index,event in enumerate(sidecar["timeline"]["events"]):
        if event["ch"] != "user" or event["kind"] == "marker": continue
        a,b = event["actual_start_ms"]/1000,event["actual_end_ms"]/1000
        words = [w for w in sidecar["user_words"] if w["start_s"] >= a-.001 and w["end_s"] <= b+.001 and w["end_s"] <= input_until_s]
        if words:
            context.append({"role":"user","utterance_id":f"user-{index}",
                "text":" ".join(w["text"] for w in words),"start_s":words[0]["start_s"],
                "end_s":words[-1]["end_s"],"utterance_complete":input_until_s >= b,
                "source":"measured completed words: diagnostic oracle"})
    group = None
    for packet in packets:
        text,time = packet["text"],packet["packet_available_s"]
        if not text or time > output_until_s: continue
        if group is None or time-group["end_s"] > .7:
            group = {"role":"assistant","text":"","start_s":time,"end_s":time,
                "source":"generated native text tokens; not a tool receipt"}
            context.append(group)
        group["text"] += text
        group["end_s"] = time
    return sorted(context,key=lambda entry:entry["start_s"])
