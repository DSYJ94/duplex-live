"""Fresh, explicitly synthetic record-work lessons; audio is unverified until judged.

ADR-0003 Amendment A permits closed-content with provenance. Generation is
bounded to forty scenes. No existing evaluation recording is used for training.
"""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

MODEL = "gpt-4o-mini-tts-2025-12-15"
TRAIN_QA = [
    ("Why do I see lightning before hearing thunder?", "Light travels much faster than sound."),
    ("What does evaporation mean?", "Evaporation is liquid changing into gas."),
    ("Why does day turn into night?", "Earth rotates, turning our part of the planet away from the sun."),
    ("How many sides does a triangle have?", "A triangle has three sides."),
    ("Why does ice float in water?", "Ice is less dense than liquid water."),
    ("How is sound produced?", "Sound comes from vibrations travelling through a medium such as air."),
    ("Why do plants need light?", "Plants use light to make sugars through photosynthesis."),
    ("Why does metal feel cold?", "Metal carries heat away from your hand quickly."),
]
QA = {"train": TRAIN_QA,
    "validation": [("Why does salt dissolve in water?", "Water molecules surround and separate the salt ions."),
                   ("Why does a pencil leave marks on paper?", "Tiny pieces of graphite rub off onto the paper.")],
    "test": [("Why does blowing on coffee cool it?", "Moving air increases heat loss and evaporation."),
             ("Why does wool keep you warm?", "Wool traps air that slows heat loss.")]}


def specifications():
    rng = random.Random(14102026)
    phrases = rng.sample([f"{colour} {shape}" for colour in
        ("red", "blue", "green", "yellow", "purple", "orange", "silver", "gold", "white", "black", "pink", "brown")
        for shape in ("circle", "square", "triangle", "star", "diamond", "oval", "heart", "cross")], 32)
    scenes, phrase_index = [], 0
    for split, private, simple, clarify, cancel, voice in (
        ("train", 8, 8, 4, 4, None), ("validation", 4, 2, 1, 1, "coral"), ("test", 4, 2, 1, 1, "verse")):
        kinds = ["private"]*private + ["simple"]*simple + ["clarify"]*clarify + ["cancel"]*cancel
        for index, kind in enumerate(kinds):
            family = ("badge" if index % 2 == 0 else "locker") if split == "train" else ("room" if split == "validation" else "shipment")
            field = {"badge": "badge phrase", "locker": "locker label", "room": "room marker", "shipment": "shipment label"}[family]
            forest = ("Birch", "Cedar", "Maple", "Elm") if split == "train" else (("Willow", "Alder") if split == "validation" else ("Ash", "Fir"))
            project = f"{forest[index % len(forest)]} {('North','South','East','West')[index // len(forest) % 4]}"
            phrase = None if kind == "simple" else phrases[phrase_index]
            if phrase is not None:
                phrase_index += 1
            qa = QA[split][(index-private) % len(QA[split])]
            if kind == "simple":
                utterances = [("user", qa[0]), ("assistant", qa[1])]
            elif kind == "clarify":
                utterances = [("user", f"What's the {field}?"), ("assistant", "Which project do you mean?"),
                    ("user", f"Project {project}."), ("assistant", "I'll check the record."),
                    ("user", "While you're checking, what does evaporation mean?"),
                    ("assistant", "Evaporation is liquid changing into gas."),
                    ("assistant", f"The {field} is {phrase}.")]
            else:
                utterances = [("user", f"Please check the {field} for project {project}."),
                    ("assistant", "I'll check the record."),
                    ("user", "While you're checking, why do I see lightning before hearing thunder?"),
                    ("assistant", "Light travels much faster than sound.")]
                if kind == "cancel":
                    utterances += [("user", "Cancel that lookup."), ("assistant", "The lookup is cancelled.")]
                else:
                    utterances += [("assistant", f"The {field} is {phrase}.")]
            scenes.append({"scenario_id": f"recovery-{split}-{index:03d}", "split": split, "kind": kind,
                "script_family": f"{split}-general" if kind == "simple" else family,
                "project": project, "field": field, "phrase": phrase, "user_voice": voice or ("alloy" if index % 2 == 0 else "echo"),
                "assistant_voice": "marin", "nominal_worker_delay_s": (0.5, 2., 5., 15.)[index % 4],
                "utterances": utterances})
    if len(scenes) != 40 or phrase_index != 28:
        raise RuntimeError("Curriculum cardinality changed")
    for left in ("train", "validation", "test"):
        for right in ("train", "validation", "test"):
            if left < right:
                for key in ("script_family", "user_voice"):
                    if {s[key] for s in scenes if s["split"]==left} & {s[key] for s in scenes if s["split"]==right}:
                        raise RuntimeError("Family/user-voice split leakage")
    return scenes


def synthesize(text, voice, path, *, instructions=None):
    import requests
    import soundfile as sf
    key = os.environ["OPENAI_API_KEY"]
    instructions = instructions or "Speak naturally in clear conversational English. Read exactly the supplied words, without additions."
    for attempt, backoff in enumerate((0, 1, 5)):
        if backoff:
            time.sleep(backoff)
        try:
            response = requests.post("https://api.openai.com/v1/audio/speech",
                headers={"Authorization": "Bearer " + key}, timeout=90,
                json={"model": MODEL, "input": text, "voice": voice, "response_format": "wav",
                      "instructions": instructions})
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            continue
        if response.status_code == 200:
            path.write_bytes(response.content)
            pcm, rate = sf.read(path)
            if pcm.ndim != 1 or len(pcm) < rate*.1 or len(pcm) > rate*30:
                raise RuntimeError("Invalid generated clip dimensions/duration")
            receipt = {"request_id": response.headers.get("x-request-id"), "source_model": MODEL,
                "voice": voice, "sha256": hashlib.sha256(response.content).hexdigest(),
                "attempt": attempt, "sample_rate": rate, "seconds": len(pcm)/rate,
                "input_text": text, "instructions": instructions}
            path.with_suffix(".receipt.json").write_text(json.dumps(receipt)+"\n")
            return receipt
        if response.status_code not in (429, 500, 502, 503):
            raise RuntimeError(f"Speech API rejected request: HTTP {response.status_code}")
    raise RuntimeError("Speech API retries exhausted")


def assemble(scene, out, receipts):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from data.synthetic.timeline import Event, Timeline
    folder = out / scene["scenario_id"]
    audio, events, end = [], [], .8
    trigger, ready, arrivals = None, None, []
    lookup_user = 2 if scene["kind"] == "clarify" else 0
    for index, (role, text) in enumerate(scene["utterances"]):
        path = folder / "utterances" / f"{index:02d}_{role}.wav"
        pcm, rate = sf.read(path, dtype="float32")
        divisor = math.gcd(rate, 24000)
        if rate != 24000:
            pcm = resample_poly(pcm, 24000//divisor, rate//divisor).astype(np.float32)
            sf.write(path, pcm, 24000)
        gap = float(scene.get('gaps_before_s', {}).get(str(index), .56))
        if not math.isfinite(gap) or not 0 <= gap <= 10:
            raise ValueError('Declared inter-utterance gap must be bounded')
        start = .8 if index == 0 else end + gap
        if scene["kind"] not in ("simple", "cancel") and index == len(scene["utterances"])-1:
            start = max(start, ready + .4)
        start = math.ceil(start/.08)*.08
        end = start + len(pcm)/24000
        events.append(Event(role, round(start*1000), text, actual_start_ms=round(start*1000),
            actual_end_ms=round(end*1000), wav=f"{scene['scenario_id']}/utterances/{index:02d}_{role}.wav"))
        audio.append((role, round(start*24000), pcm))
        if scene["kind"] != "simple" and index == lookup_user:
            trigger = math.ceil((end+.24)/.08)*.08
            ready = trigger + scene["nominal_worker_delay_s"]
    if scene["kind"] == "cancel":
        cancellation = math.ceil((events[-2].actual_end_ms/1000+.24)/.08)*.08
        ready = max(ready, cancellation + 2.)
        arrivals = [{"ready_at_s": cancellation+.08, "text": "The lookup is cancelled. No private result will be delivered.", "task_id": scene["scenario_id"]+"-cancel"}]
        events[-1].start_ms = events[-1].actual_start_ms = round(math.ceil((cancellation+.48)/.08)*.08*1000)
        events[-1].actual_end_ms = events[-1].actual_start_ms + round(len(audio[-1][2])/24)
        audio[-1] = (audio[-1][0], round(events[-1].actual_start_ms*24), audio[-1][2])
        events.append(Event("assistant", round(cancellation*1000), kind="marker", note="delegate cancellation"))
        end = max(end, ready + 2., audio[-1][1]/24000 + len(audio[-1][2])/24000)
    elif scene["kind"] != "simple":
        arrivals = [{"ready_at_s": ready, "text": f"The checked {scene['field']} for project {scene['project']} is {scene['phrase']}.", "task_id": scene["scenario_id"]}]
    if trigger is not None:
        events.append(Event("assistant", round(trigger*1000), kind="marker", note="delegate lookup"))
    frames = math.ceil((end+.8)/.08)*1920
    stems = {role: np.zeros(frames, dtype=np.float32) for role in ("assistant", "user")}
    for role, offset, pcm in audio:
        stems[role][offset:offset+len(pcm)] += pcm
    for role, pcm in stems.items():
        if not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1:
            raise RuntimeError("Invalid or clipped synthesized stems")
        sf.write(folder / f"stem_{role}.wav", pcm, 24000)
    sf.write(folder / "mixed.wav", stems["assistant"]+stems["user"], 24000)
    meta = {k:v for k,v in scene.items() if k != "utterances"}
    meta.update(synthetic=True, provenance_class="closed-content", source_model=MODEL,
        created_date=scene.get('created_date', "2026-10-14"), verified=False, voices={"assistant":"marin","user":scene["user_voice"]},
        reference_arrivals=arrivals, lookup_trigger_s=trigger, planned_lookup_ready_s=ready,
        timing_scope="synthetic supervision; decoded clip lengths plus a 240ms input margin, not microphone measurements",
        clip_receipts=receipts)
    timeline = Timeline(scene["scenario_id"], "recovery_delegation", events, meta)
    timeline.save(folder / "timeline.actual.json")
    return {"scenario_id": scene["scenario_id"], "scenario_class": "recovery_delegation", "dir": scene["scenario_id"],
        "duration_ms": round(frames/24), "split": scene["split"], "script_family": scene["script_family"],
        "voices": meta["voices"], "engine": MODEL, "verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--specifications', type=Path,
        help='Explicit sealed forty-scene curriculum; historical defaults stay unchanged')
    args = parser.parse_args()
    scenes = json.loads(args.specifications.read_text()) if args.specifications else specifications()
    if args.specifications:
        if len(scenes) != 40 or len({s['scenario_id'] for s in scenes}) != 40:
            raise ValueError('Explicit curriculum requires exactly forty distinct scenes')
        for scene in scenes:
            if (not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', scene['scenario_id'])
                    or scene['split'] not in ('train', 'validation', 'test')
                    or scene['kind'] not in ('private', 'simple', 'clarify', 'cancel')
                    or not scene['utterances'] or scene['utterances'][0][0] != 'user'
                    or scene['utterances'][-1][0] != 'assistant'
                    or any(role not in ('user', 'assistant') or not text.strip() for role, text in scene['utterances'])):
                raise ValueError('Invalid explicitly declared source scene')
        for key in ('script_family', 'user_voice'):
            groups = [{s[key] for s in scenes if s['split'] == split} for split in ('train', 'validation', 'test')]
            if any(a & b for i, a in enumerate(groups) for b in groups[i + 1:]):
                raise ValueError('Explicit source families/voices overlap sealed folds')
    if args.resume:
        if (args.out / "quality.jsonl").exists():
            raise RuntimeError("Do not mutate a corpus after quality stamping")
        if json.loads((args.out / "specifications.json").read_text()) != json.loads(json.dumps(scenes)):
            raise RuntimeError("Cannot resume a changed split/specification")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "specifications.json").write_text(json.dumps(scenes, indent=2)+"\n")
    work = []
    for scene in scenes:
        directory = args.out / scene["scenario_id"] / "utterances"
        directory.mkdir(parents=True, exist_ok=args.resume)
        for index,(role,text) in enumerate(scene["utterances"]):
            work.append((scene, index, role, text, directory / f"{index:02d}_{role}.wav"))
    receipts = {s["scenario_id"]: {} for s in scenes}
    if len(work) > 200 or sum(len(w[3]) for w in work) > 10000:
        raise RuntimeError("Curriculum exceeds bounded generation budget")
    remaining = []
    if args.resume:
        import soundfile as sf
        import numpy as np
        for scene,index,role,text,path in work:
            if not path.exists():
                remaining.append((scene,index,role,text,path))
                continue
            pcm,rate = sf.read(path)
            if pcm.ndim != 1 or not np.isfinite(pcm).all() or not .1 <= len(pcm)/rate <= 30:
                raise RuntimeError("Resume encountered an invalid saved clip")
            sidecar = path.with_suffix(".receipt.json")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if sidecar.exists():
                receipt = json.loads(sidecar.read_text())
                if receipt["sha256"] != digest or receipt["voice"] != scene[role+"_voice"] or receipt["source_model"] != MODEL:
                    raise RuntimeError("Saved clip provenance changed")
            else:
                receipt = {"source_model":MODEL,"voice":scene[role+"_voice"],"sha256":digest,
                    "request_id":None,"sample_rate":rate,"seconds":len(pcm)/rate,
                    "receipt_recovered_from": "interrupted pinned generation; request ID was not persisted"}
                sidecar.write_text(json.dumps(receipt)+"\n")
            receipts[scene["scenario_id"]][index] = receipt
    else:
        remaining = list(work)
    print(json.dumps({"reused_unverified_clips":len(work)-len(remaining),"remaining_requests":len(remaining)}),flush=True)
    # Verify model and voices before issuing the first batch.
    for scene,index,role,text,path in remaining[:2]:
        receipts[scene["scenario_id"]][index] = synthesize(text,scene[role+"_voice"],path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(synthesize, text, scene[role+"_voice"], path): (scene,index) for scene,index,role,text,path in remaining[2:]}
        for future in concurrent.futures.as_completed(futures):
            scene,index = futures[future]
            try:
                receipts[scene["scenario_id"]][index] = future.result()
            except Exception:
                for pending in futures:
                    pending.cancel()
                raise
            print(json.dumps({"rendered": scene["scenario_id"], "clip": index}), flush=True)
    rows = [assemble(scene,args.out,receipts[scene["scenario_id"]]) for scene in scenes]
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (args.out / "generation.json").write_text(json.dumps({"verified": False, "scenes": len(rows), "api_calls":len(work),
        "audio_seconds":sum(r["duration_ms"] for r in rows)/1000, "source_model": MODEL,
        "assistant_voice_shared_across_splits": True, "family_and_user_voice_disjoint":True,
        "quality_gate": "ASR per-clip and measured word timing required before training"},indent=2)+"\n")


if __name__ == "__main__":
    main()
