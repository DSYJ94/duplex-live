"""M2 LIVE LOOP — the simulated-live runner (2026-10-06).

Plain words: this is the dress rehearsal for the live mic demo. User
audio streams in one 80 ms tick at a time (from the lesson's user stem,
standing in for a microphone), the ear encodes each 4 s window onto the
tick grid exactly as training saw (non-overlapping windows, nearest-
time resample), the q2 checkpoint + activity gate decide every tick,
and when the model starts a turn we voice the NEXT scripted assistant
utterance — a whole-utterance clip from the factory render, so the
prosody is natural and the model's contribution is exactly WHEN the
turn happens (the honest claim; word content is the script's, see
EXPERIMENTS 2026-10-05). If the model yields, or the user starts
talking (causal VAD, running floor — never per-lesson percentiles),
playback cuts with a fade.

Design notes:
  - windows are NON-OVERLAPPING 4 s, encoded once per 50 ticks — the
    same structure prepare.encode_audio gave training;
  - utterances play SEQUENTIALLY (the next unspoken one at each model
    turn start): live cannot see the future, so window-matching to gold
    is not available — this is the live-honest mapping;
  - GATE GRACE (the quick-reply knob): in the first --gate-grace ticks
    after user speech ends, the activity gate is bypassed — the timing
    review showed the missed turns are exactly the fast interjections
    right after the user stops.
  - user_blocked comes from the VAD, not from gold labels.

Run (Isambard, GH200):
  python -m m2.live --ckpt ... --batch ... --scenarios dense_dialog-116 \
      --gate-ckpt ... --out runs/live-q2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SR = 16000
GRID_MS = 80
PER_TICK = int(SR * GRID_MS / 1000)  # 1280 samples
WINDOW_TICKS = 50                    # 4 s of ticks per ear window
VAD_HOLD_TICKS = 8                   # 640 ms hangover after user sound


def load_stem_mono(path: Path, n_samples: int | None = None) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    if n_samples is None:
        return x
    buf = np.zeros(n_samples, dtype=np.float32)
    buf[:min(len(buf), len(x))] = x[:len(buf)]
    return buf


class FloorVAD:
    """Causal user-speech detector: a tick is 'user' when its RMS clears
    4x a RUNNING FLOOR (the quietest recent tick) — no percentiles, no
    lookahead (ugate's lesson: per-lesson thresholds saturate on dense
    users). Floor adapts over ~10 s, so a loud room raises it."""

    def __init__(self, floor_ticks: int = 125):
        self.ring: list[float] = []
        self.floor_ticks = floor_ticks
        self.hold = 0

    def tick(self, pcm: np.ndarray) -> bool:
        rms = float(np.sqrt((pcm ** 2).mean())) if len(pcm) else 0.0
        self.ring.append(rms)
        if len(self.ring) > self.floor_ticks:
            self.ring.pop(0)
        floor = max(min(self.ring), 1e-4)
        active = rms > 4.0 * floor and rms > 5e-4
        if active:
            self.hold = VAD_HOLD_TICKS
        elif self.hold > 0:
            self.hold -= 1
            active = True
        return active


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--gate-ckpt", default=None)
    ap.add_argument("--gate-threshold", type=float, default=0.10)
    ap.add_argument("--gate-grace", type=int, default=6,
                    help="ticks after user speech ends where the gate is "
                         "bypassed (quick replies; timing review 2026-10-06)")
    ap.add_argument("--realtime", action="store_true",
                    help="pace to wall-clock 80 ms ticks (demo mode)")
    ap.add_argument("--source", choices=["stem", "stdin"], default="stem",
                    help="stem = the lesson's user wav (simulated); "
                         "stdin = raw int16 16k mono from a real mic "
                         "(the demo pipe: mic | ssh ... | speaker)")
    ap.add_argument("--sink", choices=["files", "stdout"],
                    default="files",
                    help="files = write wavs to --out; stdout = stream "
                         "the model's voice as raw int16 16k mono (the "
                         "return leg of the demo pipe; logs go to "
                         "stderr)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from data.synthetic.quality import checked_rows
    checked_rows(Path(args.batch),
                 [{"scenario_id": s.strip()} for s in args.scenarios.split(",")],
                 role="live demo")

    import torch
    from m2.parity import Decoder, load_model
    from data.synthetic.timeline import Timeline

    model, proc, adapter, n_old = load_model(Path(args.ckpt))

    # serving decode: the activity gate reads the lm_head-input hidden
    # state — same capture hook as parity / freerun_diag
    cap: dict = {}
    model.lm_head.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0]))
    gate = None
    if args.gate_ckpt:
        from layera.train_gate import GateHead
        gk = torch.load(args.gate_ckpt, weights_only=False)
        gate = GateHead(model.get_input_embeddings().weight.shape[1],
                        int(gk["gate_dim"])).to(model.device).eval()
        gate.load_state_dict(gk["gate"])
        for p in gate.parameters():
            p.requires_grad_(False)
        print(f"[live] activity gate loaded (thr {args.gate_threshold}, "
              f"grace {args.gate_grace} ticks)", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for sid in [s.strip() for s in args.scenarios.split(",")]:
        sc = Path(args.batch) / sid
        tl = Timeline.load(sc / "timeline.actual.json")
        n_ticks = round(max(e.actual_end_ms or 0 for e in tl.events)
                        / GRID_MS) + 1
        user_pcm = load_stem_mono(sc / "stem_user.wav", n_ticks * PER_TICK)
        utterances = [e for e in tl.events
                      if e.ch == "assistant" and e.text
                      and e.kind != "marker"]
        clips = []
        speech_idx = 0
        for e in tl.events:
            if e.text and e.kind != "marker":
                if e.ch == "assistant":
                    clips.append(sc / "utterances"
                                 / f"{speech_idx:02d}_assistant.wav")
                speech_idx += 1

        dec = Decoder(model, proc, adapter, max_pieces=8, n_old=n_old,
                      gate=gate, gate_threshold=args.gate_threshold,
                      cap=cap, piece_mode="arrears", pieces_per_tick=4)
        vad = FloorVAD()
        feats, cache = [], None
        window_pcm = np.zeros(0, dtype=np.float32)
        speaking = None      # (utterance idx, position in clip)
        spoken = 0
        rec = {"scenario_id": sid, "turns": [], "yields": 0,
               "ticks": n_ticks}
        out_pcm = np.zeros(0, dtype=np.float32)  # grows per tick
        clip_data = {i: load_stem_mono(p) for i, p in
                     enumerate(clips)}
        emb = model.get_input_embeddings()
        pad_id = proc.tokenizer.convert_tokens_to_ids("<|audio_pad|>")

        # stdin mode: one conversation, unbounded, fed by the mic pipe.
        # Blocking reads pace the loop (a tick happens when 80 ms of
        # sound has arrived — real time by construction); Ctrl-C ends.
        live = args.source == "stdin"
        max_ticks = 10 ** 9 if live else n_ticks
        # stdout sink: dup fd 1 for raw audio, then send ALL prints to
        # stderr — otherwise a log line lands mid-stream in the speaker
        if args.sink == "stdout":
            import os
            fd = os.dup(1)
            sink = os.fdopen(fd, "wb", buffering=0)
            sys.stdout = sys.stderr
        else:
            sink = None

        def read_mic_chunk() -> np.ndarray:
            want = PER_TICK * 2  # int16 mono
            buf = b""
            while len(buf) < want:
                got = sys.stdin.buffer.read(want - len(buf))
                if not got:
                    raise EOFError("mic stream ended")
                buf += got
            return np.frombuffer(buf, dtype="<i2").astype(
                np.float32) / 32768.0

        t0 = time.monotonic()
        for t in range(max_ticks):
            if args.realtime:
                dl = t0 + (t + 1) * GRID_MS / 1000 - time.monotonic()
                if dl > 0:
                    time.sleep(dl)
            if live:
                try:
                    chunk = read_mic_chunk()
                except EOFError:
                    break
            else:
                chunk = user_pcm[t * PER_TICK:(t + 1) * PER_TICK]
            window_pcm = np.concatenate([window_pcm, chunk])[-WINDOW_TICKS * PER_TICK:]
            if (t + 1) % WINDOW_TICKS == 0 and len(window_pcm) == WINDOW_TICKS * PER_TICK:
                from layera.prepare import encode_audio
                f, _ = encode_audio(model, proc, window_pcm.copy())
                feats = list(f)
            feat = (feats.pop(0).to(model.device) if feats
                    else emb(torch.tensor([[pad_id]],
                                          device=model.device))[0, 0])
            user_on = vad.tick(chunk)
            # GATE GRACE: in the first --gate-grace ticks of the VAD
            # hangover (user just stopped), bypass the activity gate —
            # the missed turns in the timing review were exactly these
            # fast interjections
            grace = (gate is not None and not user_on
                     and 0 < VAD_HOLD_TICKS - vad.hold <= args.gate_grace)
            saved_thr = dec.gate_threshold
            if grace:
                dec.gate_threshold = 0.0
            pick, pieces, cache = dec.tick(feat, cache,
                                           user_blocked=user_on)
            dec.gate_threshold = saved_thr
            name = dec.name_of.get(pick, "")
            # turn start: word-class pick while not speaking
            if pick < dec.n_old and not speaking and spoken < len(clips):
                i = spoken
                speaking = {"idx": i, "pos": 0}
                rec["turns"].append({"tick": t, "t_ms": t * GRID_MS,
                                     "utterance": i,
                                     "text": utterances[i].text[:60]})
                print(f"[live] {sid} t={t*GRID_MS/1000:6.1f}s TURN "
                      f"#{i}: {utterances[i].text[:40]!r}", flush=True)
            # yield: cut playback (barge-in behaviour)
            if name == "<yield>" and speaking:
                rec["yields"] += 1
                print(f"[live] {sid} t={t*GRID_MS/1000:6.1f}s YIELD "
                      f"(cut utterance #{speaking['idx']})", flush=True)
                speaking = None
                spoken += 1  # the cut utterance counts as consumed
            # advance playback: this tick's model-voice samples -> sink
            tick_out = np.zeros(PER_TICK, dtype=np.float32)
            if speaking:
                cd = clip_data[speaking["idx"]]
                n = min(PER_TICK, len(cd) - speaking["pos"])
                if n <= 0:
                    speaking = None
                    spoken += 1
                else:
                    tick_out[:n] += cd[speaking["pos"]:speaking["pos"] + n]
                    speaking["pos"] += n
            if user_on and speaking and speaking["pos"] > int(0.3 * SR):
                # user barges in: cut (clean yield) from next tick
                rec["yields"] += 1
                speaking = None
                spoken += 1
            if sink is not None:
                # the return leg: raw int16 to the pipe (the speaker
                # side plays it); stdout is AUDIO now — all logs must
                # go to stderr
                sink.write((tick_out * 32767.0).astype("<i2").tobytes())
                sink.flush()
            else:
                out_pcm = np.concatenate([out_pcm, tick_out])
        if sink is None:
            peak = float(np.max(np.abs(out_pcm))) or 1.0
            if peak > 0.95:
                out_pcm *= 0.95 / peak
            import soundfile as sf
            sf.write(out_dir / f"{sid}_model.wav", out_pcm, SR)
            mixed = out_pcm + user_pcm[:len(out_pcm)]
            peak = float(np.max(np.abs(mixed))) or 1.0
            if peak > 0.95:
                mixed *= 0.95 / peak
            sf.write(out_dir / f"{sid}_mixed.wav", mixed, SR)
        rec["n_turns"] = len(rec["turns"])
        rec["n_gold"] = len(clips)
        summary.append(rec)
        print(f"[live] {sid}: {len(rec['turns'])}/{len(clips)} turns "
              f"voiced, {rec['yields']} yields", flush=True)
        if live:
            break  # one conversation per stdin session
    (out_dir / "live_report.json").write_text(
        json.dumps(summary, indent=1))
    print("[live] done — listen: rsync + python -m viz.listen")


if __name__ == "__main__":
    main()
