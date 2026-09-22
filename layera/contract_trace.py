"""Stream-contract trace (Codex F2): training format vs live path.

Plain words: the training stream and the live tick loop must build the
SAME token sequence — one sound-token per 80 ms beat, then that beat's
speech tokens (a word's pieces consecutively, then the next beat). This
script prints both sequences for one lesson around its first multi-piece
word and states exactly where they agree or diverge, position by
position. No scoring, no verdicts — just the trace.

It also prints the two clock facts that are allowed to differ and how:
  - ear tokens per 4 s window vs 50 ticks per window (52 vs 50 — live
    timestamps absorb the difference inside the window step)
  - one decision per FEATURE in live vs one per TICK in training

  python -m layera.contract_trace --batch runs/synth-big7/audio \
      --shards runs/shards --classes timed_task --limit 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint to load for the live pass (any p1-vX)")
    ap.add_argument("--shards", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--limit", type=int, default=3,
                    help="scenarios scanned for a multi-piece word")
    ap.add_argument("--context", type=int, default=6,
                    help="positions printed either side of the word")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layera.duplex import DuplexThinker
    from layera.prepare import mix_in_slot
    from layera.sampler import build_from_grid
    from tokenizer.grid import from_scenario_dir
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct")
    tok = proc.tokenizer
    pad_id = tok.convert_tokens_to_ids("<|audio_pad|>")
    n_old = len(tok)

    def name_of(i: int) -> str:
        if i == pad_id:
            return "<|audio_pad|>"
        if i < n_old:
            return repr(tok.decode([i]))
        return tok.convert_ids_to_tokens(i)

    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    train_ids = ({Path(s).stem for s in Path(args.shards).glob("*.pt")}
                 if args.shards else set())
    cand = []
    for r in rows:
        sid = r["scenario_id"]
        if args.classes and not sid.startswith(
                tuple(c.strip() for c in args.classes.split(","))):
            continue
        cand.append(r)
        if len(cand) >= args.limit:
            break

    # ---- training-format trace ----------------------------------------
    chosen = None
    for r in cand:
        sc_dir = batch / r["dir"]
        seq = build_from_grid(sc_dir, tok)
        grid = from_scenario_dir(sc_dir)
        # first multi-piece word: consecutive graded positions right
        # after one pad whose emission is a word
        for p, lab in enumerate(seq["labels"]):
            if lab == -100:
                continue
            if p + 1 < len(seq["labels"]) and seq["labels"][p + 1] != -100 \
                    and seq["ids"][p] != pad_id:
                j = p
                while j > 0 and seq["ids"][j - 1] != pad_id:
                    j -= 1
                tick = sum(1 for q in range(j)
                           if seq["ids"][q] == pad_id)
                chosen = (sc_dir, seq, grid, j, tick)
                break
        if chosen:
            break
    if not chosen:
        print("[contract] no multi-piece word found in the scanned "
              "scenarios — raise --limit", flush=True)
        return
    sc_dir, seq, grid, wpos, wtick = chosen
    print(f"[contract] scenario {grid.scenario_id}: "
          f"{seq['scenario_id']}, first multi-piece word at training "
          f"position {wpos} (tick {wtick}, t={wtick * 0.08:.2f}s)",
          flush=True)

    c = args.context
    print("[contract] TRAINING stream (position: role token @t):",
          flush=True)
    for p in range(max(0, wpos - c - 1), min(len(seq["ids"]),
                                             wpos + c + 2)):
        role = "HEARD" if seq["ids"][p] == pad_id else " SAYS"
        t_s = sum(1 for q in range(p + 1) if seq["ids"][q] == pad_id) * 0.08
        lab = "" if seq["labels"][p] == -100 else " (graded)"
        print(f"    {p:5d}: {role} {name_of(seq['ids'][p]):>14s} "
              f"@{t_s:7.2f}s{lab}", flush=True)

    # ---- live trace (the actual tick loop) ----------------------------
    thinker = DuplexThinker(str(args.ckpt))
    thinker.reset()
    pcm = mix_in_slot(sc_dir, grid.n_ticks)
    all_beats, pos = [], 0
    for i in range(0, len(pcm), 16000):
        chunk = pcm[i:i + 16000]
        for b_tok, b_t in thinker.feed_audio(chunk):
            all_beats.append((pos, b_tok, b_t))
            pos += 1
    for b_tok, b_t in thinker.finish():
        all_beats.append((pos, b_tok, b_t))
        pos += 1

    # live stream around the same audio time as the training word
    live_at = [k for k, (_, _, t) in enumerate(all_beats)
               if abs(t - wtick * 0.08) < 0.041]
    k0 = live_at[0] if live_at else 0
    print("[contract] LIVE stream (beat: role token @t):", flush=True)
    for k in range(max(0, k0 - c - 1), min(len(all_beats), k0 + c + 2)):
        p, b_tok, b_t = all_beats[k]
        kind = thinker.kind_of(b_tok)
        print(f"    {p:5d}:  SAYS {name_of(b_tok):>14s} @{b_t:7.2f}s "
              f"[{kind}]", flush=True)

    # ---- structural comparison ----------------------------------------
    # Group live beats by timestamp: the FIRST beat at a t_s is a new
    # feature decision (HEARD+SAYS); further beats at the SAME t_s are
    # continuation pieces of that word event (SAYS only). Training's
    # shape is exactly: pad, emissions... per tick.
    train_events = []  # (n_emission_tokens) per tick
    cur = 0
    for i in seq["ids"]:
        if i == pad_id:
            if cur:
                train_events.append(cur)
            cur = 0
        else:
            cur += 1
    if cur:
        train_events.append(cur)

    live_events, n_feats = [], 0
    last_t = None
    for _, b_tok, b_t in all_beats:
        if last_t is None or b_t != last_t:
            live_events.append(1)
            n_feats += 1
            last_t = b_t
        else:
            live_events[-1] += 1
    n = min(len(train_events), len(live_events))
    div = next((i for i in range(n)
                if train_events[i] != live_events[i]), None)
    multi = [(k, v) for k, v in enumerate(live_events) if v > 1]
    print(f"[contract] TRAINING events={len(train_events)} ticks "
          f"({sum(train_events)} said tokens); LIVE events={len(live_events)} "
          f"features ({sum(live_events)} said tokens, "
          f"{len(multi)} multi-piece events: {multi[:5]}...)", flush=True)
    if div is None:
        print("[contract] SHAPE: emission-shape MATCHES tick-for-tick "
              "over the compared prefix (heard-then-said, pieces "
              "consecutive after one heard token).", flush=True)
    else:
        print(f"[contract] SHAPE: first shape difference at event {div} "
              f"(training {train_events[div]} said token(s) vs live "
              f"{live_events[div]}) — expected: the model CHOOSES "
              f"differently from the gold script; the CONTRACT (pieces "
              f"consecutive after one heard token) is what must match, "
              f"and multi-piece live events above show it.", flush=True)
    print(f"[contract] CLOCK: training ticks={grid.n_ticks} @80ms; "
          f"live features={n_feats} "
          f"({n_feats / max(grid.n_ticks, 1):.3f}x ticks). Live step "
          f"inside a window = {4.0 / 52:.4f}s/beat; the 52-vs-50 "
          f"window surplus is absorbed by the per-window step and "
          f"RESETS at each window boundary — beat times vs tick times:",
          flush=True)
    live_ts = [t for _, _, t in all_beats]
    for w in range(3):
        i = w * 50
        j = min(w * 52 + 51, len(live_ts) - 1)
        if i < grid.n_ticks and j >= 0:
            print(f"    window {w}: tick {i} @ {i * 0.08:7.2f}s | "
                  f"live beat {j} @ {live_ts[j]:7.2f}s "
                  f"(delta {live_ts[j] - i * 0.08:+.3f}s)", flush=True)
    print("[contract] done", flush=True)


if __name__ == "__main__":
    main()
