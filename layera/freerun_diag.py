"""Free-run diagnostic v2 (2026-09-22): self-history replay ONLY.

The old forced/free split is retired: the diagnostic's forced mode fed
the model's OWN word predictions back (gold word -> gold_id=None ->
self-feed; Codex review F3), so it was never the probe's teacher
forcing, and probe_p1.py already is the true forced reference. This
script is free-mode only: exact training format, the model's own
emissions fed back, one decision per tick — now with the SAME decode
knobs as live (sil-bias, word continuation), so what it measures is
what the harness runs.

Per held-out lesson it reports:
  - per-gold-kind accuracy + top steals (what replaced <sil>, etc.)
  - word-onset hits: of the gold word ticks, how many the model speaks
    on the exact tick / within +-1 s — the number that predicts the
    gate's answered-question and latency bars
  - false speech: gold-<sil> ticks where the model speaks (chatter)
  - timing-token fires at wrong positions (the thief, self-history)
  - word continuation pieces (the [pad w1 w2] contract, exercised)

  python -m layera.freerun_diag --batch runs/synth-big7/audio \
      --ckpt runs/p1-v11/p1_checkpoint.pt --shards runs/shards \
      --classes timed_task,barge_in --limit 4 --sil-bias 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer.vocab import ALL_SPECIAL, SIL, SPEAK  # noqa: E402


def speech_run_onsets(emissions) -> list[int]:
    """Ticks where a maximal gold SPEECH run starts (v2 contract: a run
    is consecutive word/<speak> ticks). These are the onset positions
    the onset metric scores — under per-tick labels, gold word ticks
    alone would be mid-word positions, not onsets."""
    out = []
    prev_speech = False
    for i, e in enumerate(emissions):
        speech = (e == SPEAK) or (e not in ALL_SPECIAL)
        if speech and not prev_speech:
            out.append(i)
        prev_speech = speech
    return out


def word_onset_stats(gold_word_ticks, spoken) -> tuple[int, int, int, int]:
    """(exact, near, false_speech, n_word) — the capacity-test metrics.

    exact: the model's first pick lands on the exact gold word tick.
    near (event-level): ANY speech within +-1 s of the gold tick.
    false_speech: speech at ticks not near any gold word (chatter).
    History: the original "near" counted gold ticks inside their OWN
    window — vacuously true (a fully muted run "hit" 14/14); and a
    2026-09-23 refactor left the print referencing a deleted variable
    (NameError killed job 6816545's evals). Both now guarded by
    evals/harness/test_freerun_metrics.py."""
    near_w = set()
    for w in gold_word_ticks:
        near_w.update(range(max(0, w - 2), w + 13))
    spoken_set = set(spoken)
    exact = sum(1 for w in gold_word_ticks if w in spoken_set)
    near = 0
    for w in gold_word_ticks:
        lo, hi = max(0, w - 2), w + 13
        if any(lo <= s < hi for s in spoken):
            near += 1
    false_speech = sum(1 for i in spoken if i not in near_w)
    return exact, near, false_speech, len(gold_word_ticks)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--classes", default=None)
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--max-ticks", type=int, default=1600)
    ap.add_argument("--sil-bias", type=float, default=0.0,
                    help="subtract from the <sil> logit (decode sweep)")
    ap.add_argument("--max-pieces", type=int, default=8,
                    help="cap on continuation pieces per word event")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated scenario_ids — replay EXACTLY "
                         "these (may include trained lessons; capacity "
                         "test needs to probe the trained conversations)")
    ap.add_argument("--save-picks", default=None,
                    help="directory for per-scenario JSON of the model's "
                         "word events (onset tick + token ids + text) — "
                         "the input to evals/render_model_audio.py, which "
                         "turns a run into listenable audio")
    ap.add_argument("--min-gap-ticks", type=int, default=0,
                    help="STOPGAP decode rule (owner-approved audition "
                         "aid, 2026-09-28): block word-FIRST picks within "
                         "N ticks of the previous word onset (0 = off). "
                         "A word already in flight always completes")
    ap.add_argument("--user-gate", action="store_true",
                    help="STOPGAP decode rule: block word-FIRST picks "
                         "while the in-slot energy (user + background, "
                         "the model's own voice is never in its ears — "
                         "what a live mic sees) says the user is "
                         "speaking, plus a short hold after")
    ap.add_argument("--gate-hold", type=int, default=5,
                    help="ticks to keep blocking after user energy "
                         "drops (with --user-gate; 5 ticks = 400 ms)")
    ap.add_argument("--gate-ckpt", default=None,
                    help="aux activity-gate checkpoint (train_gate.py): "
                         "a word-FIRST pick is accepted only if this "
                         "head's P(speech) >= --gate-threshold. Learned "
                         "quiet discipline in parameters separate from "
                         "the frozen placement model")
    ap.add_argument("--gate-threshold", type=float, default=0.5,
                    help="P(speech) cut for --gate-ckpt word accepts")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layera.prepare import encode_audio, mix_in_slot
    from tokenizer.vocab import register_into, ALL_SPECIAL, SIL

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16",
        device_map="auto", attn_implementation="sdpa",
        experts_implementation="batched_mm").eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    n_old = len(proc.tokenizer)
    register_into(proc.tokenizer)
    model.resize_token_embeddings(len(proc.tokenizer))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean = emb[:n_old].mean(dim=0)
        emb[n_old:] = mean.to(emb.dtype)
        head = model.lm_head.weight
        head[n_old:] = mean.to(head.dtype)
    ck = torch.load(args.ckpt, weights_only=False)
    with torch.no_grad():
        emb[n_old:] = ck["emb_delta_new_rows"].to(emb.device, emb.dtype)
        head[n_old:] = ck["head_delta_new_rows"].to(head.device, head.dtype)
    adapter = None
    if ck.get("adapter_state"):
        ad_dim = int(ck.get("args", {}).get("adapter_dim", 256))
        hid = emb.shape[1]  # embedding width — no config-path guessing
        adapter = torch.nn.Sequential(
            torch.nn.Linear(hid, ad_dim), torch.nn.GELU(),
            torch.nn.Linear(ad_dim, hid))
        adapter.load_state_dict(ck["adapter_state"])
        adapter = adapter.to(model.device).eval()
        print(f"[freerun] audio adapter loaded "
              f"({sum(p.numel() for p in adapter.parameters())} params)",
              flush=True)

    duplex_ids = set(range(n_old, len(proc.tokenizer)))
    name_of = {i: proc.tokenizer.convert_ids_to_tokens(i)
               for i in duplex_ids}
    mask_ids = [i for i in proc.tokenizer.all_special_ids
                if i not in duplex_ids]
    sil_id = next(i for i in duplex_ids if name_of[i] == "<sil>")
    pad_id = proc.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    tok_conv = proc.tokenizer.convert_tokens_to_ids
    emb_layer = model.get_input_embeddings()

    gate = None
    cap_h = {}  # the tick forward's lm_head input when the gate is on
    if args.gate_ckpt:
        from layera.train_gate import GateHead

        g = torch.load(args.gate_ckpt, weights_only=False)
        gate = GateHead(emb_layer.weight.shape[1],
                        int(g["gate_dim"])).to(model.device).eval()
        gate.load_state_dict(g["gate"])
        # capture the hidden state feeding the word head: the tick
        # forward's lm_head input, read immediately after each tick
        # decision (continuation forwards overwrite it, harmlessly)
        model.lm_head.register_forward_pre_hook(
            lambda mod, inp: cap_h.__setitem__("h", inp[0]))
        print(f"[freerun] activity gate loaded from {args.gate_ckpt} "
              f"(base {g.get('base_ckpt', '?')}, threshold "
              f"{args.gate_threshold:g})", flush=True)

    def clean_argmax(logits) -> int:
        for tid in mask_ids:
            logits[tid] = float("-inf")
        if args.sil_bias:
            logits[sil_id] -= args.sil_bias
        return int(logits.argmax())

    # ---- ear token-rate alignment -------------------------------------
    from tokenizer.grid import from_scenario_dir

    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    train_ids = ({Path(s).stem for s in Path(args.shards).glob("*.pt")}
                 if args.shards else set())
    held = []
    if args.scenarios:
        want = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        by_sid = {r["scenario_id"]: r for r in rows}
        missing = [s for s in want if s not in by_sid]
        if missing:
            raise SystemExit(f"[freerun] scenarios not in manifest: {missing}")
        held = [by_sid[s] for s in want]
    else:
        for r in rows:
            sid = r["scenario_id"]
            if sid in train_ids:
                continue
            if args.classes and not sid.startswith(
                    tuple(c.strip() for c in args.classes.split(","))):
                continue
            held.append(r)
            if len(held) >= args.limit:
                break
    print(f"[freerun] {len(held)} held-out scenarios "
          f"(classes={args.classes}; training={len(train_ids)} shards; "
          f"sil_bias={args.sil_bias:g}, max_pieces={args.max_pieces})",
          flush=True)

    sc0 = batch / held[0]["dir"]
    grid0 = from_scenario_dir(sc0)
    pcm0 = mix_in_slot(sc0, grid0.n_ticks)
    inputs = proc(text=["<|audio_pad|>"], audio=[pcm0[:16000 * 4]],
                  return_tensors="pt", padding=True)
    n_win = int((inputs["input_ids"][0] == pad_id).sum())
    print(f"[freerun] ALIGNMENT: ticks/4s = 50, ear tokens/4s = {n_win} "
          f"(sampler assumes equal; ratio {n_win / 50:.2f})", flush=True)

    def kind_of(tok_id: int) -> str:
        if tok_id >= n_old:
            name = name_of[tok_id]
            if name == "<sil>":
                return "sil"
            if name == "<speak>":
                return "speak"
            if name.startswith("<t+"):
                return "timing"
            if name == "<yield>":
                return "yield"
            return "control"
        return "word"

    def run_lesson(feats, n_ticks, blocked=None, gold_onsets=None):
        """Free-run one lesson tick-by-tick, feeding every emission back
        (word continuation included, exactly as live). Returns per-tick
        first picks + continuation piece count + word events
        ([{tick, ids}] — one entry per spoken word, pieces included;
        the save-picks JSON and the audio renderer both need them).
        blocked (optional): per-tick user-speech mask for the stopgap
        decode rules; word-FIRST picks are gated by it and by the
        min-gap, while a word already in flight always completes."""
        picks, pieces = [], 0
        events: list[dict] = []
        cache = None
        last_onset = -10**9
        n_gap_block = n_user_block = n_act_block = 0
        gate_proposals: list[tuple[int, float]] = []
        for i in range(min(n_ticks, args.max_ticks)):
            e_in = emb_layer(torch.tensor([[pad_id]], device=model.device)
                             ).clone()
            e_in[0, 0] = feats[i].to(e_in.dtype)
            with torch.no_grad():
                out = model(inputs_embeds=e_in, past_key_values=cache,
                            use_cache=True)
            cache = out.past_key_values
            tok = clean_argmax(out.logits[0, -1].clone())
            if kind_of(tok) == "word":
                if gate is not None:
                    # learned quiet discipline: the activity gate sits
                    # on the same hidden state the word head just used
                    p = float(torch.sigmoid(
                        gate(cap_h["h"][0, -1].detach().unsqueeze(0))[0]))
                    gate_proposals.append((i, p))
                    if p < args.gate_threshold:
                        tok = sil_id
                        n_act_block += 1
                if i - last_onset < args.min_gap_ticks:
                    tok = sil_id
                    n_gap_block += 1
                elif blocked is not None and blocked[i]:
                    tok = sil_id
                    n_user_block += 1
                else:
                    last_onset = i
            picks.append(tok)
            if kind_of(tok) == "word":
                events.append({"tick": i, "ids": [tok]})
            # continuation: keep emitting pieces of the same word event;
            # the first non-word breaker is discarded, never fed
            while kind_of(tok) == "word" and pieces + 1 < args.max_pieces:
                e_out = emb_layer(torch.tensor([[tok]], device=model.device))
                with torch.no_grad():
                    out = model(inputs_embeds=e_out,
                                past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = clean_argmax(out.logits[0, -1].clone())
                if kind_of(nxt) != "word":
                    break
                pieces += 1
                picks.append(None)  # continuation: no new tick decision
                events[-1]["ids"].append(nxt)
                tok = nxt
        if n_gap_block or n_user_block or n_act_block:
            print(f"[freerun]   decode gates: {n_gap_block} gap-blocks, "
                  f"{n_user_block} user-gate blocks, "
                  f"{n_act_block} activity-gate blocks", flush=True)
        if gate is not None and gold_onsets:
            # WHY the gate passes/blocks: its own P at the gold speech
            # onsets (gate1 lesson: dialog gold onsets scored below
            # chatter states on other lessons -> full mute)
            at = {t: p for t, p in gate_proposals if t in gold_onsets}
            if at:
                print(f"[freerun]   gate P at gold onsets: "
                      + ", ".join(f"{t}:{at[t]:.2f}"
                                  for t in sorted(at)), flush=True)
        return picks, pieces, events

    for r in held:
        sc_dir = batch / r["dir"]
        grid = from_scenario_dir(sc_dir)
        pcm = mix_in_slot(sc_dir, grid.n_ticks)
        feats, counts = encode_audio(model, proc, pcm)
        if adapter is not None:
            # encode_audio returns CPU feats by design (prepare.py line
            # 86); the adapter lives on model.device. F.linear can't
            # cross devices (setitem/copy_ can — why the rows-only arm
            # never tripped this). Move the INPUT to the adapter.
            ad_dev = next(adapter.parameters()).device
            feats = adapter(feats.to(ad_dev, torch.float32)).to(feats.dtype)
        print(f"[freerun] {grid.scenario_id}: {grid.n_ticks} ticks, "
              f"{feats.shape[0]} ear tokens, windows={counts[:3]}...",
              flush=True)
        if feats.shape[0] < grid.n_ticks:
            print(f"[freerun]   WARNING: fewer ear tokens than ticks — "
                  f"training saw time-compressed audio", flush=True)
        blocked = None
        if args.user_gate:
            # per-tick in-slot RMS: user speech vs background. The
            # model's own voice is never in its ears, so this is
            # exactly what a live mic-side VAD would see. Threshold =
            # midpoint of the lesson's quiet and loud levels.
            arr = np.asarray([
                float(np.sqrt(np.mean(
                    pcm[i * 1280:(i + 1) * 1280] ** 2)))
                for i in range(grid.n_ticks)])
            thr = float(0.5 * (np.percentile(arr, 25)
                               + np.percentile(arr, 95)))
            hot = arr > thr
            blocked = np.zeros(grid.n_ticks, dtype=bool)
            last = -10**9
            for i in range(grid.n_ticks):
                if hot[i]:
                    last = i
                blocked[i] = (i - last) <= args.gate_hold
        gold_onsets = set(speech_run_onsets(
            [t.emission for t in grid.ticks]))
        picks, pieces, events = run_lesson(feats, grid.n_ticks, blocked,
                                           gold_onsets)
        if args.save_picks:
            outdir = Path(args.save_picks)
            outdir.mkdir(parents=True, exist_ok=True)
            for ev in events:
                ev["text"] = proc.tokenizer.decode(ev["ids"]).strip()
            (outdir / f"{grid.scenario_id}.json").write_text(
                json.dumps({"scenario_id": grid.scenario_id,
                            "n_ticks": grid.n_ticks, "words": events},
                           indent=1))
        # first pick per tick = the tick's decision
        tick_picks = [p for p in picks if p is not None]

        per = defaultdict(lambda: [0, 0, Counter()])
        gold_speech, spoken, gold_names = [], [], []
        for i in range(len(tick_picks)):
            gold_name = grid.ticks[i].emission
            gold_names.append(gold_name)
            gold = "<word>" if gold_name not in ALL_SPECIAL else \
                ("<sil>" if gold_name == SIL else gold_name)
            pick = tick_picks[i]
            kind = kind_of(pick)
            if kind in ("word", "speak"):
                spoken.append(i)
            e = per[gold]
            e[0] += 1
            gold_is_word = gold == "<word>"
            ok = (kind == "word") if gold_is_word else \
                (pick == tok_conv(gold))
            e[1] += int(ok)
            if not ok:
                e[2][name_of.get(pick, kind)] += 1
            if gold_is_word:
                gold_speech.append(i)

        # v2 metrics: onsets = gold speech-run starts; the dense
        # headline is per-tick speech P/R/F1; chatter counts the
        # model's speech on ticks whose gold is genuinely quiet
        onsets = speech_run_onsets(gold_names)
        exact, near, _, n_onset = word_onset_stats(onsets, spoken)
        gs, sp = set(gold_speech), set(spoken)
        hit = len(gs & sp)
        precision = hit / len(sp) if sp else 0.0
        recall = hit / len(gs) if gs else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        chatter = sum(1 for i in spoken if i not in gs)

        print(f"[freerun]   free (self-history):", flush=True)
        for g in sorted(per):
            n, ok, th = per[g]
            line = f"    gold {g:8s} n={n:4d}  acc {ok / n:5.3f}"
            if th:
                line += "  <- " + ", ".join(
                    f"{k} x{c}" for k, c in th.most_common(3))
            print(line, flush=True)
        print(f"[freerun]   SPEECH: P {precision:.3f}  R {recall:.3f}  "
              f"F1 {f1:.3f}; false speech on {chatter} quiet ticks",
              flush=True)
        print(f"[freerun]   WORD ONSETS: {exact}/{n_onset} exact, "
              f"{near}/{n_onset} within +-1s; "
              f"{pieces} continuation pieces", flush=True)
    print("[freerun] done", flush=True)


if __name__ == "__main__":
    main()
