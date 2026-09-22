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
from collections import Counter, defaultdict
from pathlib import Path

import torch


def main() -> None:
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
            if name.startswith("<t+"):
                return "timing"
            if name == "<yield>":
                return "yield"
            return "control"
        return "word"

    def run_lesson(feats, n_ticks):
        """Free-run one lesson tick-by-tick, feeding every emission back
        (word continuation included, exactly as live). Returns per-tick
        first picks + continuation piece count."""
        picks, pieces = [], 0
        cache = None
        for i in range(min(n_ticks, args.max_ticks)):
            e_in = emb_layer(torch.tensor([[pad_id]], device=model.device)
                             ).clone()
            e_in[0, 0] = feats[i].to(e_in.dtype)
            with torch.no_grad():
                out = model(inputs_embeds=e_in, past_key_values=cache,
                            use_cache=True)
            cache = out.past_key_values
            tok = clean_argmax(out.logits[0, -1].clone())
            picks.append(tok)
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
                tok = nxt
        return picks, pieces

    for r in held:
        sc_dir = batch / r["dir"]
        grid = from_scenario_dir(sc_dir)
        pcm = mix_in_slot(sc_dir, grid.n_ticks)
        feats, counts = encode_audio(model, proc, pcm)
        if adapter is not None:
            feats = adapter(feats.to(torch.float32)).to(feats.dtype)
        print(f"[freerun] {grid.scenario_id}: {grid.n_ticks} ticks, "
              f"{feats.shape[0]} ear tokens, windows={counts[:3]}...",
              flush=True)
        if feats.shape[0] < grid.n_ticks:
            print(f"[freerun]   WARNING: fewer ear tokens than ticks — "
                  f"training saw time-compressed audio", flush=True)
        picks, pieces = run_lesson(feats, grid.n_ticks)
        # first pick per tick = the tick's decision
        tick_picks = [p for p in picks if p is not None]

        per = defaultdict(lambda: [0, 0, Counter()])
        gold_word_ticks, spoken = [], []
        for i in range(len(tick_picks)):
            gold_name = grid.ticks[i].emission
            gold = "<word>" if gold_name not in ALL_SPECIAL else \
                ("<sil>" if gold_name == SIL else gold_name)
            pick = tick_picks[i]
            kind = kind_of(pick)
            if kind == "word":
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
                gold_word_ticks.append(i)

        # word-onset hits: exact tick, or any speech within +-1 s
        near_w = set()
        for w in gold_word_ticks:
            near_w.update(range(max(0, w - 2), w + 13))
        spoken_set = set(spoken)
        exact = sum(1 for w in gold_word_ticks if w in spoken_set)
        near = sum(1 for w in gold_word_ticks if w in near_w)
        false_speech = sum(1 for i in spoken if i not in near_w)
        n_word = len(gold_word_ticks)

        print(f"[freerun]   free (self-history):", flush=True)
        for g in sorted(per):
            n, ok, th = per[g]
            line = f"    gold {g:8s} n={n:4d}  acc {ok / n:5.3f}"
            if th:
                line += "  <- " + ", ".join(
                    f"{k} x{c}" for k, c in th.most_common(3))
            print(line, flush=True)
        print(f"[freerun]   WORD ONSETS: {exact}/{n_word} exact, "
              f"{near}/{n_word} within +-1s; false speech at "
              f"{false_speech} non-gold ticks; "
              f"{pieces} continuation pieces", flush=True)
    print("[freerun] done", flush=True)


if __name__ == "__main__":
    main()
