"""Free-run diagnostic: teacher-forced exam vs live behavior, in isolation.

Plain words: the exam feeds the model the CORRECT history (gold answers
so far) and checks its next answer — that can look great even while live
behavior (where the model must eat its OWN answers) falls apart. This
script replays held-out lessons in the exact training format but lets
the model's own emissions ride along in the context, then compares:

  forced  = gold history (should match the probe: sil ~95%+)
  free    = self-history (does it chatter? does it recover to <sil>?)

It also prints the ear's token-rate alignment (one feature per 80 ms
tick is what the sampler assumes; anything else means training saw
time-warped audio).

  python -m layera.freerun_diag --batch runs/synth-big7/audio \
      --ckpt runs/p1-v3/p1_checkpoint.pt --shards runs/shards \
      --classes timed_task --limit 2
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

    duplex_ids = set(range(n_old, len(proc.tokenizer)))
    name_of = {i: proc.tokenizer.convert_ids_to_tokens(i)
               for i in duplex_ids}
    mask_ids = [i for i in proc.tokenizer.all_special_ids
                if i not in duplex_ids]
    pad_id = proc.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    tok_conv = proc.tokenizer.convert_tokens_to_ids
    emb_layer = model.get_input_embeddings()

    # ---- ear token-rate alignment -------------------------------------
    from tokenizer.grid import from_scenario_dir

    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    train_ids = ({Path(s).stem for s in Path(args.shards).glob("*.pt")}
                 if args.shards else set())
    held = []
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
          f"(classes={args.classes}; training={len(train_ids)} shards)",
          flush=True)

    # alignment on the FIRST lesson's audio: full stream + first window
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

    def run_lesson(sc_dir, n_ticks, feats, free: bool):
        """Replay one lesson tick-by-tick. free=False feeds the gold
        emission back (teacher forcing); free=True feeds the model's own
        pick. Returns per-gold-kind stats + recovery counts."""
        per = defaultdict(lambda: [0, 0, Counter()])
        wrong_streak = 0
        recovered = missed = 0
        cache = None
        for i in range(min(n_ticks, args.max_ticks)):
            e_in = emb_layer(torch.tensor([[pad_id]], device=model.device)
                             ).clone()
            e_in[0, 0] = feats[i].to(e_in.dtype)
            with torch.no_grad():
                out = model(inputs_embeds=e_in, past_key_values=cache,
                            use_cache=True)
            cache = out.past_key_values
            logits = out.logits[0, -1].clone()
            for tid in mask_ids:
                logits[tid] = float("-inf")
            tok_pick = int(logits.argmax())
            gold_name = grid.ticks[i].emission
            if gold_name in ALL_SPECIAL:
                gold_id = tok_conv(gold_name)
                gold = "<sil>" if gold_name == SIL else gold_name
            else:
                gold_id = None
                gold = "<word>"
            kind = kind_of(tok_pick)
            e = per[gold]
            e[0] += 1
            ok = (kind == "word") if gold_id is None else \
                (tok_pick == gold_id)
            e[1] += int(ok)
            if not ok:
                e[2][name_of.get(tok_pick, kind)] += 1
                wrong_streak += 1
            elif wrong_streak and gold == "<sil>":
                recovered += 1
                wrong_streak = 0
            # feed the emission (gold or own) back into the KV
            feed_tok = gold_id if (not free and gold_id is not None) \
                else tok_pick
            e_out = emb_layer(torch.tensor([[feed_tok]],
                                           device=model.device))
            with torch.no_grad():
                out = model(inputs_embeds=e_out, past_key_values=cache,
                            use_cache=True)
            cache = out.past_key_values
        return per, recovered

    for r in held:
        sc_dir = batch / r["dir"]
        grid = from_scenario_dir(sc_dir)
        pcm = mix_in_slot(sc_dir, grid.n_ticks)
        feats, counts = encode_audio(model, proc, pcm)
        print(f"[freerun] {grid.scenario_id}: {grid.n_ticks} ticks, "
              f"{feats.shape[0]} ear tokens, windows={counts[:3]}...",
              flush=True)
        if feats.shape[0] < grid.n_ticks:
            print(f"[freerun]   WARNING: fewer ear tokens than ticks — "
                  f"training saw time-compressed audio", flush=True)
        for tag, free in (("forced", False), ("free", True)):
            per, rec = run_lesson(sc_dir, grid.n_ticks, feats, free)
            print(f"[freerun]   {tag}:", flush=True)
            for g in sorted(per):
                n, ok, th = per[g]
                line = (f"    gold {g:8s} n={n:4d}  acc {ok / n:5.3f}")
                if th:
                    line += "  <- " + ", ".join(
                        f"{k} x{c}" for k, c in th.most_common(3))
                print(line, flush=True)
            if free:
                print(f"[freerun]   sil recoveries after wrong beats: "
                      f"{rec}", flush=True)
    print("[freerun] done", flush=True)


if __name__ == "__main__":
    main()
