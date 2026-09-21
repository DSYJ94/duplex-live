"""Phase-1 trainer v1 (docs/09 §3 step 2): first gradients on factory data.

Plain words: the model's whole brain is FROZEN except the ten new timing
words (<sil>, <yield>, <delegate>, ...). We show it the interleaved
lessons (sound heard, then the answer it should have given each beat) and
let it learn ONLY what those new words mean in context. Old-vocabulary
knowledge can't drift: their gradient rows are zeroed by a hook.

  python -m layera.train_p1 --shards runs/shards --batch runs/synth-big7/audio \
      --steps 32 --out runs/p1-v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def mask_old_rows(grad, n_old: int):
    """Zero gradients for every vocabulary row below the new tokens."""
    g = grad.clone()
    g[:n_old] = 0
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--batch", required=True, help="factory batch (grids+stems)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--timing-weight", type=float, default=50.0,
                    help="loss weight on rare duplex tokens (timing; "
                         "+<yield> with --weight-yield). Dose-response "
                         "v2-v5: only the DATA×WEIGHT product matters "
                         "(~50 effective = timing learns); concentrating "
                         "it all in weight (v3, 50x) breaks <sil>")
    ap.add_argument("--weight-yield", action="store_true",
                    help="also weight <yield> — same rarity problem "
                         "(v4 probe: 0%% on held-out; v5 showed yield "
                         "responds to data, timing only to weight)")
    ap.add_argument("--out", default="runs/p1-v1")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layera.sampler import build_from_grid, collate
    from tokenizer.vocab import register_into, TIMING_BUCKETS_S, YIELD

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16", device_map="auto",
        attn_implementation="sdpa",
        # eager experts for TRAINING ONLY: batched_mm gathers per-token
        # expert weights (~24 GiB up + ~12 GiB down alive together per
        # layer) and OOMs the 95 GiB GH200 once backward must hold them;
        # eager loops experts one at a time (~6 MiB each). Inference jobs
        # keep batched_mm for speed.
        experts_implementation="eager")
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    n_old = len(proc.tokenizer)
    register_into(proc.tokenizer)
    model.resize_token_embeddings(len(proc.tokenizer))
    # docs/09 §2c: new rows start at mean-of-neighbors, not random
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean = emb[:n_old].mean(dim=0)
        emb[n_old:] = mean.to(emb.dtype)
        head = model.lm_head.weight if hasattr(model, "lm_head") else \
            model.get_output_embeddings().weight
        head[n_old:] = mean.to(head.dtype)

    # freeze the world, then unfreeze only embed+head with row masking
    for p in model.parameters():
        p.requires_grad_(False)
    emb_w = model.get_input_embeddings().weight
    head_w = model.lm_head.weight if hasattr(model, "lm_head") else \
        model.get_output_embeddings().weight
    for w in (emb_w, head_w):
        w.requires_grad_(True)
        w.register_hook(lambda g: mask_old_rows(g, n_old))
    model.cuda().train()
    # backward must reach the embedding table at the bottom; without
    # checkpointing the batched_mm expert path keeps a ~12 GiB scratch per
    # layer and OOMs (95 GiB GH200). Enable AFTER .train() — the per-layer
    # check is `flag and self.training` — then verify the flag actually
    # landed on the decoder layers (first run silently didn't).
    model.gradient_checkpointing_enable()
    flagged = sum(1 for m in model.modules()
                  if getattr(m, "gradient_checkpointing", False))
    n_layers = sum(1 for m in model.modules()
                   if isinstance(m, torch.nn.Module)
                   and "DecoderLayer" in type(m).__name__)
    print(f"[p1] checkpoint flags set on {flagged} modules "
          f"({n_layers} decoder layers); training={model.training}",
          flush=True)
    assert flagged > 0, "gradient checkpointing did not engage"

    # rare-token weighted loss: timing (~0.06% of graded) and <yield>
    # (~0.02%) never win plain CE (v2-v5 probes); give their positions
    # extra loss weight. v6 plan: weight 10x + 5x oversampled lessons =
    # ~50x effective, spread across the run instead of concentrated.
    rare_names = [f"<t+{n}s>" for n in TIMING_BUCKETS_S]
    if args.weight_yield:
        rare_names.append(YIELD)
    rare_ids = torch.tensor(
        [proc.tokenizer.convert_tokens_to_ids(t) for t in rare_names],
        device=model.device)

    shards = sorted(Path(args.shards).glob("*.pt"))
    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    by_id = {r["scenario_id"]: r for r in rows}
    opt = torch.optim.AdamW([emb_w, head_w], lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "a")
    step = 0
    t0 = time.time()
    losses = []
    while step < args.steps:
        for sp in shards:
            shard = torch.load(sp, weights_only=False)
            sid = shard["scenario_id"]
            if sid not in by_id:
                continue
            seq = build_from_grid(batch / by_id[sid]["dir"], proc.tokenizer)
            if len(seq["ids"]) < 8:
                continue
            batch_in = collate(seq, shard, model.get_input_embeddings())
            batch_in = {k: v.to(model.device) for k, v in batch_in.items()}
            if step == 0:
                print(f"[p1] mem before fwd: "
                      f"{torch.cuda.memory_allocated()/1e9:.1f} GB, "
                      f"seq len {batch_in['inputs_embeds'].shape[1]}",
                      flush=True)
            out_logits = model(**batch_in, use_cache=False).logits
            # causal shift: logits[t] predicts labels[t+1]
            logits = out_logits[:, :-1]
            labels = batch_in["labels"][:, 1:]
            per_tok = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1), ignore_index=-100, reduction="none")
            w = torch.ones_like(per_tok)
            is_rare = (labels.reshape(-1).unsqueeze(-1)
                       == rare_ids.unsqueeze(0)).any(-1)
            w = w.masked_fill(is_rare, args.timing_weight)
            loss = (per_tok * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            losses.append(float(loss))
            log.write(json.dumps({"step": step, "scenario": sid,
                                  "loss": round(float(loss), 4)}) + "\n")
            log.flush()
            if step % 8 == 0:
                print(f"[p1] step {step}: loss {sum(losses[-8:])/8:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step >= args.steps:
                break

    # text_config.tie_word_embeddings is False -> embed and head are
    # SEPARATE tensors; both new-row deltas must be saved or the trained
    # head is lost on reload.
    torch.save({"emb_delta_new_rows": emb_w[n_old:].detach().cpu(),
                "head_delta_new_rows": head_w[n_old:].detach().cpu(),
                "n_old": n_old, "steps": step, "args": vars(args)},
               out / "p1_checkpoint.pt")
    print(f"[p1] done: {step} steps, final loss {losses[-1]:.4f} -> {out}")


if __name__ == "__main__":
    main()
