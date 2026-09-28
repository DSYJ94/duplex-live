"""Aux activity gate (owner decision 2026-09-28, after probe 11).

The three-point law (probes 9/10/11): the word-placement mapping lives
EXACTLY in the clean-gold history distribution — any training-side
corruption of that history erodes placement ~1:1 against the quiet-
discipline gain. Diagnosis: ONE softmax head carries both skills, and
the ww3 word pole keeps the speech-rate prior high no matter what the
history says.

This script adds a SEPARATE yes/no head — the activity gate — that
answers only "should anyone be speaking this tick?". The base model is
a FROZEN checkpoint (dense-pt3: perfect placement) — nothing about it
moves. The gate is a small MLP on the per-tick trunk hidden state (the
same state that feeds the word head at the tick's audio position),
trained with plain binary CE on gold speak/quiet labels. Half the
samples use the probe-11 corrupt-gold histories, so the gate sees the
off-gold states decode actually visits. At decode (freerun_diag
--gate-ckpt) a word counts only if the model proposes it AND the gate
says speech — each skill in its own parameters.

  python -m layera.train_gate --shards runs/shards-dense2 \
      --batch runs/synth-dense1/audio --steps 200 \
      --lessons dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004 \
      --init-ckpt runs/dense-pt3/p1_checkpoint.pt --out runs/dense-gate1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from layera.train_p1 import corrupt_gold


class GateHead(torch.nn.Module):
    """Per-tick speech probability from the trunk hidden state that
    feeds the word head. Own LayerNorm first — the captured lm_head
    input may or may not be pre-normed, and the gate learns around
    whichever it gets."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # [..., dim] -> [...] logit
        return self.net(self.norm(h.to(torch.float32))).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lessons", default=None,
                    help="comma-separated scenario_ids (capacity mode)")
    ap.add_argument("--corrupt-prob", type=float, default=0.5,
                    help="fraction of samples trained on corrupt-gold "
                         "histories (probe-11 corruption; the rest stay "
                         "clean gold — same split as the SH probes)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gate-dim", type=int, default=256)
    ap.add_argument("--init-ckpt", required=True,
                    help="FROZEN base checkpoint (p1_checkpoint.pt) — "
                         "placement lives here and must not move")
    ap.add_argument("--out", default="runs/dense-gate1")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layera.sampler import build_from_grid
    from tokenizer.vocab import register_into

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16",
        device_map="auto", attn_implementation="sdpa",
        # forward-only (no backward reaches the backbone), so the fast
        # inference expert path is safe here — the eager switch exists
        # only because training must hold gathered expert weights
        experts_implementation="batched_mm").eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    n_old = len(proc.tokenizer)
    register_into(proc.tokenizer)
    model.resize_token_embeddings(len(proc.tokenizer))
    ck = torch.load(args.init_ckpt, weights_only=False)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        head = model.lm_head.weight
        emb[n_old:] = ck["emb_delta_new_rows"].to(emb.device, emb.dtype)
        head[n_old:] = ck["head_delta_new_rows"].to(head.device, head.dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[gate] base FROZEN from {args.init_ckpt} "
          f"(base run: {ck.get('steps')} steps)", flush=True)

    # capture the hidden state that feeds the word head: a forward
    # pre-hook on lm_head works no matter how the wrapper's forward is
    # signed (no reliance on output_hidden_states pass-through)
    cap = {}
    model.lm_head.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0]))

    gate = GateHead(emb.shape[1], args.gate_dim).cuda().train()
    n_gate = sum(p.numel() for p in gate.parameters())
    print(f"[gate] training {n_gate} gate params "
          f"(dim {emb.shape[1]} -> {args.gate_dim})", flush=True)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr)

    # duplex-token machinery for sh_kind (mirrors train_p1)
    duplex_range = range(n_old, len(proc.tokenizer))
    name_of = {i: proc.tokenizer.convert_ids_to_tokens(i)
               for i in duplex_range}

    def sh_kind(tok: int) -> str:
        if tok < n_old:
            return "word"
        name = name_of[tok]
        if name == "<sil>":
            return "sil"
        if name == "<speak>":
            return "speak"
        return "special"

    shards = sorted(Path(args.shards).glob("*.pt"))
    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    by_id = {r["scenario_id"]: r for r in rows}
    want = ({s.strip() for s in args.lessons.split(",") if s.strip()}
            if args.lessons else None)
    if want is not None:
        shards = [sp for sp in shards if sp.stem in want]
        print(f"[gate] capacity mode: {[sp.stem for sp in shards]}",
              flush=True)
    emb_layer = model.get_input_embeddings()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    t0 = time.time()
    stats = {"loss": 0.0, "n": 0, "hit": 0, "speech": 0, "pred": 0}
    while step < args.steps:
        for sp in shards:
            shard = torch.load(sp, weights_only=False)
            sid = shard["scenario_id"]
            if want is not None and sid not in want:
                continue
            seq = build_from_grid(batch / by_id[sid]["dir"], proc.tokenizer)
            if len(seq["ids"]) < 8:
                continue
            pos = seq["audio_pos"]
            # per-tick gate labels from the GOLD emission, captured
            # BEFORE corruption touches the inputs
            tick_lab = [1 if sh_kind(seq["ids"][p + 1]) in ("word", "speak")
                        else 0 for p in pos]
            if args.corrupt_prob > 0 and \
                    random.random() < args.corrupt_prob:
                seq = corrupt_gold(seq, sh_kind)
            ids = torch.tensor([seq["ids"]], device=model.device)
            with torch.no_grad():
                embeds = emb_layer(ids).clone()
                feats = shard["audio_feats"].to(embeds.device,
                                                embeds.dtype)
                n = min(len(pos), feats.shape[0])
                embeds[0, torch.tensor(pos[:n], device=embeds.device)] = \
                    feats[:n]
                model(inputs_embeds=embeds, use_cache=False)
            if "h" not in cap:
                raise RuntimeError(
                    "lm_head pre-hook never fired — the wrapper's "
                    "forward does not route logits through model.lm_head")
            h = cap["h"]  # [1, L, d] — the tick decisions read pos[t]
            n_t = min(n, len(tick_lab))
            tick_pos = torch.tensor(pos[:n_t], device=h.device)
            logits = gate(h[0, tick_pos])
            labels = torch.tensor(tick_lab[:n_t], device=h.device,
                                  dtype=logits.dtype)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = (logits > 0).float()
                stats["loss"] += float(loss)
                stats["n"] += n_t
                stats["hit"] += int((pred == labels).sum())
                stats["speech"] += int(labels.sum())
                stats["pred"] += int(pred.sum())
            step += 1
            if step % 8 == 0:
                acc = stats["hit"] / max(1, stats["n"])
                print(f"[gate] step {step}: loss "
                      f"{stats['loss'] / max(1, step):.4f} acc {acc:.3f} "
                      f"(speech {stats['speech'] / max(1, stats['n']):.2f}, "
                      f"pred {stats['pred'] / max(1, stats['n']):.2f}) "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if step >= args.steps:
            break
    gate.eval()
    torch.save({"gate": gate.state_dict(), "gate_dim": args.gate_dim,
                "steps": step, "base_ckpt": args.init_ckpt,
                "corrupt_prob": args.corrupt_prob,
                "train_acc": stats["hit"] / max(1, stats["n"])},
               out / "gate_checkpoint.pt")
    print(f"[gate] done: {step} steps, acc "
          f"{stats['hit'] / max(1, stats['n']):.3f} -> "
          f"{out / 'gate_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
