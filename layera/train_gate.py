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

Round 2 (--dagger, owner decision 2026-09-28 after gate1 half-worked):
gate1's scores didn't rank gold onsets above chatter because the gate
only ever saw gold + bounded synthetic-corrupt histories, while decode
visits the model's own compounded states. --dagger first free-runs
each lesson EXACTLY as decode does (same loop as freerun_diag, with
--dagger-gate optionally ACTIVE during the rollout — the states this
decode config really visits), records the hidden state at every
word-first proposal tick with its GOLD label, then trains on that
buffer mixed with the gate1 gold/corrupt stream.
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
    ap.add_argument("--dagger", action="store_true",
                    help="collect on-policy states first: free-run each "
                         "lesson exactly as decode does and add the "
                         "word-first-proposal states (gold labels) to "
                         "the training mix")
    ap.add_argument("--dagger-gate", default=None,
                    help="gate checkpoint ACTIVE during the rollout "
                         "(iterated DAgger: collect the states THIS "
                         "decode config actually visits)")
    ap.add_argument("--rollout-threshold", type=float, default=0.35,
                    help="gate threshold used during a --dagger-gate "
                         "rollout")
    ap.add_argument("--buffer-batch", type=int, default=256,
                    help="on-policy buffer states sampled per step into "
                         "the loss")
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
    from layera.prepare import encode_audio, mix_in_slot
    from tokenizer.grid import from_scenario_dir
    from tokenizer.vocab import (  # noqa: E402
        is_speech_emission, is_speech_tok, register_into)

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16",
        device_map="auto", attn_implementation="sdpa",
        # eager experts even though this is forward-only: batched_mm
        # gathers per-token expert weights and at training sequence
        # lengths (~2800 tokens) that is a single ~43 GiB allocation
        # on top of the ~60 GiB model — OOM (job 6912797). Eager loops
        # experts one at a time (~6 MiB each), same as train_p1
        experts_implementation="eager").eval()
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
    adapter = None
    if ck.get("adapter_state"):
        # mirror freerun_diag: an adapted base changes the ear feats the
        # rollout states are computed from
        ad_dim = int(ck.get("args", {}).get("adapter_dim", 256))
        hid = model.get_input_embeddings().weight.shape[1]
        adapter = torch.nn.Sequential(
            torch.nn.Linear(hid, ad_dim), torch.nn.GELU(),
            torch.nn.Linear(ad_dim, hid))
        adapter.load_state_dict(ck["adapter_state"])
        adapter = adapter.to(model.device).eval()
        print("[gate] audio adapter loaded (rollout feats adapted)",
              flush=True)

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

    pad_id = proc.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    mask_ids = [i for i in proc.tokenizer.all_special_ids
                if i not in set(duplex_range)]
    sil_id = next(i for i in duplex_range if name_of[i] == "<sil>")

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

    # ---- round 2: on-policy state collection (DAgger) ------------------
    # free-run each lesson exactly as decode does and keep the hidden
    # state at every word-first PROPOSAL tick (the only ticks the gate
    # is ever queried at) with its GOLD label
    buf_h: list = []
    buf_y: list = []
    H = Y = pw = None
    if args.dagger:
        roll_gate = None
        if args.dagger_gate:
            g1 = torch.load(args.dagger_gate, weights_only=False)
            roll_gate = GateHead(emb.shape[1],
                                 int(g1["gate_dim"])).to(model.device).eval()
            roll_gate.load_state_dict(g1["gate"])
            print(f"[gate] dagger rollout with ACTIVE gate "
                  f"{args.dagger_gate} (threshold "
                  f"{args.rollout_threshold:g})", flush=True)
        else:
            print("[gate] dagger rollout with RAW base decode",
                  flush=True)
        for sp in shards:
            shard = torch.load(sp, weights_only=False)
            sid = shard["scenario_id"]
            if want is not None and sid not in want:
                continue
            sc = batch / by_id[sid]["dir"]
            grid = from_scenario_dir(sc)
            pcm = mix_in_slot(sc, grid.n_ticks)
            feats, _counts = encode_audio(model, proc, pcm)
            if adapter is not None:
                feats = adapter(feats.to(model.device, torch.float32)
                                ).to(feats.dtype)
            cache = None
            n_prop = 0
            for i in range(min(grid.n_ticks, feats.shape[0])):
                e_in = emb_layer(torch.tensor([[pad_id]],
                                              device=model.device)).clone()
                e_in[0, 0] = feats[i].to(e_in.dtype)
                with torch.no_grad():
                    out = model(inputs_embeds=e_in,
                                past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                h_tick = cap["h"][0, -1].detach().clone()
                lg = out.logits[0, -1].clone()
                for tid in mask_ids:
                    lg[tid] = float("-inf")
                tok = int(lg.argmax())
                if tok < n_old:
                    n_prop += 1
                    em = grid.ticks[i].emission
                    buf_h.append(h_tick)
                    buf_y.append(1 if is_speech_emission(em) else 0)
                    if roll_gate is not None:
                        p = float(torch.sigmoid(
                            roll_gate(h_tick.unsqueeze(0))[0]))
                        if p < args.rollout_threshold:
                            tok = sil_id  # blocked: never fed, word dies
                # an accepted word completes through its pieces, as decode
                n_pc = 0
                while tok < n_old and n_pc < 8:
                    e_out = emb_layer(torch.tensor([[tok]],
                                                   device=model.device))
                    with torch.no_grad():
                        out = model(inputs_embeds=e_out,
                                    past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    lg2 = out.logits[0, -1].clone()
                    for tid in mask_ids:
                        lg2[tid] = float("-inf")
                    tok = int(lg2.argmax())
                    if tok >= n_old:
                        break
                    n_pc += 1
            print(f"[gate] dagger {sid}: {n_prop} proposal states",
                  flush=True)
        if buf_y:
            H = torch.stack(buf_h)
            Y = torch.tensor(buf_y, dtype=torch.float32, device=H.device)
            n_pos = int(Y.sum())
            pw = torch.tensor([float((Y == 0).sum()) / max(1, n_pos)],
                              dtype=torch.float32, device=H.device)
            print(f"[gate] dagger buffer: {len(buf_y)} states "
                  f"({n_pos} speech, {n_pos / len(buf_y):.2%}) "
                  f"pos_weight {pw.item():.1f}", flush=True)

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
            # BEFORE corruption touches the inputs (same speech
            # definition as the freerun metrics — tokenizer.vocab)
            tick_lab = [1 if is_speech_tok(seq["ids"][p + 1], n_old,
                                           name_of) else 0 for p in pos]
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
            if buf_y:
                # on-policy half of round 2: the states decode actually
                # visits, gold labels; pos_weight rebalances the heavy
                # quiet majority (pt3 proposes words on most ticks and
                # nearly all of those are chatter = label 0)
                idx = torch.randint(0, len(buf_y),
                                    (args.buffer_batch,), device=H.device)
                loss = loss + F.binary_cross_entropy_with_logits(
                    gate(H[idx]), Y[idx], pos_weight=pw)
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
                extra = ""
                if buf_y:
                    with torch.no_grad():
                        bidx = torch.randint(0, len(buf_y), (512,),
                                             device=H.device)
                        bacc = float(((gate(H[bidx]) > 0).float()
                                      == Y[bidx]).float().mean())
                    extra = f", buf {bacc:.3f}"
                print(f"[gate] step {step}: loss "
                      f"{stats['loss'] / max(1, step):.4f} acc {acc:.3f} "
                      f"(speech {stats['speech'] / max(1, stats['n']):.2f}, "
                      f"pred {stats['pred'] / max(1, stats['n']):.2f}"
                      f"{extra}) ({time.time() - t0:.0f}s)", flush=True)
        if step >= args.steps:
            break
    gate.eval()
    torch.save({"gate": gate.state_dict(), "gate_dim": args.gate_dim,
                "steps": step, "base_ckpt": args.init_ckpt,
                "corrupt_prob": args.corrupt_prob,
                "dagger": bool(args.dagger),
                "dagger_gate": args.dagger_gate,
                "buffer_states": len(buf_y),
                "train_acc": stats["hit"] / max(1, stats["n"])},
               out / "gate_checkpoint.pt")
    print(f"[gate] done: {step} steps, acc "
          f"{stats['hit'] / max(1, stats['n']):.3f} -> "
          f"{out / 'gate_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
