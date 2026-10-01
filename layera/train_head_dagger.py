"""RL-probe 1: head-DAgger on a frozen trunk (2026-10-01).

The five-point law (gate rounds 1-3 + scale1 + gate1-on-scale1) says
the trunk state cannot separate turn-start from chatter well enough
for a SIDE classifier (the gate family). This probe attacks the source
instead: the model's OWN duplex-token head rows. The base model
proposes a word on ~91% of quiet ticks — the word/quiet competition is
won by words almost always. DAgger rollout (raw decode, same as
train_gate --dagger) captures the exact lm_head input state at every
word-first-proposal tick WITH ITS GOLD TOKEN; then ONLY the
n_old..end head rows (the 11 duplex tokens) train on those states with
full-vocab cross-entropy. Trunk and word rows stay frozen; at quiet
states CE raises <sil>'s logit (words lose relatively through the same
softmax), at speech states CE lowers the duplex rows (the frozen gold
word piece wins). Placement must not move: input-embedding deltas pass
through from the base checkpoint unchanged.

Outputs:
  --out/p1_checkpoint.pt   same format as train_p1 (freerun_diag loads
                           it directly; emb deltas = base ck's)
  --save-states file       the rollout buffer (h fp32, gold token,
                           sampled token, scenario) so later iterations
                           skip the ~1.5 h rollout via --states

Derived from train_gate.py's DAgger block (proven in gate rounds 1-3);
the additions are gold-token recording and states saving.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--batch", required=True)
    ap.add_argument("--init-ckpt", required=True,
                    help="frozen base checkpoint (p1 format)")
    ap.add_argument("--out", default="runs/scale1-headdag")
    ap.add_argument("--lessons", default=None,
                    help="comma-separated rollout lessons")
    ap.add_argument("--states", default=None,
                    help="load a saved rollout buffer instead of "
                         "re-rolling (the cheap iteration path)")
    ap.add_argument("--save-states", default=None,
                    help="save the rollout buffer here")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--buffer-batch", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoProcessor, \
        Qwen3OmniMoeThinkerForConditionalGeneration

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layera.prepare import encode_audio, mix_in_slot
    from tokenizer.grid import from_scenario_dir
    from tokenizer.vocab import ALL_SPECIAL, SIL, register_into  # noqa: E402

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16",
        device_map="auto", attn_implementation="sdpa",
        # eager experts: forward-only here, and batched_mm's giant expert
        # gather is pointless per-tick (same reason as train_gate)
        experts_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    tokz = proc.tokenizer
    n_old = len(tokz)
    register_into(tokz)
    model.resize_token_embeddings(len(tokz))

    # fresh-init the new rows to the old-rows mean, then apply the base
    # checkpoint's deltas — exactly what freerun_diag does before a run
    with torch.no_grad():
        mean = model.get_input_embeddings().weight[:n_old].mean(dim=0)
        model.get_input_embeddings().weight[n_old:] = mean.to(
            model.get_input_embeddings().weight.dtype)
        model.lm_head.weight[n_old:] = mean.to(model.lm_head.weight.dtype)
    ck_base = torch.load(args.init_ckpt, weights_only=False)
    with torch.no_grad():
        model.get_input_embeddings().weight[n_old:] = \
            ck_base["emb_delta_new_rows"].to(model.device,
                                             torch.bfloat16)
        model.lm_head.weight[n_old:] = \
            ck_base["head_delta_new_rows"].to(model.device, torch.bfloat16)
    print(f"[headdag] base FROZEN from {args.init_ckpt} "
          f"(base run: {ck_base.get('steps')} steps)", flush=True)

    cap = {}
    model.lm_head.register_forward_pre_hook(
        lambda mod, inp: cap.__setitem__("h", inp[0]))

    duplex_ids = list(range(n_old, len(tokz)))
    name_of = {i: tokz.convert_ids_to_tokens(i) for i in duplex_ids}
    sil_id = next(i for i in duplex_ids if name_of[i] == "<sil>")
    mask_ids = [i for i in tokz.all_special_ids if i not in set(duplex_ids)]
    pad_id = tokz.convert_tokens_to_ids("<|audio_pad|>")
    tok_conv = tokz.convert_tokens_to_ids
    emb_layer = model.get_input_embeddings()
    V = len(tokz)

    def gold_token_of(emission: str) -> int:
        if emission == SIL:
            return sil_id
        if emission in ALL_SPECIAL:
            return tok_conv(emission)
        return tokz(emission, add_special_tokens=False)["input_ids"][0]

    # ---- DAgger rollout (raw decode; derived from train_gate) ---------
    rec = {"h": [], "gold": [], "pred": [], "sid": []}
    if args.states and Path(args.states).exists():
        buf = torch.load(args.states, weights_only=False)
        rec = {k: buf[k] for k in ("h", "gold", "pred", "sid")}
        print(f"[headdag] loaded {len(rec['gold'])} saved states from "
              f"{args.states}", flush=True)
    else:
        shards = sorted(Path(args.shards).glob("*.pt"))
        rows = [json.loads(l)
                for l in open(Path(args.batch) / "train_manifest.jsonl")]
        by_id = {r["scenario_id"]: r for r in rows}
        want = ({s.strip() for s in args.lessons.split(",") if s.strip()}
                if args.lessons else None)
        if want is not None:
            shards = [sp for sp in shards if sp.stem in want]
        print(f"[headdag] RAW-decode rollout on "
              f"{[sp.stem for sp in shards]}", flush=True)
        for sp in shards:
            shard = torch.load(sp, weights_only=False)
            sid = shard["scenario_id"]
            sc = Path(args.batch) / by_id[sid]["dir"]
            grid = from_scenario_dir(sc)
            pcm = mix_in_slot(sc, grid.n_ticks)
            feats, _counts = encode_audio(model, proc, pcm)
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
                lg = out.logits[0, -1].clone()
                for tid in mask_ids:
                    lg[tid] = float("-inf")
                tok_id = int(lg.argmax())
                if tok_id < n_old:
                    # word-first PROPOSAL tick: this is the decision the
                    # probe retrains — record state + gold token
                    n_prop += 1
                    rec["h"].append(
                        cap["h"][0, -1].detach().float().cpu())
                    rec["gold"].append(gold_token_of(grid.ticks[i].emission))
                    rec["pred"].append(tok_id)
                    rec["sid"].append(sid)
                # an accepted word completes through its pieces (decode)
                n_pc = 0
                while tok_id < n_old and n_pc < 8:
                    e_out = emb_layer(torch.tensor([[tok_id]],
                                                   device=model.device))
                    with torch.no_grad():
                        out = model(inputs_embeds=e_out,
                                    past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    lg2 = out.logits[0, -1].clone()
                    for tid in mask_ids:
                        lg2[tid] = float("-inf")
                    tok_id = int(lg2.argmax())
                    if tok_id >= n_old:
                        break
                    n_pc += 1
            print(f"[headdag] dagger {sid}: {n_prop} proposal states",
                  flush=True)

    if args.save_states and not (args.states and Path(args.states).exists()):
        out_s = Path(args.save_states)
        out_s.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_s.with_suffix(".tmp")
        torch.save(rec, tmp)
        os.replace(tmp, out_s)
        print(f"[headdag] states saved: {out_s}", flush=True)

    H = torch.stack(rec["h"]).to(model.device)          # [N, d] fp32
    gold = torch.tensor(rec["gold"], device=model.device)
    N = H.shape[0]
    n_word_gold = int((gold < n_old).sum())
    print(f"[headdag] buffer: {N} states "
          f"({n_word_gold} word-gold = {n_word_gold / N:.1%}, "
          f"{N - n_word_gold} control-gold)", flush=True)

    # ---- train ONLY the duplex head rows ------------------------------
    W = model.lm_head.weight                             # [V, d] frozen
    # rollout states are captured fp32 (job 7001927: fp32 h @ bf16 W
    # refuses to run) — train the head in fp32 throughout, cast back
    # to bf16 only at checkpoint save. W is ~151k x 2048 fp32 = 1.2 GB.
    W_frozen = W.detach().float()
    dW = torch.nn.Parameter(W[n_old:].detach().float().clone())
    opt = torch.optim.AdamW([dW], lr=args.lr)
    idx = torch.tensor(duplex_ids, device=H.device)
    mask = torch.zeros(V, dtype=torch.bool, device=H.device)
    mask[mask_ids] = True

    def full_logits(h: torch.Tensor) -> torch.Tensor:
        # frozen vocab columns + trainable duplex columns, one tensor
        lg = h @ W_frozen.T
        return lg.scatter(1, idx.unsqueeze(0).expand(h.shape[0], -1),
                          (h @ dW.T).to(lg.dtype))

    step = 0
    t0 = time.time()
    while step < args.steps:
        sel = torch.randint(0, N, (min(args.buffer_batch, N),),
                            device=H.device)
        lg = full_logits(H[sel]).masked_fill(mask, float("-inf"))
        loss = F.cross_entropy(lg, gold[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 25 == 0 or step == args.steps:
            with torch.no_grad():
                lga = full_logits(H).masked_fill(mask, float("-inf"))
                pred = lga.argmax(1)
                hit = (pred == gold).float().mean()
                wmask = gold < n_old
                hit_word = (pred[wmask] == gold[wmask]).float().mean() \
                    if wmask.any() else torch.tensor(0.0)
                hit_ctl = (pred[~wmask] == gold[~wmask]).float().mean() \
                    if (~wmask).any() else torch.tensor(0.0)
                sil_ok = (pred[~wmask] == sil_id).float().mean() \
                    if (~wmask).any() else torch.tensor(0.0)
            print(f"[headdag] step {step}: loss {float(loss):.4f} "
                  f"acc {hit:.3f} (word {hit_word:.3f} / ctl "
                  f"{hit_ctl:.3f}, sil-among-ctl {sil_ok:.3f}) "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = {"emb_delta_new_rows": ck_base["emb_delta_new_rows"],
          "head_delta_new_rows": dW.detach().to(torch.bfloat16).cpu(),
          "n_old": n_old, "steps": args.steps,
          "args": {**vars(args), "head_dagger": True}}
    tmp = out / "p1_checkpoint.pt.tmp"
    torch.save(ck, tmp)
    os.replace(tmp, out / "p1_checkpoint.pt")
    print(f"[headdag] done -> {out / 'p1_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
