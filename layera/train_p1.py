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
import os
import random
import time
from pathlib import Path

import torch


def mask_old_rows(grad, n_old: int):
    """Zero gradients for every vocabulary row below the new tokens."""
    g = grad.clone()
    g[:n_old] = 0
    return g


CORRUPT_STATS = [0, 0]  # samples corrupted, emission slots overwritten


def corrupt_gold(seq: dict, sh_kind, rng=random) -> dict:
    """Self-history redesign after probe 10 (2026-09-28). Recollected
    self-histories only ever covered the two degenerate attractors
    (all-chatter / all-mute — the recollection log flip-flopped
    1285/1344 -> 2/1344 picks), so training saw contradictory stories
    about itself and probe 9's placement eroded. Instead: corrupt the
    GOLD history the way decode actually drifts, keeping the mistakes
    small and graded —

      chatter burst: a run of quiet ticks whose <sil> slot becomes a
        word-first pick (the exact decode-chatter signature: one new
        word per tick),
      early onset: a word's first piece bleeding into the preceding
        quiet tick (decode starts the word before the gold turn).

    LABELS stay gold — every corrupted sample is a small mistake with
    its own correction as the target ("I just babbled -> hush"; "I
    started early -> still hit the real onset"). ~1% of slots per
    lesson, so the placement signal keeps its clean-gold backbone."""
    ids = list(seq["ids"])
    pos = seq["audio_pos"]
    ticks = []  # (first_slot, end_slot, kind) per tick
    for t, p in enumerate(pos):
        end = pos[t + 1] if t + 1 < len(pos) else len(ids)
        if end - p >= 2:
            ticks.append((p + 1, end, sh_kind(ids[p + 1])))
    piece_ids = [j for s, e, k in ticks if k == "word" for j in ids[s:e]]
    quiet_runs, cur = [], []
    for i, (s, e, k) in enumerate(ticks):
        if k == "sil":
            cur.append(i)
        elif cur:
            quiet_runs.append(cur)
            cur = []
    if cur:
        quiet_runs.append(cur)
    if not piece_ids or not quiet_runs:
        return dict(seq)
    n = 0
    if rng.random() < 0.75:
        run = rng.choice(quiet_runs)
        k = rng.randint(1, min(12, len(run)))
        start = rng.randint(0, len(run) - k)
        for i in run[start:start + k]:
            ids[ticks[i][0]] = rng.choice(piece_ids)
            n += 1
    if rng.random() < 0.35:
        for i, (s, e, k) in enumerate(ticks):
            if (k == "word" and i > 0 and ticks[i - 1][2] == "sil"
                    and rng.random() < 0.5):
                ids[ticks[i - 1][0]] = ids[s]
                n += 1
                break
    if n:
        CORRUPT_STATS[0] += 1
        CORRUPT_STATS[1] += n
        if CORRUPT_STATS[0] % 100 == 0:
            print(f"[p1] corrupt-gold: {CORRUPT_STATS[0]} samples "
                  f"corrupted, {CORRUPT_STATS[1]} slots", flush=True)
    return dict(seq, ids=ids)


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
    ap.add_argument("--class-cost", type=float, default=0.0,
                    help="per-CLASS cost (v8). RESULT: label-blind — "
                         "dL/dz_k = p_k(c_k - E[c]) and -c_y is a constant, "
                         "so the correct answer never enters the gradient; "
                         "the model only learns 'never emit rare tokens' "
                         "(v8 probe: t+*s prob 0.000, sil/yield stuck at a "
                         "flat 0.250 in every scenario). Superseded by "
                         "--thief-penalty (v9)")
    ap.add_argument("--thief-penalty", type=float, default=0.0,
                    help="v9: down-push on rare-token probability mass at "
                         "positions whose label is NOT a rare token. The "
                         "correct v8: the 50x up-push stays in the weighted "
                         "CE (label-dependent), the thief tax is separate "
                         "and only fires where rare tokens are wrong — "
                         "v3's failure was exactly that (t+30s stealing "
                         "584/588 sil spots)")
    ap.add_argument("--word-weight", type=float, default=1.0,
                    help="v11: loss weight on WORD positions (labels that "
                         "are regular-vocab tokens). <sil> covers ~95%%+ "
                         "of graded ticks, so every model — even "
                         "unweighted v2 — learns a silence attractor and "
                         "mutes live (all three gates 2/5, t1 zero word "
                         "beats). Same lever that taught timing, aimed at "
                         "speaking. Bracket per cliffs-not-slopes: 3, "
                         "then 10 if words don't move, 1.5 if sil "
                         "control breaks")
    ap.add_argument("--adapter", action="store_true",
                    help="capacity test (Codex F5-2): also train a small "
                         "audio-adapter MLP (hidden-size -> --adapter-dim "
                         "-> hidden-size) on the ear features, instead of "
                         "rows-only. ~1M params vs ~41k. Answers whether "
                         "the 10-row approach can learn conversation "
                         "behavior at all")
    ap.add_argument("--adapter-dim", type=int, default=256)
    ap.add_argument("--lessons", default=None,
                    help="comma-separated scenario_ids — capacity-test "
                         "mode: train ONLY these (e.g. the 4 probe "
                         "conversations), ignore the rest of the shard dir")
    ap.add_argument("--self-history", type=float, default=0.0,
                    help="self-history probe (exposure fix): each step "
                         "trains on the model's OWN free-run history with "
                         "this probability (0.5 -> half the steps). The "
                         "capacity test (6792730/6814321) indicted "
                         "teacher-forcing: models graded free-run never "
                         "see their own chatter in training. Audio pads "
                         "and gold LABELS are unchanged — only the input "
                         "tokens at emission slots become on-policy. "
                         "Random draw, not a stride: the shard loop "
                         "restarts every pass, so a fixed stride would "
                         "give shards PERMANENT roles")
    ap.add_argument("--sh-refresh", type=int, default=8,
                    help="recollect the model's own history only every "
                         "Nth self-history step per lesson (one free-run "
                         "pass costs one forward per ear feature)")
    ap.add_argument("--sh-mode", choices=["recollect", "corrupt"],
                    default="recollect",
                    help="self-history source (probe 11 redesign, "
                         "2026-09-28): recollect = free-run the current "
                         "model and train on its picks — probe 10 showed "
                         "that flip-flops between the chatter and mute "
                         "attractors and erodes placement. corrupt = "
                         "bounded corruption of the GOLD history (word-"
                         "first chatter bursts in quiet stretches, word "
                         "starts bleeding early); labels stay gold so "
                         "every corrupted sample teaches the recovery")
    ap.add_argument("--max-pieces", type=int, default=8,
                    help="cap on continuation pieces when free-running "
                         "for self-history (matches live decode)")
    ap.add_argument("--init-ckpt", default=None,
                    help="curriculum phase 2: start the new-row deltas "
                         "from a p1_checkpoint (emb+head) instead of "
                         "mean-init. Why: probe 9 showed placement forms "
                         "under pure teacher forcing but chatter needs "
                         "exposure — resume the rows and turn SH on, "
                         "rather than re-learning both from scratch")
    ap.add_argument("--out", default="runs/p1-v1")
    ap.add_argument("--save-every", type=int, default=0,
                    help="periodic checkpoint every N steps (0 = only "
                         "at the end). Why: a long run killed by a "
                         "timeout keeps its trained rows instead of "
                         "losing everything (gate-3 lesson)")
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
    if args.init_ckpt:
        # curriculum resume: overwrite the mean-init with a previous
        # phase's trained rows (same delta keys the checkpoint saves)
        prev = torch.load(args.init_ckpt, weights_only=False)
        with torch.no_grad():
            emb[n_old:] = prev["emb_delta_new_rows"].to(
                emb.device, emb.dtype)
            head[n_old:] = prev["head_delta_new_rows"].to(
                head.device, head.dtype)
        print(f"[p1] INIT from {args.init_ckpt} "
              f"(previous run: {prev.get('steps')} steps)", flush=True)

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
    want = ({s.strip() for s in args.lessons.split(",") if s.strip()}
            if args.lessons else None)
    if want is not None:
        # capacity mode: drop unwanted shards BEFORE the loop — with 4
        # lessons and hundreds of steps, sweep-skipping would re-load the
        # whole shard dir every pass (~100x wasted I/O)
        shards = [sp for sp in shards if sp.stem in want]
        print(f"[p1] capacity mode: training only {[sp.stem for sp in shards]}",
              flush=True)
    adapter = None
    if args.adapter:
        dim = emb_w.shape[1]  # embedding width — no config-path guessing
        adapter = torch.nn.Sequential(
            torch.nn.Linear(dim, args.adapter_dim), torch.nn.GELU(),
            torch.nn.Linear(args.adapter_dim, dim)).cuda()
        print(f"[p1] ADAPTER arm: {sum(p.numel() for p in adapter.parameters())}"
              f" trainable adapter params (dim {args.adapter_dim})",
              flush=True)
    opt = torch.optim.AdamW(
        [emb_w, head_w] + (list(adapter.parameters()) if adapter else []),
        lr=args.lr)

    # ---- self-history plumbing (exposure fix) --------------------------
    # Decode mirrors live/freerun exactly: chat-envelope specials masked,
    # word continuation fed back, first non-word breaker dropped.
    pad_id = proc.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    duplex_range = range(n_old, len(proc.tokenizer))
    name_of = {i: proc.tokenizer.convert_ids_to_tokens(i)
               for i in duplex_range}
    sh_mask_ids = [i for i in proc.tokenizer.all_special_ids
                   if i not in duplex_range]

    def sh_kind(tok: int) -> str:
        if tok < n_old:
            return "word"
        name = name_of[tok]
        if name == "<sil>":
            return "sil"
        if name == "<speak>":
            return "speak"
        return "special"

    emb_layer = model.get_input_embeddings()

    @torch.no_grad()
    def own_history(feats, n_ticks, sid: str):
        """Free-run the lesson (no grad): one decision per ear feature,
        continuations fed back. Returns per-tick own token lists
        ([first_pick, *pieces]) — the model's real behaviour."""
        was_training = model.training
        model.eval()
        try:
            own, cache = [], None
            for _i in range(min(len(feats), n_ticks)):
                e_in = emb_layer(
                    torch.tensor([[pad_id]], device=model.device)).clone()
                e_in[0, 0] = feats[_i].to(e_in.dtype)
                out = model(inputs_embeds=e_in,
                            past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits = out.logits[0, -1].clone()
                for tid in sh_mask_ids:
                    logits[tid] = float("-inf")
                tok = int(logits.argmax())
                picks = [tok]
                if sh_kind(tok) == "word":
                    while len(picks) < args.max_pieces:
                        e = emb_layer(torch.tensor(
                            [[picks[-1]]], device=model.device))
                        out = model(inputs_embeds=e,
                                    past_key_values=cache, use_cache=True)
                        cache = out.past_key_values
                        logits = out.logits[0, -1].clone()
                        for tid in sh_mask_ids:
                            logits[tid] = float("-inf")
                        nxt = int(logits.argmax())
                        if sh_kind(nxt) != "word":
                            break
                        picks.append(nxt)
                own.append(picks)
        finally:
            model.train(was_training)
        n_words = sum(1 for p in own if sh_kind(p[0]) == "word")
        n_speech = sum(1 for p in own if sh_kind(p[0]) in ("word", "speak"))
        print(f"[p1] SH recollect {sid}: {len(own)} own ticks, "
              f"{n_speech} speech-first picks ({n_words} word-first)",
              flush=True)
        return own

    def apply_own(seq: dict, own) -> dict:
        """Overwrite input ids at emission slots with the model's own
        picks, slot by slot. Extra own pieces beyond the gold skeleton
        are dropped; unfilled slots keep gold. LABELS untouched — the
        target stays what the model SHOULD have said."""
        ids = list(seq["ids"])
        pos = seq["audio_pos"]
        for t, pad_pos in enumerate(pos):
            if t >= len(own):
                break
            end = pos[t + 1] if t + 1 < len(pos) else len(ids)
            for j, slot in enumerate(range(pad_pos + 1, end)):
                if j < len(own[t]):
                    ids[slot] = own[t][j]
        return dict(seq, ids=ids)

    sh_state: dict = {}  # sid -> [own_history, steps_since_recollect]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.jsonl", "a")
    step = 0
    t0 = time.time()
    losses = []

    def save_ckpt(step):
        # text_config.tie_word_embeddings is False -> embed and head are
        # SEPARATE tensors; both new-row deltas must be saved or the
        # trained head is lost on reload.
        ck = {"emb_delta_new_rows": emb_w[n_old:].detach().cpu(),
              "head_delta_new_rows": head_w[n_old:].detach().cpu(),
              "n_old": n_old, "steps": step, "args": vars(args)}
        if adapter is not None:
            ck["adapter_state"] = {k: v.detach().cpu()
                                   for k, v in adapter.state_dict().items()}
        # tmp + rename: a kill mid-save leaves the previous good
        # checkpoint, not a truncated file
        tmp = out / "p1_checkpoint.pt.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, out / "p1_checkpoint.pt")

    while step < args.steps:
        for sp in shards:
            shard = torch.load(sp, weights_only=False)
            sid = shard["scenario_id"]
            if sid not in by_id:
                continue
            if want is not None and sid not in want:
                continue
            seq = build_from_grid(batch / by_id[sid]["dir"], proc.tokenizer)
            if len(seq["ids"]) < 8:
                continue
            if args.self_history > 0 and \
                    random.random() < args.self_history:
                if args.sh_mode == "corrupt":
                    seq = corrupt_gold(seq, sh_kind)
                else:
                    st = sh_state.setdefault(sid, [None, args.sh_refresh])
                    if st[0] is None or st[1] >= args.sh_refresh:
                        st[0] = own_history(
                            shard["audio_feats"].to(model.device),
                            len(seq["audio_pos"]), sid)
                        st[1] = 0
                    st[1] += 1
                    seq = apply_own(seq, st[0])
            if len(seq["ids"]) < 8:
                continue
            batch_in = collate(seq, shard, model.get_input_embeddings(),
                               adapter=adapter)
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
            if args.class_cost > 1.0:
                # per-CLASS cost CE (v8): L_t = sum_j c_j p_j - c_{y_t}.
                # Rare tokens cost extra BOTH as the correct answer AND as
                # a wrong prediction anywhere — suppresses the t+30s
                # over-firing at sil positions that broke v3, while
                # keeping the up-push that taught timing.
                V = logits.shape[-1]
                logp = torch.log_softmax(logits.reshape(-1, V).float(), -1)
                y = labels.reshape(-1)
                cost = torch.ones(V, device=logits.device,
                                  dtype=logp.dtype)
                cost[rare_ids] = args.class_cost
                L = (logp.exp() * cost).sum(-1) - cost[y.clamp(min=0)]
                loss = L[y.ne(-100)].mean()
            else:
                per_tok = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    labels.reshape(-1), ignore_index=-100, reduction="none")
                w = torch.ones_like(per_tok)
                is_rare = (labels.reshape(-1).unsqueeze(-1)
                           == rare_ids.unsqueeze(0)).any(-1)
                w = w.masked_fill(is_rare, args.timing_weight)
                if args.word_weight != 1.0:
                    # v11 speak-up: word-label positions (regular-vocab
                    # gold, i.e. < n_old) vs the 95%-silence background.
                    # Disjoint from rare positions by construction.
                    y = labels.reshape(-1)
                    is_word = y.ne(-100) & y.lt(n_old)
                    w = w.masked_fill(is_word, args.word_weight)
                loss = (per_tok * w).sum() / w.sum()
                if args.thief_penalty > 0:
                    # v9 down-push: how much probability the model puts on
                    # rare tokens at positions whose label is NOT rare
                    # (thief territory). logsumexp form avoids building the
                    # full (N, V) probability matrix.
                    V = logits.shape[-1]
                    z = logits.reshape(-1, V).float()
                    logZ = torch.logsumexp(z, -1)
                    zr = z[:, rare_ids]
                    rare_mass = torch.exp(torch.logsumexp(zr, -1) - logZ)
                    y = labels.reshape(-1)
                    y_rare = (y.unsqueeze(-1)
                              == rare_ids.unsqueeze(0)).any(-1)
                    valid = y.ne(-100)
                    pen = (rare_mass * (~y_rare).float()
                           * valid.float()).sum() / valid.sum()
                    loss = loss + args.thief_penalty * pen
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
            if args.save_every and step % args.save_every == 0:
                save_ckpt(step)
                print(f"[p1] checkpoint saved @ step {step}", flush=True)
            if step >= args.steps:
                break

    save_ckpt(step)
    print(f"[p1] done: {step} steps, final loss {losses[-1]:.4f} -> {out}")


if __name__ == "__main__":
    main()
