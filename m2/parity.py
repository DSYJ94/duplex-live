"""M2 milestone 1 — real-time tick parity (docs/04 shadow test).

Plain words: the live system is a clock that ticks every 80 ms; on each
tick it feeds the next ear feature to the model and reads out one
decision (word / <sil> / <speak> / <yield> / <t+Ns>). Nothing about
that clock may change WHAT the model says — only how fast it says it.
This runner proves it on a real GPU:

  1. DRY pass  — the exact offline free-run decode (one decision per
     ear feature, word continuation pieces fed back, first non-word
     breaker dropped), as fast as the GPU allows. This is the
     reference: the same computation every probe has been scored with.
  2. PACE pass — the SAME decode, but tick i becomes DUE at
     t0 + (i+1)*80 ms (the decision for an 80 ms audio window is due
     when the window closes). If compute finishes after the NEXT
     tick's due time, the stream has slipped: the next tick starts
     late, and that lateness cascades — exactly what a real session
     would suffer, so it is measured, not hidden.

It then reports, per lesson:
  - schedule adherence (docs/04 SLO: 99.9% of micro-turns on time) —
    a tick is LATE when it starts >10 ms after its due time (10 ms
    grace covers OS scheduler jitter; cascading slips blow far past it)
  - decision latency once started (p50/p95/p999 — tails are
    first-class: a p95 event fires several times per minute when
    inferring continuously)
  - PARITY — first picks must match the dry pass on every tick
    (gate: 0 mismatches). Continuation pieces must finish inside
    their tick's budget, so pieces are timed as part of the tick.

Ear features come from the training shards (shard["audio_feats"]) —
the exact rows self-history training consumed. Mic capture and
feature-encoding latency are milestone 2; this milestone isolates the
model loop on the clock.

  python -m m2.parity --ckpt runs/dense-pt1/p1_checkpoint.pt \
      --shards runs/shards-dense2 \
      --scenarios dense_dialog-001,dense_dialog-004,... \
      --out runs/m2-parity-dense-pt1

Gates (exit 1 on failure): adherence >= 99.9%, parity mismatches = 0,
p999 decision latency < 80 ms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer.vocab import register_into  # noqa: E402

TICK_S = 0.080  # the Layer-A contract (docs/09): one decision per 80 ms
LATE_GRACE_MS = 10.0  # scheduler jitter allowance; slips exceed it widely


def load_model(ckpt_path: Path, experts: str = "batched_mm"):
    """Base model + Phase-1 checkpoint deltas — mirrors
    layera/freerun_diag.py exactly (mean init then delta overwrite,
    optional audio adapter). experts picks the MoE dispatch kernel:
    batched_mm (grouped GEMM — the probe/free-run default) vs eager
    (loop only the ~8 active experts — docs/04's gather+GEMV advice
    for batch-1 decode). Different kernels = slightly different
    numerics = occasional argmax flips, so cross-expert divergence is
    REPORTED (--ref-picks), never assumed zero."""
    import torch
    from transformers import (AutoProcessor,
                              Qwen3OmniMoeThinkerForConditionalGeneration)

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16",
        device_map="auto", attn_implementation="sdpa",
        experts_implementation=experts).eval()
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
    ck = torch.load(ckpt_path, weights_only=False)
    with torch.no_grad():
        emb[n_old:] = ck["emb_delta_new_rows"].to(emb.device, emb.dtype)
        head[n_old:] = ck["head_delta_new_rows"].to(head.device, head.dtype)
    adapter = None
    if ck.get("adapter_state"):
        ad_dim = int(ck.get("args", {}).get("adapter_dim", 256))
        hid = emb.shape[1]
        adapter = torch.nn.Sequential(
            torch.nn.Linear(hid, ad_dim), torch.nn.GELU(),
            torch.nn.Linear(ad_dim, hid))
        adapter.load_state_dict(ck["adapter_state"])
        adapter = adapter.to(model.device).eval()
    return model, proc, adapter, n_old


class Decoder:
    """The tick loop's brain: everything per-decision that live and the
    free-run must share to stay in lockstep.

    gate/cap mirror freerun_diag's serving decode (M2.2a): when a gate
    checkpoint is loaded, a word-first pick counts only if the activity
    gate agrees — queried on the SAME hidden state the word head used
    (the lm_head pre-hook `cap`), same precedence: activity block, then
    the per-tick user-speech mask. The gate is a deterministic function
    of the tick state, so dry and paced passes stay in lockstep and the
    parity gate keeps its meaning; the MLP cost lands in compute_ms,
    which is the point (latency truth for the config we would serve)."""

    def __init__(self, model, proc, adapter, max_pieces: int, n_old: int,
                 gate=None, gate_threshold: float = 0.5, cap=None,
                 piece_mode: str = "inline", pieces_per_tick: int = 4,
                 user_cut: bool = False):
        import torch
        self.torch = torch
        self.model = model
        self.adapter = adapter
        self.max_pieces = max_pieces
        # n_old = the PRE-registration vocab size (captured in
        # load_model before register_into) — recomputing it here would
        # count the duplex tokens as old-vocab words and misroute
        # <sil>/<speak> into the word-continuation path
        self.n_old = n_old
        self.name_of = {i: proc.tokenizer.convert_ids_to_tokens(i)
                        for i in range(self.n_old, len(proc.tokenizer))}
        self.mask_ids = [i for i in proc.tokenizer.all_special_ids
                         if i < self.n_old]
        self.pad_id = proc.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        self.emb = model.get_input_embeddings()
        self.gate = gate
        self.gate_threshold = gate_threshold
        self.cap = cap
        self.sil_id = next((i for i, n in self.name_of.items()
                            if n == "<sil>"), None)
        self.n_gate_block = 0
        # Milestone 2b (piece-budget scheduling): "inline" is the
        # M2.2a contract — a word pick runs its whole continuation
        # inside the same tick (measured: speaking ticks p50 417 ms,
        # nowhere near 80 ms). "arrears" makes the ear-frame DECISION
        # the only latency-critical work of a tick: the model's own
        # word tokens are owed to the cache and drain on the following
        # ticks, pieces_per_tick forwards per tick, BEFORE the next ear
        # frame (so whenever the backlog fits the budget, the cache
        # order is exactly the inline order and the picks match; runs
        # longer than the budget interleave with ear frames — the
        # divergence vs the inline reference is printed). The drain
        # rule is deliberately NOT deadline-aware: dry and paced passes
        # must take the same path or the parity gate loses meaning.
        # user_cut (arrears only): a per-piece user-speech check — when
        # the user starts talking, the remaining pieces are dropped
        # (the model is cut off mid-word) instead of finishing the
        # word.
        if piece_mode not in ("inline", "arrears"):
            raise ValueError(f"unknown piece_mode {piece_mode!r}")
        self.piece_mode = piece_mode
        self.pieces_per_tick = pieces_per_tick
        self.user_cut = user_cut
        self.pending: list[int] = []   # word tokens owed to the cache
        self.n_dropped_user = 0        # pieces dropped by a barge-in cut
        self.n_user_cuts = 0

    def kind_of(self, tok: int) -> str:
        if tok >= self.n_old:
            name = self.name_of[tok]
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

    def clean_argmax(self, logits) -> int:
        torch = self.torch
        for tid in self.mask_ids:
            logits[tid] = float("-inf")
        return int(logits.argmax())

    def _feed(self, cache, tok: int):
        """One forward that inserts an already-decided word token;
        returns (cache, next pick)."""
        e = self.emb(self.torch.tensor(
            [[tok]], device=self.model.device))
        out = self.model(inputs_embeds=e, past_key_values=cache,
                         use_cache=True)
        return out.past_key_values, \
            int(self.clean_argmax(out.logits[0, -1].clone()))

    def tick(self, feat, cache, user_blocked: bool = False):
        """One tick: ear feature in, decision + continuation pieces out.
        Returns (pick, n_pieces, cache). Body mirrors freerun_diag
        run_lesson — the first non-word breaker is discarded, and with
        the serving decode the word-first pick must pass the activity
        gate and clear the user-speech mask (a word already in flight
        always completes, same as live — except in arrears mode with
        user_cut, where the barge-in cuts the word instead)."""
        torch = self.torch
        if self.piece_mode == "arrears":
            return self._tick_arrears(feat, cache, user_blocked)
        e_in = self.emb(torch.tensor(
            [[self.pad_id]], device=self.model.device)).clone()
        e_in[0, 0] = feat.to(e_in.dtype)
        with torch.no_grad():
            out = self.model(inputs_embeds=e_in, past_key_values=cache,
                             use_cache=True)
        cache = out.past_key_values
        pick = self.clean_argmax(out.logits[0, -1].clone())
        if pick < self.n_old and self.gate is not None:
            p = float(torch.sigmoid(self.gate(
                self.cap["h"][0, -1].detach().unsqueeze(0))[0]))
            if p < self.gate_threshold:
                pick = self.sil_id
                self.n_gate_block += 1
            elif user_blocked:
                pick = self.sil_id
        pieces = 0
        tok = pick
        while self.kind_of(tok) == "word" and pieces + 1 < self.max_pieces:
            cache, nxt = self._feed(cache, tok)
            if self.kind_of(nxt) != "word":
                break
            pieces += 1
            tok = nxt
        return pick, pieces, cache

    def _tick_arrears(self, feat, cache, user_blocked: bool):
        """The 2b tick. 1) drain owed word tokens (fixed budget, the
        same rule every tick so dry == paced), each insertion cut short
        if the user started talking and user_cut is on; 2) ear frame
        in, decision out — a word pick is QUEUED, not run, so the tick
        costs one forward plus at most pieces_per_tick drains."""
        torch = self.torch
        n_drain = 0
        while self.pending and n_drain < self.pieces_per_tick:
            if user_blocked and self.user_cut:
                self.n_dropped_user += len(self.pending)
                self.n_user_cuts += 1
                self.pending = []
                break
            cache, nxt = self._feed(cache, self.pending.pop(0))
            n_drain += 1
            if self.kind_of(nxt) == "word":
                self.pending.append(nxt)
        e_in = self.emb(torch.tensor(
            [[self.pad_id]], device=self.model.device)).clone()
        e_in[0, 0] = feat.to(e_in.dtype)
        with torch.no_grad():
            out = self.model(inputs_embeds=e_in, past_key_values=cache,
                             use_cache=True)
        cache = out.past_key_values
        pick = self.clean_argmax(out.logits[0, -1].clone())
        if pick < self.n_old and self.gate is not None:
            p = float(torch.sigmoid(self.gate(
                self.cap["h"][0, -1].detach().unsqueeze(0))[0]))
            if p < self.gate_threshold:
                pick = self.sil_id
                self.n_gate_block += 1
            elif user_blocked:
                pick = self.sil_id
        if self.kind_of(pick) == "word":
            self.pending.append(pick)
        return pick, n_drain, cache


def run_pass(dec: Decoder, feats, paced: bool, tick_s: float,
             blocked=None):
    """One full lesson. paced=False: as fast as possible (the dry
    reference). paced=True: tick i starts no earlier than its due time
    t0 + (i+1)*tick_s; a tick that overruns pushes the next one late
    and the slip cascades. blocked (optional): per-tick user-speech
    mask, same source as the free-run. Returns (first_picks, records);
    per tick: {tick, compute_ms, n_pieces[, start_delay_ms, late]}."""
    t0 = time.monotonic()
    # per-pass decoder state reset: pending pieces and cut counters
    # must start clean or the dry pass leaks its queue into the paced
    # pass (and the second scenario inherits the first's)
    dec.pending = []
    dec.n_dropped_user = 0
    dec.n_user_cuts = 0
    picks, records, cache = [], [], None
    for i in range(len(feats)):
        deadline = t0 + (i + 1) * tick_s
        start_delay_ms = 0.0
        if paced:
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)
            start_delay_ms = (time.monotonic() - deadline) * 1000
        ub = bool(blocked[i]) if blocked is not None and i < len(blocked) \
            else False
        c0 = time.monotonic()
        pick, n_pieces, cache = dec.tick(feats[i], cache, user_blocked=ub)
        c1 = time.monotonic()
        rec = {"tick": i, "compute_ms": round((c1 - c0) * 1000, 3),
               "n_pieces": n_pieces,
               "backlog": len(getattr(dec, "pending", []))}
        if paced:
            rec["start_delay_ms"] = round(start_delay_ms, 3)
            rec["late"] = start_delay_ms > LATE_GRACE_MS
        picks.append(pick)
        records.append(rec)
    return picks, records


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--scenarios", required=True,
                    help="comma-separated scenario_ids (must be in the "
                         "shard dir)")
    ap.add_argument("--tick-ms", type=float, default=TICK_S * 1000)
    ap.add_argument("--max-pieces", type=int, default=8)
    ap.add_argument("--experts", default="batched_mm",
                    choices=["batched_mm", "eager"],
                    help="MoE dispatch kernel at batch-1 decode (m2 "
                         "latency lever; docs/04 favours gather+GEMV "
                         "= eager for small batches)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model forward (dynamic "
                         "shapes; graph breaks fall back to eager per "
                         "op — the win, if any, is measured not "
                         "assumed)")
    ap.add_argument("--save-picks", default=None,
                    help="write first-pick id lists to <dir>/picks.json "
                         "(the cross-config reference)")
    ap.add_argument("--ref-picks", default=None,
                    help="picks.json from another kernel config — "
                         "reports per-tick divergence against it "
                         "(information only; the parity GATE stays "
                         "dry-vs-paced under THIS config)")
    ap.add_argument("--gate-ckpt", default=None,
                    help="aux activity gate checkpoint (M2.2a: parity "
                         "on the SERVING decode — a word-first pick "
                         "counts only if the gate agrees, exactly as "
                         "freerun_diag --gate-ckpt)")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--user-gate", action="store_true",
                    help="stopgap user-speech mask from the scenario "
                         "pcm (needs --batch), same user_block_mask as "
                         "the free-run")
    ap.add_argument("--gate-hold", type=int, default=3)
    ap.add_argument("--piece-mode", default="inline",
                    choices=["inline", "arrears"],
                    help="inline: continuation pieces run inside the "
                         "picking tick (M2.2a contract; speaking ticks "
                         "measured 417 ms). arrears: the milestone-2b "
                         "scheduler — the ear-frame decision is the "
                         "only latency-critical work; the model's word "
                         "tokens drain on later ticks, at most "
                         "--pieces-per-tick forwards per tick, before "
                         "the next ear frame so cache order tracks the "
                         "inline order whenever the backlog fits")
    ap.add_argument("--pieces-per-tick", type=int, default=4)
    ap.add_argument("--user-cut", action="store_true",
                    help="arrears only: a per-piece user-speech check — "
                         "the user talking drops the remaining pieces "
                         "(model cut off mid-word) instead of letting "
                         "words in flight complete")
    ap.add_argument("--batch", default=None,
                    help="scenario batch dir (train_manifest.jsonl) — "
                         "required with --user-gate")
    ap.add_argument("--out", default="runs/m2-parity")
    args = ap.parse_args()

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tick_s = args.tick_ms / 1000

    model, proc, adapter, n_old = load_model(Path(args.ckpt),
                                             experts=args.experts)
    if adapter is not None:
        print(f"[m2] audio adapter loaded "
              f"({sum(p.numel() for p in adapter.parameters())} params)",
              flush=True)
    if args.compile:
        model = torch.compile(model, dynamic=True)
        print("[m2] torch.compile ON (dynamic)", flush=True)

    # serving decode: the activity gate reads the lm_head-input hidden
    # state — same capture hook as freerun_diag / train_gate
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
        print(f"[m2] activity gate loaded from {args.gate_ckpt} "
              f"(threshold {args.gate_threshold:g})", flush=True)

    blocked_by_sid = None
    if args.user_gate:
        if not args.batch:
            raise SystemExit("[m2] --user-gate needs --batch")
        from layera.freerun_diag import user_block_mask
        from layera.prepare import mix_in_slot
        from data.synthetic.quality import checked_rows
        rows = [json.loads(l) for l in
                open(Path(args.batch) / "train_manifest.jsonl")]
        # fail-closed data rule: live-loop runs only on verified lessons
        rows = checked_rows(Path(args.batch), rows, role="m2 live loop")
        blocked_by_sid = {r["scenario_id"]: Path(args.batch) / r["dir"]
                          for r in rows}
        print(f"[m2] user-gate ON (hold {args.gate_hold}, "
              f"{len(blocked_by_sid)} scenarios in batch)", flush=True)

    dec = Decoder(model, proc, adapter, args.max_pieces, n_old,
                  gate=gate, gate_threshold=args.gate_threshold, cap=cap,
                  piece_mode=args.piece_mode,
                  pieces_per_tick=args.pieces_per_tick,
                  user_cut=args.user_cut)
    print(f"[m2] piece mode {args.piece_mode}"
          + (f" (pieces_per_tick={args.pieces_per_tick}"
             f", user_cut={args.user_cut})"
             if args.piece_mode == "arrears" else ""), flush=True)

    shard_paths = []
    for s in args.scenarios.split(","):
        s = s.strip()
        sp = Path(args.shards) / f"{s}.pt"
        if not sp.is_file():
            raise SystemExit(f"[m2] shard missing: {sp}")
        shard_paths.append(sp)

    gates = []
    all_recs = []
    dry_all: dict[str, list[int]] = {}
    for sp in shard_paths:
        shard = torch.load(sp, weights_only=False)
        sid = shard["scenario_id"]
        feats = shard["audio_feats"]
        if adapter is not None:
            feats = adapter(feats.to(next(adapter.parameters()).device,
                                     torch.float32)).to(feats.dtype)
        else:
            feats = feats.to(model.device)
        n = len(feats)

        blocked = None
        if blocked_by_sid is not None:
            sc = blocked_by_sid[sid]
            # n*1280 samples (feats run ~4% longer than the audio grid;
            # the zero tail is quiet and harmless to the RMS quantiles)
            pcm = mix_in_slot(sc, n)
            blocked = user_block_mask(pcm, len(pcm) // 1280,
                                      args.gate_hold)

        dec.n_gate_block = 0
        dec.tick(feats[0], None)  # warmup: compile + kernels untimed
        dry_picks, _ = run_pass(dec, feats, paced=False, tick_s=tick_s,
                                blocked=blocked)
        pace_picks, recs = run_pass(dec, feats, paced=True, tick_s=tick_s,
                                    blocked=blocked)

        mismatch = sum(1 for a, b in zip(dry_picks, pace_picks) if a != b)
        compute = [r["compute_ms"] for r in recs]
        late = sum(1 for r in recs if r.get("late"))
        adherence = 100.0 * (1 - late / max(n, 1))
        pieces = sum(r["n_pieces"] for r in recs)
        gate_blk = f", {dec.n_gate_block} gate blocks" \
            if gate is not None else ""
        # 2b arrears accounting: decision ticks (no drains) are the
        # latency-critical ones; backlog shows whether the piece queue
        # keeps up
        dec_ticks = [r["compute_ms"] for r in recs
                     if r["n_pieces"] == 0]
        bl = [r["backlog"] for r in recs]
        arrears = ""
        if args.piece_mode == "arrears":
            arrears = (f" | decision-only ticks {len(dec_ticks)}/{n} "
                       f"p50 {pct(dec_ticks, .5) if dec_ticks else 0:.1f}"
                       f" p95 {pct(dec_ticks, .95) if dec_ticks else 0:.1f}"
                       f" ms | backlog avg {sum(bl) / max(n, 1):.2f}"
                       f" max {max(bl)} | user cuts {dec.n_user_cuts}"
                       f" ({dec.n_dropped_user} pieces dropped)")
        print(f"[m2] {sid}: {n} ticks, {pieces} piece-feeds | "
              f"adherence {adherence:.2f}% ({late} late) | "
              f"compute p50 {pct(compute, .5):.1f} "
              f"p95 {pct(compute, .95):.1f} "
              f"p999 {pct(compute, .999):.1f} ms "
              f"max {max(compute):.1f} ms | parity mismatches {mismatch}"
              f"{gate_blk}{arrears}",
              flush=True)
        (out / f"{sid}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n")
        all_recs.append({"scenario_id": sid, "n_ticks": n,
                         "late": late, "adherence": adherence,
                         "mismatch": mismatch, "pieces": pieces,
                         "gate_blocks": dec.n_gate_block,
                         "piece_mode": args.piece_mode,
                         "pieces_per_tick": args.pieces_per_tick,
                         "backlog_max": max(bl),
                         "user_cuts": dec.n_user_cuts,
                         "p50_ms": pct(compute, .5),
                         "p95_ms": pct(compute, .95),
                         "p999_ms": pct(compute, .999),
                         "max_ms": max(compute)})
        gates.append((f"{sid} adherence>=99.9", adherence >= 99.9))
        gates.append((f"{sid} parity=0", mismatch == 0))
        gates.append((f"{sid} p999<{tick_s * 1000:.0f}ms",
                      pct(compute, .999) < tick_s * 1000))
        dry_all[sid] = dry_picks

    if args.save_picks:
        sdir = Path(args.save_picks)
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "picks.json").write_text(json.dumps(dry_all))
        print(f"[m2] dry picks saved -> {sdir / 'picks.json'}", flush=True)
    if args.ref_picks:
        ref = json.loads(Path(args.ref_picks).read_text())
        for sid, mine in dry_all.items():
            if sid not in ref:
                print(f"[m2] ref-picks: {sid} not in reference — skipped",
                      flush=True)
                continue
            theirs = ref[sid]
            m = min(len(mine), len(theirs))
            div = sum(1 for a, b in zip(mine[:m], theirs[:m]) if a != b)
            print(f"[m2] DIVERGENCE vs ref ({args.experts}"
                  f"{'+' if args.compile else ''}"
                  f"{'compile' if args.compile else ''}) {sid}: "
                  f"{div}/{m} ticks ({100 * div / max(m, 1):.2f}%) "
                  f"argmax flips vs reference kernel",
                  flush=True)

    (out / "summary.json").write_text(json.dumps(all_recs, indent=1))
    print("[m2] GATES: " + "; ".join(
        f"{'PASS' if ok else 'FAIL'}:{name}" for name, ok in gates),
          flush=True)
    if not all(ok for _, ok in gates):
        raise SystemExit(1)
    print("[m2] parity PASS — the clock does not change what the model "
          "says", flush=True)


if __name__ == "__main__":
    main()
