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
    free-run must share to stay in lockstep."""

    def __init__(self, model, proc, adapter, max_pieces: int, n_old: int):
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

    def tick(self, feat, cache):
        """One tick: ear feature in, decision + continuation pieces out.
        Returns (pick, n_pieces, cache). Body mirrors freerun_diag
        run_lesson — the first non-word breaker is discarded."""
        torch = self.torch
        e_in = self.emb(torch.tensor(
            [[self.pad_id]], device=self.model.device)).clone()
        e_in[0, 0] = feat.to(e_in.dtype)
        with torch.no_grad():
            out = self.model(inputs_embeds=e_in, past_key_values=cache,
                             use_cache=True)
        cache = out.past_key_values
        pick = self.clean_argmax(out.logits[0, -1].clone())
        pieces = 0
        tok = pick
        while self.kind_of(tok) == "word" and pieces + 1 < self.max_pieces:
            e_out = self.emb(torch.tensor([[tok]], device=self.model.device))
            with torch.no_grad():
                out = self.model(inputs_embeds=e_out,
                                 past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            nxt = self.clean_argmax(out.logits[0, -1].clone())
            if self.kind_of(nxt) != "word":
                break
            pieces += 1
            tok = nxt
        return pick, pieces, cache


def run_pass(dec: Decoder, feats, paced: bool, tick_s: float):
    """One full lesson. paced=False: as fast as possible (the dry
    reference). paced=True: tick i starts no earlier than its due time
    t0 + (i+1)*tick_s; a tick that overruns pushes the next one late
    and the slip cascades. Returns (first_picks, records); per tick:
    {tick, compute_ms, n_pieces[, start_delay_ms, late]}."""
    t0 = time.monotonic()
    picks, records, cache = [], [], None
    for i in range(len(feats)):
        deadline = t0 + (i + 1) * tick_s
        start_delay_ms = 0.0
        if paced:
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)
            start_delay_ms = (time.monotonic() - deadline) * 1000
        c0 = time.monotonic()
        pick, n_pieces, cache = dec.tick(feats[i], cache)
        c1 = time.monotonic()
        rec = {"tick": i, "compute_ms": round((c1 - c0) * 1000, 3),
               "n_pieces": n_pieces}
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
    dec = Decoder(model, proc, adapter, args.max_pieces, n_old)

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

        dec.tick(feats[0], None)  # warmup: compile + kernels untimed
        dry_picks, _ = run_pass(dec, feats, paced=False, tick_s=tick_s)
        pace_picks, recs = run_pass(dec, feats, paced=True, tick_s=tick_s)

        mismatch = sum(1 for a, b in zip(dry_picks, pace_picks) if a != b)
        compute = [r["compute_ms"] for r in recs]
        late = sum(1 for r in recs if r.get("late"))
        adherence = 100.0 * (1 - late / max(n, 1))
        pieces = sum(r["n_pieces"] for r in recs)
        print(f"[m2] {sid}: {n} ticks, {pieces} continuation pieces | "
              f"adherence {adherence:.2f}% ({late} late) | "
              f"compute p50 {pct(compute, .5):.1f} "
              f"p95 {pct(compute, .95):.1f} "
              f"p999 {pct(compute, .999):.1f} ms "
              f"max {max(compute):.1f} ms | parity mismatches {mismatch}",
              flush=True)
        (out / f"{sid}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n")
        all_recs.append({"scenario_id": sid, "n_ticks": n,
                         "late": late, "adherence": adherence,
                         "mismatch": mismatch, "pieces": pieces,
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
