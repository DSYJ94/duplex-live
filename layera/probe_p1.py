"""Phase-1 probe: does the checkpoint actually know the new tokens?

Plain words: show the model lessons it NEVER trained on and, at each beat
where the answer key says something, check what it would have said. Run
each lesson twice — with the brand-new tokens as meaningless average
blobs (before) and with the trained values (after). If training worked,
the "after" run should pick the right special token (<sil>, <yield>, ...)
most of the time, and the "before" run should be near chance.

Ear features (the frozen ear's read of the sound) are computed inline —
held-out scenarios have no prepared shards.

  python -m layera.probe_p1 --batch runs/synth-big7/audio \
      --ckpt runs/p1-v1/p1_checkpoint.pt --skip 8 --limit 4
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="factory batch (grids+stems)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--skip", type=int, default=8,
                    help="manifest rows to skip (0..skip-1 = training set)")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layera.prepare import encode_audio, mix_in_slot
    from layera.sampler import build_from_grid, collate
    from tokenizer.vocab import register_into

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16", device_map="auto",
        attn_implementation="sdpa",
        experts_implementation="eager")  # same reason as train_p1
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    n_old = len(proc.tokenizer)
    register_into(proc.tokenizer)
    model.resize_token_embeddings(len(proc.tokenizer))

    # replay the exact training init: new rows start as mean-of-neighbors
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean = emb[:n_old].mean(dim=0)
        emb[n_old:] = mean.to(emb.dtype)
        head = model.lm_head.weight if hasattr(model, "lm_head") else \
            model.get_output_embeddings().weight
        head[n_old:] = mean.to(head.dtype)

    ck = torch.load(args.ckpt, weights_only=False)
    emb_after = ck["emb_delta_new_rows"].to(emb.device, emb.dtype)
    head_after = ck["head_delta_new_rows"].to(head.device, head.dtype)
    moved = float((emb_after.float() - mean.float()).norm())
    print(f"[probe] trained rows moved |v-mean| = {moved:.2f} "
          f"from the init blob (0 = nothing learned)", flush=True)

    # special-token ids -> names straight from the tokenizer
    spec_names = {}
    for i in range(n_old, len(proc.tokenizer)):
        spec_names[proc.tokenizer.convert_ids_to_tokens(i)] = i

    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    held = rows[args.skip:args.skip + args.limit]
    print(f"[probe] {len(held)} held-out scenarios "
          f"(rows {args.skip}..{args.skip + len(held) - 1}; "
          f"training set was rows 0..{args.skip - 1})", flush=True)

    report = {}
    for r in held:
        sc_dir = batch / r["dir"]
        seq = build_from_grid(sc_dir, proc.tokenizer)
        if len(seq["ids"]) < 8:
            continue
        pcm = mix_in_slot(sc_dir, len(seq["audio_pos"]))
        feats, _counts = encode_audio(model, proc, pcm)
        shard = {"audio_feats": feats}

        res = {}
        for tag in ("before", "after"):
            with torch.no_grad():
                if tag == "after":
                    emb[n_old:] = emb_after
                    head[n_old:] = head_after
                else:
                    emb[n_old:] = mean.to(emb.dtype)
                    head[n_old:] = mean.to(head.dtype)
                batch_in = collate(seq, shard, model.get_input_embeddings())
                batch_in = {k: v.to(model.device) for k, v in batch_in.items()}
                logits = model(**batch_in, use_cache=False).logits[0].float()

            labels = torch.tensor(seq["labels"], device=logits.device)
            id2name = {v: k for k, v in spec_names.items()}
            per_type = defaultdict(lambda: [0, 0, 0.0])  # n, correct, prob
            for t, lab in enumerate(labels):
                if int(lab) == -100:
                    continue
                pred = logits[t - 1]
                probs = torch.softmax(pred, -1)
                kind = id2name.get(int(lab), "<bpe-word>")
                e = per_type[kind]
                e[0] += 1
                e[1] += int(int(pred.argmax()) == int(lab))
                e[2] += float(probs[int(lab)])
            res[tag] = {k: {"n": v[0], "acc": round(v[1] / v[0], 3),
                            "mean_prob": round(v[2] / v[0], 3)}
                        for k, v in sorted(per_type.items())}
        report[seq["scenario_id"]] = res
        print(f"[probe] {seq['scenario_id']}:", flush=True)
        for kind in sorted({k for t in res.values() for k in t}):
            b = res["before"].get(kind, {"acc": 0, "mean_prob": 0, "n": 0})
            a = res["after"].get(kind, {"acc": 0, "mean_prob": 0, "n": 0})
            print(f"    {kind:12s} n={a.get('n', b.get('n', 0)):3d}  "
                  f"acc {b['acc']:5.3f} -> {a['acc']:5.3f}   "
                  f"prob {b['mean_prob']:5.3f} -> {a['mean_prob']:5.3f}",
                  flush=True)

    out = Path(args.out) if args.out else Path(args.ckpt).parent / "probe.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"[probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
