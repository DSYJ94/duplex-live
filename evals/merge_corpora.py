"""Merge the verified corpora into ONE training view (2026-10-11, the
full-speed restructure: spare compute, parallel arms, 5-day buffer).

Plain words: we have four verified batches but their lesson names
collide (every batch has its own dense_dialog-001). Training reads ONE
shard dir and ONE batch manifest, so this script stitches them
together with a batch prefix (a1_/a2_/a3_/q2_) on every lesson name —
rewriting the id inside each shard, symlinking shard files into one
dir, and concatenating manifests + quality records with the same
prefix. The fail-closed gate then sees one coherent batch.

Run (cluster, CPU container is fine):
  PYTHONPATH=... apptainer exec ... python -m evals.merge_corpora
"""
from __future__ import annotations

import json
from pathlib import Path

DL = Path("/projects/u6si/duplex-live/runs")
SOURCES = [  # (prefix, shards_dir, batch_audio_dir)
    ("q2", DL / "shards-scale1-q2", DL / "synth-scale1-q/audio"),
    ("a1", DL / "shards-synth-scale2-api", DL / "synth-scale2-api/audio"),
    ("a2", DL / "shards-synth-scale2-api-local", DL / "synth-scale2-api-local/audio"),
    ("a3", DL / "shards-synth-scale3-api", DL / "synth-scale3-api/audio"),
    # delegation round: APPEND mode (2026-10-12) — reuse the same merged
    # view, adding d_/t_ prefixed delegation + timed_task lessons
    ("d", DL / "shards-deleg", DL / "synth-deleg/audio"),
]
OUT_SHARDS = DL / "shards-merged"
OUT_BATCH = DL / "corpus-merged"


def main() -> None:
    import torch

    OUT_SHARDS.mkdir(parents=True, exist_ok=True)
    OUT_BATCH.mkdir(parents=True, exist_ok=True)
    n_rows = n_recs = 0
    mf = open(OUT_BATCH / "train_manifest.jsonl", "w")
    qf = open(OUT_BATCH / "quality.jsonl", "w")
    for prefix, sdir, bdir in SOURCES:
        if not sdir.is_dir():
            print(f"[merge] MISSING shards dir {sdir} — skipped")
            continue
        rows = {json.loads(l)["scenario_id"]: json.loads(l)
                for l in open(bdir / "train_manifest.jsonl")}
        recs = {json.loads(l)["scenario_id"]: json.loads(l)
                for l in open(bdir / "quality.jsonl")}
        kept = 0
        for sp in sorted(sdir.glob("*.pt")):
            sid = sp.stem
            rec = recs.get(sid)
            if rec is None or not rec.get("verified"):
                continue  # fail-closed: only verified lessons enter
            sh = torch.load(sp, weights_only=False)
            new_id = f"{prefix}_{sid}"
            sh["scenario_id"] = new_id
            torch.save(sh, OUT_SHARDS / f"{new_id}.pt")
            row = dict(rows[sid])
            row["scenario_id"] = new_id
            row["dir"] = new_id          # symlink name below
            mf.write(json.dumps(row) + "\n")
            rec["scenario_id"] = new_id
            qf.write(json.dumps(rec) + "\n")
            # symlink the audio dir so grid/timeline lookups resolve
            dst = OUT_BATCH / new_id
            if not dst.exists():
                dst.symlink_to(bdir / sid, target_is_directory=True)
            kept += 1
        print(f"[merge] {prefix}: {kept} verified lessons from {sdir.name}")
        n_rows += kept
    mf.close()
    qf.close()
    print(f"[merge] done: {n_rows} lessons -> {OUT_SHARDS} + {OUT_BATCH}")


if __name__ == "__main__":
    main()
