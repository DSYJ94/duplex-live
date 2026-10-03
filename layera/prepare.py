"""Phase-1 prep: factory scenario -> training shard (docs/09 §3 step 1).

Plain words: turns one lesson (a factory scenario) into the two things the
model must see: (a) the stream of sound-tokens for what it HEARD — the
user+background mix, encoded by the model's own ear (AuT audio tower) —
and (b) the answer key — per tick, what it should have EMITTED (a word,
<sil>, <yield>, <delegate>, <t+Ns>) from the tokenizer grid.

Output: one .pt shard per scenario: {audio_feats, audio_token_ids,
emission_ids, meta}. Thinker-level supervision only in Phase-1 (the
Talker already voices text; codec targets are a later phase).

Run on Isambard (prepare.slurm):
  python -m layera.prepare --batch runs/synth-big7 --out runs/shards
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

GRID_MS = 80


def load_stem(p: Path, sr: int = 16000) -> np.ndarray:
    import librosa
    import soundfile as sf

    x, s = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if s != sr:
        x = librosa.resample(x, orig_sr=s, target_sr=sr)
    return x


def mix_in_slot(sc_dir: Path, n_ticks: int) -> np.ndarray:
    """In-slot audio: user + background stems mixed on the tick grid
    (background rides the input stream — T5 addressee learning)."""
    sr = 16000
    per = int(sr * GRID_MS / 1000)
    buf = np.zeros(n_ticks * per, dtype=np.float32)
    for ch in ("user", "background"):
        p = sc_dir / f"stem_{ch}.wav"
        if not p.exists():
            continue
        x = load_stem(p)
        buf[:min(len(buf), len(x))] += x[:len(buf)]
    peak = float(np.max(np.abs(buf))) or 1.0
    if peak > 0.95:
        buf *= 0.95 / peak
    return buf


def encode_audio(model, proc, pcm: np.ndarray, window_s: float = 4.0):
    """Run the ear (AuT tower) over the in-slot stream in windows; returns
    TICK-ALIGNED tower outputs (what the thinker scatters into
    <|audio_pad|> positions — verified in its forward). Phase-1 v1 keeps
    the ear frozen, so precomputed outputs are the training inputs.

    EAR-CLOCK FIX (2026-10-03): the tower emits ~52 tokens per 4 s
    window while the tick grid holds 50 — every consumer (sampler,
    free-run, live loop) feeds one token per tick, so the surplus was
    silently truncated: a 4% skew, up to 160 ms of within-window
    misalignment, drifting onset information across the history. Here,
    and only here, each window's tokens are resampled onto the tick
    grid by nearest-time gather — tokens map to their nearest tick's
    time centre; ~2 tokens per window are dropped/kept by proximity
    instead of losing the window tail. pcm is tick-quantized
    (n_ticks*1280 samples), so ticks-per-window is exact.
    """
    n = int(16000 * window_s)
    per_tick = int(16000 * GRID_MS / 1000)  # 1280 samples
    feats, counts = [], []
    for i in range(0, len(pcm), n):
        chunk = pcm[i:i + n]
        if len(chunk) < 800:  # <50ms tail: skip
            break
        inputs = proc(text=["<|audio_pad|>"], audio=[chunk],
                      return_tensors="pt", padding=True).to(model.device)
        n_tok = int((inputs["input_ids"][0]
                     == proc.tokenizer.convert_tokens_to_ids("<|audio_pad|>")).sum())
        with torch.no_grad():
            tower = model.get_audio_features(
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                return_dict=True).last_hidden_state
        # the mask path returns flat [tokens, hidden]; reshape handles both
        tower = tower.reshape(-1, tower.shape[-1])
        thinker_hidden = model.get_input_embeddings().weight.shape[1]
        assert tower.shape[1] == thinker_hidden, \
            f"tower hidden {tower.shape[1]} != thinker {thinker_hidden}"
        toks = tower[-n_tok:]
        n_ticks_win = len(chunk) // per_tick
        if n_tok != n_ticks_win:
            # nearest-time gather: tick t takes the token whose time
            # centre is closest
            idx = ((torch.arange(n_ticks_win, device=toks.device,
                                 dtype=torch.float32) + 0.5)
                   * (n_tok / n_ticks_win) - 0.5).round().clamp(
                       0, n_tok - 1).long()
            toks = toks[idx]
            counts.append(int(n_tok))  # keep the raw rate for the record
        else:
            counts.append(n_tok)
        feats.append(toks.float().cpu())
    return torch.cat(feats), counts


def prepare_scenario(model, proc, sc_dir: Path, tok) -> dict:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizer.grid import from_scenario_dir
    from tokenizer.vocab import ALL_SPECIAL

    grid = from_scenario_dir(sc_dir)
    n = grid.n_ticks
    pcm = mix_in_slot(sc_dir, n)
    feats, counts = encode_audio(model, proc, pcm)
    # answer key: emission per tick as token ids (specials registered ids,
    # words tokenized on the fly by the sampler)
    emis = []
    for t in grid.ticks:
        e = t.emission
        if e in ALL_SPECIAL:
            emis.append(tok.convert_tokens_to_ids(e))
        else:
            emis.append(-100)  # word tokens resolved by the sampler (BPE)
    return {"audio_feats": feats, "audio_counts": counts,
            "emission_hint": emis, "n_ticks": n,
            "scenario_id": grid.scenario_id,
            "scenario_class": grid.scenario_class}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, help="factory batch dir (audio/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=None,
                    help="N scenarios per class (class = id prefix before "
                         "the trailing number) instead of --limit total")
    args = ap.parse_args()

    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

    sys_path_hack = str(Path(__file__).resolve().parents[1])
    import sys as _s
    _s.path.insert(0, sys_path_hack)
    from tokenizer.vocab import register_into

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", dtype="bfloat16", device_map="auto",
        attn_implementation="sdpa", experts_implementation="batched_mm").eval()
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    added = register_into(proc.tokenizer)
    print(f"[prep] registered {added} duplex token types "
          f"(first id: {proc.tokenizer.convert_tokens_to_ids('<sil>')})",
          flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    # fail-closed data rule: shards are the answer keys — no conversation
    # without a passing quality record (per-clip ASR bar + timing audit)
    # may become one
    from data.synthetic.quality import checked_rows
    rows = checked_rows(batch, rows, role="shard preparation")
    if args.per_class:
        groups: dict[str, list] = defaultdict(list)
        for row in rows:
            cls = re.sub(r"-\d+$", "", row["scenario_id"])
            groups[cls].append(row)
        picked = []
        for cls in sorted(groups):
            picked.extend(groups[cls][:args.per_class])
        print(f"[prep] per-class {args.per_class}: {len(picked)} scenarios "
              f"across {len(groups)} classes", flush=True)
    else:
        picked = rows[:args.limit]
    for row in picked:
        sc_dir = batch / row["dir"]
        # resume: skip scenarios already sharded, so a timed-out job
        # resumes on resubmit (same pattern as the render clip cache)
        if (out / f"{row['scenario_id']}.pt").exists():
            print(f"[prep] {row['scenario_id']}: shard exists, skipping",
                  flush=True)
            continue
        shard = prepare_scenario(model, proc, sc_dir, proc.tokenizer)
        torch.save(shard, out / f"{shard['scenario_id']}.pt")
        print(f"[prep] {shard['scenario_id']}: {shard['n_ticks']} ticks, "
              f"{shard['audio_feats'].shape[0]} audio tokens", flush=True)


if __name__ == "__main__":
    main()
