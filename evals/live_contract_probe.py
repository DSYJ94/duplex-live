"""Plain words: run the real microphone-loop scheduler with numbered audio ticks.

No GPU or model inference: the fake ear returns each input tick's number.
This measures scheduling, NOT recognition quality or attention lookahead.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np


def probe() -> dict:
    from m2 import live

    decisions = []

    class Feature:
        def __init__(self, source_tick=None):
            self.source_tick = source_tick

        def to(self, *args):
            return self

        def __getitem__(self, item):
            return self

    class Embedding:
        def __call__(self, ids):
            return Feature()

    class Decoder:
        n_old = 1
        name_of = {1: "<sil>"}
        gate_threshold = 0.1

        def __init__(self, *args, **kwargs):
            pass

        def tick(self, feat, cache, **kwargs):
            decisions.append({"decision_tick": len(decisions), "source_tick": feat.source_tick})
            return 1, [], None

    model = types.SimpleNamespace(
        device="cpu", get_input_embeddings=lambda: Embedding(),
        lm_head=types.SimpleNamespace(register_forward_pre_hook=lambda hook: None))
    proc = types.SimpleNamespace(tokenizer=types.SimpleNamespace(convert_tokens_to_ids=lambda s: 0))
    parity = types.ModuleType("m2.parity")
    parity.Decoder = Decoder
    parity.load_model = lambda path: (model, proc, None, 1)
    prepare = types.ModuleType("layera.prepare")

    def encode(model, proc, pcm):
        return [Feature(int(pcm[i])) for i in range(0, len(pcm), live.PER_TICK)], []

    prepare.encode_audio = encode
    torch = types.ModuleType("torch")
    torch.tensor = lambda x, **kwargs: np.asarray(x)
    log = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scenario = root / "probe"
        scenario.mkdir()
        (scenario / "timeline.actual.json").write_text(json.dumps({
            "scenario_id": "probe", "scenario_class": "probe", "events": [
                {"ch": "user", "start_ms": 0, "text": "clock fixture",
                 "actual_start_ms": 0, "actual_end_ms": 8000}]}))
        # Real main reads this function; numbered ticks replace PCM amplitudes.
        def numbered_audio(path, n_samples):
            return np.repeat(np.arange(n_samples // live.PER_TICK, dtype=np.float32), live.PER_TICK)

        argv = ["m2.live", "--ckpt", "fixture", "--batch", str(root),
                "--scenarios", "probe", "--out", str(root / "output")]
        with patch.dict("sys.modules", {"torch": torch, "m2.parity": parity,
                                        "layera.prepare": prepare}), \
             patch("sys.argv", argv), patch.object(live, "load_stem_mono", numbered_audio), \
             contextlib.redirect_stdout(log):
            live.main()
    with_features = [d for d in decisions if d["source_tick"] is not None]
    ages = [(d["decision_tick"] - d["source_tick"]) * live.GRID_MS for d in with_features]
    return {"evidence": "synthetic control-flow probe of actual m2.live.main; no model inference",
            "tick_ms": live.GRID_MS, "first_feature_decision_tick": with_features[0]["decision_tick"],
            "first_feature_source_tick": with_features[0]["source_tick"],
            "feature_age_min_ms": min(ages), "feature_age_max_ms": max(ages),
            "decisions_without_audio_features": len(decisions) - len(with_features),
            "total_decisions": len(decisions),
            "quality_stamp_absent_but_loop_ran": True,
            "note": "Age excludes encoding, inference, transport and playback time."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-feature-age-ms", type=int, default=200)
    args = ap.parse_args()
    report = probe()
    report["within_age_budget"] = report["feature_age_max_ms"] <= args.max_feature_age_ms
    serialized = json.dumps(report, indent=2) + "\n"
    print(serialized, end="")
    if args.out:
        args.out.write_text(serialized)
    return 0 if report["within_age_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
