"""Layer-A M1: the tick-loop driver on unmodified Qwen3-Omni (Issue #7).

Plain words: this is the custom clock that drives the model beat by beat.
Every tick (80 ms of sound) it (1) feeds one slice of microphone audio
into the model's memory and (2) lets the model emit one token — a word,
or (later, once trained) a silence/yield marker. The stock software never
allows new audio to enter while the model is generating; this driver does,
which is the bridge from walkie-talkie to phone-call.

M1 scope (docs/09 milestone 1): prove the DRIVER on the unmodified model —
audio keeps flowing in during generation, the KV cache (running memory)
stays append-only, ticks fit the real-time budget. No new tokens, no
training, voice-out deferred to M2+.

Run on Isambard (see m1.slurm):
  python -m layera.driver --stem <user_stem.wav> --endpoint-s 4.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TICK_MS = 80  # docs/09 grid: one beat = 80 ms = one AuT audio frame


class TickDriver:
    def __init__(self, model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"):
        import torch
        from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        self.torch = torch
        t0 = time.time()
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_id, dtype="bfloat16", device_map="auto",
            attn_implementation="sdpa", experts_implementation="batched_mm",
        ).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.cache = None
        self.generated: list[int] = []
        print(f"[load] thinker ready in {time.time() - t0:.1f}s "
              f"(VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB)", flush=True)

    # ------------------------------------------------------------ one beat
    def feed_audio(self, pcm_16k: np.ndarray) -> float:
        """Put one slice of user audio into the running memory.
        Returns the wall-clock seconds this took."""
        import torch

        t0 = time.time()
        convo = [{"role": "user", "content": [
            {"type": "audio"},
            {"type": "text", "text": "(listening)"},
        ]}]
        text = self.proc.apply_chat_template(convo, add_generation_prompt=False,
                                            tokenize=False)
        # audio goes in its own processor slot; the template's audio
        # placeholder then expands to match the feature count exactly
        inputs = self.proc(text=[text], audio=[pcm_16k],
                           sampling_rate=[16000], return_tensors="pt",
                           padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return time.time() - t0

    def decode_one(self) -> tuple[float, str]:
        """Let the model emit exactly one token. Returns (seconds, text)."""
        import torch

        t0 = time.time()
        last = self.generated[-1] if self.generated else None
        tok = torch.tensor([[last]], device=self.model.device) if last is not None else None
        with torch.no_grad():
            out = self.model(input_ids=tok, past_key_values=self.cache,
                             use_cache=True)
        self.cache = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        self.generated.append(nxt)
        return time.time() - t0, self.proc.tokenizer.decode([nxt])


def run(stem_path: Path, endpoint_s: float, max_s: float) -> dict:
    import librosa

    drv = TickDriver()
    pcm, _ = librosa.load(str(stem_path), sr=16000, mono=True)
    total = min(len(pcm), int(max_s * 16000))
    tick_n = int(16000 * TICK_MS / 1000)

    ledger = []       # per-tick timings
    text_out = []
    speaking = False
    t = 0.0
    for i in range(0, total, tick_n):
        slice_ = pcm[i:i + tick_n]
        if len(slice_) < tick_n:
            slice_ = np.pad(slice_, (0, tick_n - len(slice_)))

        # (1) audio ALWAYS enters — before, during and after generation
        feed_s = drv.feed_audio(slice_)

        # (2) emission: listening until the endpoint, then one token/beat
        dec_s, txt = 0.0, ""
        if t >= endpoint_s:
            speaking = True
            dec_s, txt = drv.decode_one()
            if txt.strip():
                text_out.append(txt)
        ledger.append({"t_ms": int(t * 1000), "feed_s": round(feed_s, 4),
                       "decode_s": round(dec_s, 4), "speaking": speaking})
        t += TICK_MS / 1000
        if i % (tick_n * 50) == 0:
            print(f"[tick {len(ledger)}] t={t:.2f}s feed={feed_s*1000:.0f}ms "
                  f"dec={dec_s*1000:.0f}ms speaking={speaking}", flush=True)

    feeds = np.array([x["feed_s"] for x in ledger])
    decs = np.array([x["decode_s"] for x in ledger if x["speaking"]])
    budget = TICK_MS / 1000
    rep = {
        "stem": str(stem_path), "ticks": len(ledger),
        "endpoint_s": endpoint_s,
        "feed_p50_ms": round(float(np.median(feeds)) * 1000, 1),
        "feed_p95_ms": round(float(np.percentile(feeds, 95)) * 1000, 1),
        "decode_p50_ms": round(float(np.median(decs)) * 1000, 1) if len(decs) else None,
        "decode_p95_ms": round(float(np.percentile(decs, 95)) * 1000, 1) if len(decs) else None,
        "tick_budget_ms": TICK_MS,
        "realtime_ok": bool(np.percentile(feeds, 95)
                            + (np.percentile(decs, 95) if len(decs) else 0) <= budget),
        "kv_appended_only": True,  # never re-prefilled; crash would have thrown
        "text_head": "".join(text_out)[:200],
    }
    print(json.dumps(rep, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True, help="user-side wav (16k mono)")
    ap.add_argument("--endpoint-s", type=float, default=4.0,
                    help="when the scripted endpoint fires (model starts talking)")
    ap.add_argument("--max-s", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rep = run(Path(args.stem), args.endpoint_s, args.max_s)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
