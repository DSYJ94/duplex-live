"""DuplexThinker: a Phase-1 checkpoint behind a live tick loop.

M1's TickDriver proved audio can enter DURING generation (append-only KV).
This is that loop with the two Phase-1 upgrades: the trained duplex-token
rows are loaded from a checkpoint, and each beat's emission is the
MODEL's decision instead of a scripted endpoint — <sil> keeps it quiet,
word pieces speak, <yield> cuts speech when interrupted.

Voice-out (talker/code2wav) is M2: speaking beats are represented by a
synthetic tone upstream (evals/harness LayerATarget) so energy metrics
measure WHEN the model talks, not what it sounds like. Ticks are not
real-time paced here — the harness replays offline; per-tick wall time is
logged so serving-phase latency work stays measurable.

  python -c "from layera.duplex import DuplexThinker; \
             DuplexThinker('runs/p1-v3/p1_checkpoint.pt')"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

TICK_MS = 80
SR_IN = 16000
TICK_SAMPLES = int(SR_IN * TICK_MS / 1000)  # 1280


class DuplexThinker:
    def __init__(self, ckpt_path: str,
                 model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"):
        import torch
        from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tokenizer.vocab import register_into

        t0 = time.time()
        # batched_mm experts for INFERENCE (eager is the training-only
        # memory fix — see train_p1.py); append-only KV like M1
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_id, dtype="bfloat16", device_map="auto",
            attn_implementation="sdpa", experts_implementation="batched_mm",
        ).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.n_old = len(self.proc.tokenizer)
        register_into(self.proc.tokenizer)
        self.model.resize_token_embeddings(len(self.proc.tokenizer))

        # replay training init, then the checkpoint's two deltas
        # (text_config.tie_word_embeddings=False -> embed and head are
        # separate tensors; both must be restored)
        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight
            mean = emb[:self.n_old].mean(dim=0)
            emb[self.n_old:] = mean.to(emb.dtype)
            head = self.model.lm_head.weight
            head[self.n_old:] = mean.to(head.dtype)
        ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        with torch.no_grad():
            emb[self.n_old:] = ck["emb_delta_new_rows"].to(emb.device, emb.dtype)
            head[self.n_old:] = ck["head_delta_new_rows"].to(head.device,
                                                             head.dtype)

        # chat-envelope specials only — the snapshot is taken AFTER
        # registration, so filter OUT the duplex ids: they must stay
        # decodable (M1 masked all specials; M1 predates trained rows)
        duplex_ids = set(range(self.n_old, len(self.proc.tokenizer)))
        self.mask_ids = [i for i in self.proc.tokenizer.all_special_ids
                         if i not in duplex_ids]
        self.name_of = {i: self.proc.tokenizer.convert_ids_to_tokens(i)
                        for i in duplex_ids}
        self.cache = None
        self.pending_logits = None
        print(f"[duplex] {ckpt_path} loaded ({time.time() - t0:.0f}s), "
              f"{len(duplex_ids)} duplex tokens live", flush=True)

    def kind_of(self, tok_id: int) -> str:
        if tok_id < self.n_old:
            return "word"
        name = self.name_of[tok_id]
        if name == "<sil>":
            return "sil"
        if name == "<yield>":
            return "yield"
        if name.startswith("<t+"):
            return "timing"
        return "control"  # <delegate>/<barge-in>: logged; M2 acts on them

    def feed_tick(self, pcm_16k: np.ndarray) -> float:
        """One 80 ms slice of user audio into the running memory."""
        import torch

        t0 = time.time()
        if self.cache is None:
            convo = [{"role": "user", "content": [
                {"type": "audio"},
                {"type": "text", "text": "(listening)"},
            ]}]
            text = self.proc.apply_chat_template(
                convo, add_generation_prompt=False, tokenize=False)
        else:
            text = "<|audio|>"  # bare placeholder: append-only beats
        inputs = self.proc(text=[text], audio=[pcm_16k],
                           return_tensors="pt", padding=True
                           ).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, past_key_values=self.cache,
                             use_cache=True)
        self.cache = out.past_key_values
        self.pending_logits = out.logits
        return time.time() - t0

    def decide(self) -> tuple[int, float]:
        """One emission beat: the model's own choice. Chat-envelope
        specials are masked; the trained duplex tokens compete."""
        import torch

        t0 = time.time()
        logits = self.pending_logits[0, -1].clone()
        for tid in self.mask_ids:
            logits[tid] = float("-inf")
        tok = int(logits.argmax())
        step = torch.tensor([[tok]], device=self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=step, past_key_values=self.cache,
                             use_cache=True)
        self.cache = out.past_key_values
        self.pending_logits = out.logits
        return tok, time.time() - t0
