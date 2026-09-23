"""DuplexThinker v2: a Phase-1 checkpoint behind a live tick loop.

v2 feeds audio EXACTLY like training. The freerun diagnostic (job
6762979) showed why that matters: in the training format the checkpoint
free-runs clean silence (sil acc 0.944 self-fed vs 0.383 teacher-forced
on held-out timed_task-017); through the old chat-template/processor
path the same checkpoint babbled old-vocab fragments 100% of the time.
Format was the bug, not the model.

The v2 contract (mirrors layera/prepare.encode_audio + sampler):
  - no chat envelope — the stream starts at position 0
  - audio enters as <|audio_pad|> embeddings overwritten by ear
    (AuT tower) features computed over 4 s windows, ~52 feats per
    window; one decision beat per feature
  - each beat: audio feature in -> emission argmax (chat-envelope
    specials masked, duplex tokens compete) -> emission embedding fed
    back into the KV

Voice-out (talker/code2wav) is M2: speaking beats are represented by a
synthetic tone upstream (evals/harness LayerATarget) so energy metrics
measure WHEN the model talks. Beats carry their audio timestamp so the
harness can place output at the right wall-clock offset despite the
windowed feature lag. Not real-time paced — the harness replays
offline; per-beat wall time is logged for serving-phase work.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

TICK_MS = 80
SR_IN = 16000
TICK_SAMPLES = int(SR_IN * TICK_MS / 1000)  # 1280
WINDOW_S = 4.0
WINDOW_SAMPLES = int(SR_IN * WINDOW_S)


class DuplexThinker:
    def __init__(self, ckpt_path: str,
                 model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                 sil_bias: float = 0.0, max_pieces: int = 8):
        import torch
        from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tokenizer.vocab import register_into

        self.sil_bias = float(sil_bias)
        self.max_pieces = int(max_pieces)

        t0 = time.time()
        # batched_mm experts for INFERENCE (eager is the training-only
        # memory fix — see train_p1.py)
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
            ck = torch.load(ckpt_path, weights_only=False,
                            map_location="cpu")
            emb[self.n_old:] = ck["emb_delta_new_rows"].to(emb.device,
                                                           emb.dtype)
            head[self.n_old:] = ck["head_delta_new_rows"].to(head.device,
                                                             head.dtype)

        # capacity-test arm (Codex F5-2): checkpoints trained with
        # --adapter carry a small trainable MLP applied to ear features
        # before they enter the stream — apply it here exactly as
        # training did
        self.adapter = None
        if ck.get("adapter_state"):
            ad_dim = int(ck.get("args", {}).get("adapter_dim", 256))
            hid = emb.shape[1]  # embedding width — no config-path guessing
            ad = torch.nn.Sequential(
                torch.nn.Linear(hid, ad_dim),
                torch.nn.GELU(),
                torch.nn.Linear(ad_dim, hid))
            ad.load_state_dict(ck["adapter_state"])
            self.adapter = ad.to(self.model.device).eval()
            print(f"[duplex] audio adapter loaded "
                  f"({sum(p.numel() for p in self.adapter.parameters())} "
                  f"params, dim {ad_dim})", flush=True)

        # chat-envelope specials only — the snapshot is taken AFTER
        # registration, so filter OUT the duplex ids: they must stay
        # decodable
        duplex_ids = set(range(self.n_old, len(self.proc.tokenizer)))
        self.mask_ids = [i for i in self.proc.tokenizer.all_special_ids
                         if i not in duplex_ids]
        self.name_of = {i: self.proc.tokenizer.convert_ids_to_tokens(i)
                        for i in duplex_ids}
        self.sil_id = next(i for i in duplex_ids
                           if self.name_of[i] == "<sil>")
        self.pad_id = self.proc.tokenizer.convert_tokens_to_ids(
            "<|audio_pad|>")
        self.emb_layer = self.model.get_input_embeddings()
        self.cache = None
        self._pcm = np.zeros(0, dtype=np.float32)
        self._t_s = 0.0  # audio time of the next feature
        print(f"[duplex] {ckpt_path} loaded ({time.time() - t0:.0f}s), "
              f"{len(duplex_ids)} duplex tokens live "
              f"(training-format stream, {WINDOW_S:.0f}s windows, "
              f"sil_bias={self.sil_bias:g}, max_pieces={self.max_pieces})",
              flush=True)

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

    def _encode_window(self, pcm: np.ndarray):
        """Ear features for one window — layera/prepare.encode_audio's
        exact recipe (same text prompt, same tower call, same tail)."""
        import torch

        inputs = self.proc(text=["<|audio_pad|>"], audio=[pcm],
                           return_tensors="pt", padding=True
                           ).to(self.model.device)
        n_tok = int((inputs["input_ids"][0] == self.pad_id).sum())
        with torch.no_grad():
            tower = self.model.get_audio_features(
                input_features=inputs["input_features"],
                feature_attention_mask=inputs["feature_attention_mask"],
                return_dict=True).last_hidden_state
        tower = tower.reshape(-1, tower.shape[-1])
        tower = tower[-n_tok:]
        if self.adapter is not None:
            # anchor to the adapter's device, not model.device — under
            # device_map="auto" the tower output and model.device can
            # disagree, and F.linear can't cross devices
            ad_dev = next(self.adapter.parameters()).device
            tower = self.adapter(tower.to(ad_dev, torch.float32)
                                 ).to(tower.dtype)
        return tower

    def _clean_argmax(self, logits) -> int:
        """Argmax over the live emission vocabulary: chat-envelope
        specials masked; <sil> penalised by sil_bias (decode-time knob —
        the model's 92%-silence training prior makes it win ties it
        should lose; the sweep asks whether subtracting from the <sil>
        logit restores speech without retraining)."""
        import torch

        for tid in self.mask_ids:
            logits[tid] = float("-inf")
        if self.sil_bias:
            logits[self.sil_id] -= self.sil_bias
        return int(logits.argmax())

    def _beat(self, feat, t_s: float) -> tuple[int, float, float]:
        """One decision beat: audio feature in, argmax emission out,
        emission embedding fed back. Returns (token, audio_t_s, wall_s)."""
        import torch

        t0 = time.time()
        e_in = self.emb_layer(
            torch.tensor([[self.pad_id]], device=self.model.device)
        ).clone()
        e_in[0, 0] = feat.to(e_in.dtype)
        with torch.no_grad():
            out = self.model(inputs_embeds=e_in,
                             past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        tok = self._clean_argmax(out.logits[0, -1].clone())
        e_out = self.emb_layer(
            torch.tensor([[tok]], device=self.model.device))
        with torch.no_grad():
            out = self.model(inputs_embeds=e_out,
                             past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return tok, t_s, time.time() - t0

    def _continue_word(self, tok: int, t_s: float) -> list[int]:
        """Word continuation — the stream-contract fix (Codex F2).
        Training emits multi-piece words CONSECUTIVELY after one audio
        input ([pad w1 w2] then the next pad; 33.6% of training words
        have >1 piece). The old live loop fed one piece, then the next
        audio feature — a sequence shape training never shows. This
        feeds each emitted piece and predicts the next position with no
        new audio, matching training exactly; the first non-word breaker
        (usually <sil>) is DISCARDED, not fed — training has no token
        between a word's last piece and the next audio pad. Returns the
        extra pieces (first piece already placed by _beat)."""
        pieces = []
        while len(pieces) + 1 < self.max_pieces:
            import torch

            e = self.emb_layer(
                torch.tensor([[tok]], device=self.model.device))
            with torch.no_grad():
                out = self.model(inputs_embeds=e,
                                 past_key_values=self.cache, use_cache=True)
            self.cache = out.past_key_values
            nxt = self._clean_argmax(out.logits[0, -1].clone())
            if self.kind_of(nxt) != "word":
                break  # breaker: drop it, the next feature decides anew
            pieces.append(nxt)
            tok = nxt
        return pieces

    def reset(self) -> None:
        """Fresh session: clear KV memory, audio buffer, and clock."""
        self.cache = None
        self._pcm = np.zeros(0, dtype=np.float32)
        self._t_s = 0.0

    def _run_feature(self, feat, t_s: float) -> list[tuple[int, float]]:
        """One feature -> its beat plus any word-continuation pieces
        (all at the same audio timestamp, as in training)."""
        tok, _t, _w = self._beat(feat, t_s)
        beats = [(tok, t_s)]
        if self.kind_of(tok) == "word":
            for piece in self._continue_word(tok, t_s):
                beats.append((piece, t_s))
        return beats

    def feed_audio(self, pcm_16k: np.ndarray) -> list[tuple[int, float]]:
        """Absorb user audio; one decision per ear feature. Returns
        (token, audio_t_s) pairs — timestamps let the harness place
        output despite the windowed feature lag."""
        self._pcm = np.concatenate(
            [self._pcm, pcm_16k.astype(np.float32)])
        beats: list[tuple[int, float]] = []
        while len(self._pcm) >= WINDOW_SAMPLES:
            win, self._pcm = (self._pcm[:WINDOW_SAMPLES],
                              self._pcm[WINDOW_SAMPLES:])
            feats = self._encode_window(win)
            step = WINDOW_S / max(len(feats), 1)
            for i in range(len(feats)):
                beats.extend(self._run_feature(feats[i], self._t_s))
                self._t_s += step
        return beats

    def finish(self) -> list[tuple[int, float]]:
        """Flush the final partial window (encode_audio encodes tails
        down to 800 samples and skips shorter)."""
        beats: list[tuple[int, float]] = []
        if len(self._pcm) >= 800:
            feats = self._encode_window(self._pcm)
            step = (len(self._pcm) / SR_IN) / max(len(feats), 1)
            for i in range(len(feats)):
                beats.extend(self._run_feature(feats[i], self._t_s))
                self._t_s += step
        self._pcm = np.zeros(0, dtype=np.float32)
        return beats
