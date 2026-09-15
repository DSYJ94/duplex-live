# MiniCPM-o 4.5 — Layer-A code reading notes

2026-09-15 · Source: HF snapshot `openbmb/MiniCPM-o-4_5` @ 503e7542
(`modeling_minicpmo.py`, `utils.py`, `processing_minicpmo.py`) + the bundled
`stepaudio2`/`minicpmo` pip packages. Feeds the Layer-A spec (docs/09).
Bake-off results that motivated specific readings: `evals/baselines/minicpm-o/`.

## 1. TDM time-slice format

One **unit = 1000 ms** of wall time (`chunk_ms: 1000`, plus `first_chunk_ms:
1035`, `cnn_redundancy_ms: 20` — `_default_duplex_params`,
modeling_minicpmo.py:2446-2471). Every unit is bracketed by dedicated tokens:

```
<unit><image>[img_embed_64]</image>[audio_embed_50]<|listen|>or<|speak|>generated_content</unit>
```
(modeling_minicpmo.py:3415, `get_session_schema`)

- Audio is 50 encoder embeddings per second of input
  (`chunk_num_frame = int(chunk_length * 50)`, modeling_minicpmo.py:677).
- Per chunk the driver calls `streaming_prefill()` once: feeds `<unit>`, then
  vision embeds (optional), then audio embeds; final forward pass leaves
  `pending_logits` for the generate step (modeling_minicpmo.py:2777-2866).
- Special tokens (all new vocabulary, modeling_minicpmo.py:2557-2582):
  `<unit>`, `</unit>`, `<|listen|>`, `<|speak|>`, `<|tts_bos|>`, `<|tts_eos|>`,
  `<|chunk_eos|>`, `<|chunk_tts_eos|>`, `<|turn_eos|>`, `<|tts_pad|>`.

## 2. Speak/listen control (the 1 Hz decision)

`streaming_generate()` (modeling_minicpmo.py:3151-3405) runs once per unit,
after prefill:

- Samples up to `max_new_speak_tokens_per_chunk` (default 20) tokens. The
  **first sampled token is the decision**: `<|listen|>` → return silence for
  this unit; `<|speak|>` → continue generating content tokens.
- Budget exhaustion force-feeds `<|chunk_eos|>` (explicit `ls_mode`,
  line 3216-3220).
- **Mid-turn listen guard**: if `<|listen|>` is sampled while
  `current_turn_ended == False`, it is replaced with `<|tts_bos|>` — the model
  may not abandon a turn mid-sentence (line 3237-3239).
- Knobs: `force_listen_count` (force listen for the first N generate calls,
  line 3204), `listen_prob_scale` (bias the listen logit in
  `StreamDecoder.decode`), `listen_top_k`.
- `<|turn_eos|>` ends a turn (`turn_terminator_token_ids`);
  `chunk_terminator_token_ids` = listen / chunk_eos / chunk_tts_eos.

## 3. Text→speech handoff (hidden-state conditioned, no text detour)

When a unit contains spoken content (modeling_minicpmo.py:3317-3388):

1. The LLM's **hidden states** of the generated tokens (not detokenised text)
   become the TTS condition: `_convert_results_to_tts_input` (line 3339, def
   at 3508).
2. The CosyVoice2-class TTS LLM runs `generate_chunk()` with
   **≤ 26 new speech-codec tokens per unit** (~25 codec tokens ≈ 1 s audio),
   min 26 mid-turn but free at turn start/end so short utterances flush
   (line 3342-3350).
3. Token2wav (flow-matching + HiFT vocoder, streaming caches) emits 1 s of
   24 kHz audio per unit; `_generate_silence_waveform()` pads listen units
   (line 3617-3621).
4. Per-turn state: `tts_past_key_values` (TTS KV) and `tts_text_start_pos`
   carry across units, reset at `<|turn_eos|>`; token2wav stream cache resets
   per turn with a 3-token silence prefix `[4218]*3` (line 2653-2657).

This is the "interleaved text–speech" design: text semantics flow to the
vocoder through hidden states, unit-synchronous, so speech stays aligned with
new input arriving every second.

## 4. Barge-in / yield / retraction

Two levels — **policy** and **systems**:

- Policy: the per-unit `<|listen|>` decision *is* the yield mechanism. In the
  bake-off (T2) an interruption flipped the model to listen within 1 unit.
- Systems: `break_event`, a `threading.Event` (modeling_minicpmo.py:2527,
  2626-2640). External drivers (e.g. client VAD) call `set_break_event()`;
  both `streaming_prefill` (line 2838) and `streaming_generate` (line 3166)
  then short-circuit: generate returns `is_listen=True, end_of_turn=True` +
  silence immediately. `set_session_stop()` = break + stop.
- **No in-model retraction**: already-vocoded audio is not recalled; since
  audio is emitted in 1-s units, at most ~1 s of stale speech can be in flight.
  Retraction is a playback-buffer concern, not a model concern — worth copying.
- Backchannels are handled purely by the learned listen/speak policy; there is
  no hard-coded backchannel filter.

## 5. Session API (exact call sequence)

```python
model = AutoModel.from_pretrained(REPO, trust_remote_code=True, ...,
    init_vision=True, init_audio=True, init_tts=True)   # modeling_minicpmo.py
model.eval().cuda(); model.init_tts()                   # loads Token2wav (5 HF assets)
duplex = model.as_duplex()                              # -> MiniCPMODuplex wrapper (line 2429)
duplex.prepare(prefix_system_prompt=...,                # resets all session state,
               ref_audio=np_16k, prompt_wav_path=...)   # builds audio system prompt, (line 2692)
# per 1-s chunk:
duplex.streaming_prefill(audio_waveform=chunk_16000)     # sets pending_logits
res = duplex.streaming_generate(prompt_wav_path=...)    # dict: is_listen, text,
                                                        # audio_waveform (24k), end_of_turn,
                                                        # cost_llm/cost_tts/cost_token2wav
base = duplex.as_simplex(reset_session=True)            # back to half-duplex (line 3629)
```

`MiniCPMODuplex.from_existing_model` (line 2470-2624) wires the token IDs,
the `StreamDecoder` (persistent KV incremental decode, utils.py:1187, `feed`
at 2080, `decode` at 2106), and the sliding-window config.

## 6. Context management over long sessions

`StreamDecoder` + `DuplexWindowConfig` (defaults at modeling_minicpmo.py:2590-2607,
enforcement utils.py:1453 / 1917):

- `sliding_window_mode`: `"off"` (default), `"basic"` (drop oldest units when
  > 8000 tokens, down to 6000), `"context"` (also keeps a ≤500-token
  `"\n\nprevious: "` text digest, max 24 units of history).
- The digest uses **generated text only** (special tokens filtered,
  modeling_minicpmo.py:3280-3291).

## 7. Porting notes for Qwen3-Omni Layer-A

- The unit cadence (1 Hz) is much coarser than our 80 ms tick target — Layer-A
  should treat MiniCPM-o's unit as the *policy* tick and keep a finer audio
  clock beneath it.
- Copy: explicit listen/speak decision tokens; mid-turn listen guard;
  force-listen warmup; per-unit RTF telemetry (`cost_*` fields) — our harness
  reuses them directly.
- Copy: hidden-state-conditioned TTS (no text detour) — matches the Layer-A
  interleaved causal stream.
- Don't copy: systems-side barge-in via external event only. We want an
  in-model `<yield>` (docs/09) so policy can retract gracefully.
- Watch: bake-off T4 (3 s thinking pause → premature takeover) and T5
  (background speech → false engagement) failed on MiniCPM-o 4.5. Both are
  *policy* failures, exactly what our synthetic factory (Issue #3) must train
  against.
