# Context Spanning: bounded TRAIN-only adaptation seams

Plain words: fine-tune the released duplex checkpoint with a small, verified
adapter proof. Do not start foundation pretraining. The owner accepts its voice;
the remaining failure is reliable result grounding and ordinary factual
behavior. Keep the released model as the audio/behavior baseline. Six native
ASR development cases made the proper private calls but only one sampled its
returned value, according to the parent evaluation; these held-out failures
must not become training examples.

Source pin: `79df302fa2820cae10c61bb2b7f302c52edb6ed2`. This is a source review,
not a training run or GPU memory measurement. The existing probe has already
strictly loaded all 475 model keys with 16 audio heads; do not repeat a download
or mistake strict loading for adaptation quality.

## Native entry points and their limits

The [documented raw format](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/docs/TRAINING.md)
is one same-stem JSON/WAV pair, stereo **L=user, R=agent**:

```json
{
  "system_prompt": "You are a helpful assistant.",
  "voice": "/pinned/voices/main.pt",
  "turns": [
    {"speaker": "user", "words": [{"w": "Please", "t": 1.04}]},
    {"speaker": "agent", "words": [{"w": "The", "t": 6.08}],
     "ret": true, "reference": "The actual returned record value.",
     "body_word_index": 0}
  ]
}
```

This is schema illustration, not a valid approved training dialogue. Word
timestamps must come from the source measurements.

[`prepare(in_dir,out_dir,device='cuda')`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/training/prepare.py)
encodes both tracks with eight-codebook Mimi, writes an assistant text row,
and saves `agent`, `user`, `ch0`, JSON-string `spans`, `system_prompt`, `voice`
in NPZ. It truncates waveform length to full 80ms frames. Text subwords use
`spm.encode(' '+word)` from `floor(t*12.5)`; occupied/out-of-range slots are
silently skipped. A grounded turn overwrites the preceding frame with RET=4.
Its reference record contains `ret_frame`, `cap_frame`, `reference`, where
cap derives from the body word boundary minus two frames. There is no actual
worker receipt input. Therefore this convenience converter needs fail-closed
checks before use with our evidence requirements.

The native sampling function accepts one NPZ path, tokenizer and Python RNG.
It selects a simulated arrival between retrieval and grounded-body cap, builds
17 clean rows, inserts spans and prepends a masked persona/voice prefix. The
stock trainer randomly crops after assembly, potentially discarding earlier
context. It offers neither distributed/FSDP wiring nor activation checkpointing.
Use its layout as a reference; do not label sampled arrivals as actual worker
receipts. [Trainer source](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/training/finetune.py).

## Converter contract for the approved corpus

Use a repository converter around the native tensor/sequence functions rather
than exporting our files blindly through `main.py prepare`:

1. Call `evals.recovery_data.validate_conversion` for each approved source:
   base `recovery-data-7407554`, mixed `recovery-mixed-data-7410969`, paired
   `recovery-teacher-data-7434749`, controls `recovery-teacher-data-7435530`.
   Resolve actual paths from their receipts. Select only explicit TRAIN rows;
   deduplicate source identities and wave hashes, since mixed views can already
   include base examples. Preserve validation/test membership and hashes.
2. Record ingestion, approval, WAV, sidecar, timeline, native codec/tokenizer,
   released model/source and voice hashes. Record each source model and whether
   context is teacher or actual student history. Revalidate all of these before
   training, not only during one-time conversion.
3. Existing recovery WAVs are **assistant-left/user-right**, as implemented in
   `evals/recovery_teacher_convert.py` and `evals/recovery_encode.py`. Swap only
   if writing the native raw-WAV format; the canonical tensor order remains
   `[assistant text, assistant eight codes, user eight codes]`. Reuse a cache
   only after codec, clean-versus-delayed coordinate convention, transcript
   timing, provenance and code ranges match the loaded checkpoint.
4. Existing measured word alignments and marker frames are preferable to
   invented timings. Verify that every intended ordinary subword appears once,
   in order, without RET overwriting a real word. Retain the 200ms actual first
   subword timing guard for this Moshi-family layout. Do not shift overlapping
   words until they pass; reject or regenerate approved audio when alignment
   cannot represent it faithfully.
5. For actual receipts, derive the insertion boundary from the real arrived
   input timeline. Use the first eligible native frame at or after receipt,
   with original receipt time and grid error retained. Keep a map between real
   frame index and virtual index after earlier spans. Assert every private fact's
   first text token follows actual result visibility, not merely its audible
   waveform onset. A late reference must not be placed earlier just to fit the
   teacher's answer. Exclude or separately regenerate causally impossible pairs.
6. Use explicit direct-result spans for a first result-reading proof. Retrieval
   and cancellation targets still require trusted causal triggers; no result
   target after cancellation, and no invented reference on simple or clarify
   cases. Mark fixed/sampled-arrival augmentations separately from real receipts.
   Newly collected actual student histories must be TRAIN-only.
7. Keep complete persona, relevant user request, pending/returned event and
   response in bounded windows. If a long example cannot fit, choose an explicit
   causal segment and document its context boundary. Do not silently drop the
   request/receipt with a random window or insert a private value in the initial
   persona. Include ordinary, clarify, no-result, cancellation and paired-value
   controls so the adapter cannot learn to invent facts or always delegate.

## Span and loss alignment

[`assemble_training_sequence(codes,audio_mask,spans,spm)`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/model/sequence_convention.py)
accepts `[17,T]` clean codes and explicit `inject_frame`/`reference` records.
It inserts open=12, text and close=13 columns with silence/sine placeholders;
span text/audio loss is masked. Acoustic rows 2–8 and 10–16 from the preceding
frame move to the span's last column; original audio loss there is masked.
`persona_prefix` provides native voice/system tokens. Preserve this boundary
logic rather than concatenating ordinary text tokens into an audio cache.

The native
[`forward_train`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/models/lm.py)
applies codebook delays, prepends initial tokens, predicts temporal/depth
outputs, then reverses delays and returns aligned validity masks. Feed clean
coordinates once; do not pass an already delayed tensor or apply another
generic next-token shift to its returned logits. Verify span insertion against
an actual forced-step/native prefill reference at both boundaries.

For the loaded 16-head model, the upstream objective covers assistant and user
audio. Text padding weight is .5; assistant semantic head weight is 1 and all
15 remaining audio heads are .02. That is not two separately equal-weighted
speaker objectives. Prefix/span loss is excluded. Existing source `nan_to_num`
must not hide a nonfinite **valid** prediction in our proof. Any undefined
negative target or initialization token must be excluded before CE, since
multiplying a computed invalid CE by a zero mask cannot repair an out-of-range
class. Our loss wrapper should report channel ownership and valid label counts.

Do not call masking user-target positions AEC. A separate conditioning-only
user-loss ablation is reasonable, but changing the published loss requires an
explicit configuration and comparison. Preserve genuine overlapping speech and
assistant supervision. A result-reading text-only diagnostic cannot qualify
native speech, retrieval or interruption behavior.

## LoRA and eight-update feasibility

The native
[`lora.py`](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/model/lora.py)
reconstructs inference adapters; it is not a training installer. Both factors
start at zero and have `requires_grad=False`. Merely enabling gradients leaves
a zero-by-zero dead update. Freeze base parameters, initialize A nonzero/B zero
inside preserved RNG, enable only named adapter factors, and assert the exact
trainable set. Reload format uses tensor keys `.lora_A`/`.lora_B`, wrapped
`.base.weight` and top-level `lora={r,alpha}`. Our old adapters use different
keys. BF16 cold merging is not the same deployment computation as this wrapper.

The native attention input projection is a raw `in_proj_weight` Parameter,
not an `nn.Linear` child. Gated FFNs read `.weight` directly. A generic linear
replacement therefore needs a native-compatible weight property or explicit
forward handling; never report adapting raw QKV when only output projections
changed. Existing `evals/personaplex_adapters.py` documents these traps but its
serialization/bridge must not be imported wholesale. A bounded initial choice
is the text head and selected last temporal output/FFN projections, with exact
scope and rank recorded. This shares temporal state with audio; freezing codec
and depth parameters does not guarantee unchanged audio after semantic updates.
[Attention source](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/modules/transformer.py),
[gated FFN source](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/modules/gating.py).

A one-GH200 adapter proof is a plausible first experiment, **not a measured
memory guarantee**. Approximately 16.7GB of BF16 checkpoint tensors must fit;
frozen-base adapter training avoids full-model gradient and Adam state storage.
Full-model BF16 weights+gradients+two Adam tensors alone are approximately
67GB, before activations, temporary optimizer buffers and runtime allocations.
Native training has no automatic sharding; requesting multiple GPUs does not
divide this state. Use a short complete approved example first, record peak
allocated/reserved memory, then profile the longest planned context. If needed,
use explicit layer checkpointing after zero-update parity is established.
Neither FSDP nor full-foundation retraining is necessary for eight adapter
updates; they should not be introduced before a measured memory need.

Proof order: zero-adapter native logits and RNG parity; one finite backward
with intended nonzero B gradients; at most eight updates with later nonzero A
gradients; unchanged frozen-weight hash; exact fresh-process adapter reload;
native one-frame/teacher-forced alignment and span-boundary parity. Then run
fresh actual-ASR/tool development cases, paired values, ordinary answers and
interruptions with speech transcription/listening. Preserve implementation
proof, TRAIN fit and live-task qualification as separate verdicts. Stop scaling
if factual use improves only on teacher histories or damages accepted voice.

## Additional parity seams found before scaling

Plain words: batch-to-batch zero/reload equality does not yet establish that
the training graph sees the same history as native streaming inference. Source
inspection identifies three narrowly testable differences. No GPU job or
source change was made for this analysis.

**Depth head 15.** The pinned transformer still uses the ring-cache
`delta <= 0` position rule. With a 16-slot depth ring, the final query at
position 15 labels slot zero as future position 16 and excludes it. Batched
depth attention with `context=None` includes position zero. Consequently:

| Depth query | Native streaming visible positions | Unmodified batch positions |
| --- | --- | --- |
| 0–14 | 0 through query | 0 through query |
| 15 | 1–15 | 0–15 |

The existing `evals.personaplex_depth.native_depth_context` compatibility
context of 15 on the **unstreamed** path is a candidate match, not an inference
change. Verify per-head logits on identical hidden states and forced preceding
codes before adoption. Directly this concerns the last user acoustic head;
shared-gradient effects do not establish it as the cause of factual failures.
[Pinned attention source](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/modules/transformer.py).

**Startup prefix.** Let the forced native prefix columns be `C0...C(P-1)`.
At offset zero, `prepare_step_input` replaces the cache with initial tokens
and returns no forward. At offset one, delay-one initialization also replaces
the delayed acoustic values from C0. Thus C0 is omitted on all 17 rows.
The formal temporal inputs subsequently correspond to an explicit initial
column followed by delayed `C1...C(P-1)`. Native `forward_train` instead
prepends initial tokens to every supplied clean column. A candidate exact
training mapping is therefore `persona_prefix[:, 1:]` plus every real frame,
reducing prefix and real-to-virtual offsets by one. This drops only the native
startup's first synthetic prompt column, never real user audio. Verify actual
input IDs and temporal positions, not just decoded prompt text.
[Pinned LMGen](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/models/lm.py),
[prefix prefill](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/model/prefix_prefill.py).

**Span boundary.** The clean assembler moves the preceding real frame's
acoustic rows 2–8 and 10–16 into the last span column, so delay-one places them
in the first delayed real column after the span. The native prefill fills
inserted delayed columns with placeholders; its Engine readout compensation
restores one pending assistant semantic code, not those 14 acoustic rows.
A pure-Python integer simulation of the published write/delay rules, after
startup correction, finds a difference in precisely those rows when the first
post-span real column is later consumed. This is structural evidence, not an
actual loaded-model numerical result. Do not copy future assistant cache rows
as a fix: user acoustic rows can already be provided while the corresponding
assistant acoustic codes may not yet have been sampled.
[Span prefill](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/model/context_span_block.py),
[Engine](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/model/engine.py).

The smallest combined check is one actual accepted voice/persona prefix, two
forced real frames, one short injected reference and two further forced real
frames. Capture the actual temporal input columns and RoPE/KV positions at
each native `forward_codes` call. Compare against initial-plus-delayed clean
assembly, first without and then with the proposed one-column startup mapping.
At both span boundaries compare all 17 rows, state offsets and provided masks;
compare hidden/text logits after input equality and all 16 forced depth logits
separately. Use real TRAIN codes and a distinctive reference rather than all
silence, which can hide a moved/dropped acoustic row. The prefix and span
component comparisons should consume no random samples; if using the native
sampling loop, preserve/report RNG separately. Do not scale training while
these differences remain unmeasured or accidentally hidden by masked losses.
