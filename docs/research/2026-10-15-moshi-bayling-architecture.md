# Moshi, BayLing-Duplex and Context Spanning: implementation review

Plain words: retain the smooth Lychee baseline and finish its live result-use
checks. The strongest new alternative in these resources is the released Context
Spanning checkpoint: it was trained specifically to consume asynchronous text
results inside a PersonaPlex conversation. Audition it before reproducing that
training. BayLing offers useful supervision and timing ideas, but its token
layout is a different model interface, not a fusion layer we can insert into
Lychee. An echo loss mask cannot repair the current Lychee result-use failures,
because that trainer has no user-audio prediction loss.

Reviewed 15 October 2026. This note records source inspection, not a model run.
Only the Moshi/BayLing/Context Spanning subset of the supplied resources is
covered here. Runtime, audio quality and task claims require local reproduction.

## Verified identities

| Supplied resource | Actual identity | Availability checked |
| --- | --- | --- |
| [2410.00037](https://arxiv.org/abs/2410.00037) and [Kyutai repository](https://github.com/kyutai-labs/moshi) | Moshi: a speech-text foundation model for real-time dialogue | Paper and public source |
| [2606.14528v1](https://arxiv.org/html/2606.14528v1) | BayLing-Duplex: Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM | Paper, [inference source](https://github.com/BayLing-Models/BayLing-Duplex), [public four-shard model tree](https://huggingface.co/BayLing-Models/BayLing-Duplex/tree/main), approximately 19.1GB |
| [2609.33443v1](https://arxiv.org/html/2609.33443v1) | Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends | Paper, [source](https://github.com/mindlogic-ai/Context-Spanning), [16.7GB checkpoint and voice/assets tree](https://huggingface.co/mindlogicinc/context-spanning-7b/tree/main) |

The third resource is a separate delegation/retrieval method, not another
BayLing architecture paper. Public file listings were checked; checkpoint
downloads, hashes and Isambard authentication were not exercised in this review.

## Moshi: time and depth are separate axes

The [paper, sections 3.3–3.4 and 4.4](https://arxiv.org/html/2410.00037v2#S3.SS4)
uses 24kHz Mimi audio, eight 2048-entry codebooks per speaker at 12.5Hz.
One temporal step represents 80ms. Its large temporal Transformer summarizes
previous frames; a smaller depth Transformer factorizes the current frame.
Assistant text precedes assistant semantic and acoustic codes; user codes are
separate parallel channels. Acoustic codebooks lag their semantic codebook by
one frame in the final dialogue recipe. During inference, user codes are
observations rather than sampled assistant output. The published joint loss
balances text against weighted audio, weighting semantic codes 100 and acoustic
codes 1. Echo augmentation mixes delayed assistant waveform into the user
waveform: gain 0–0.2 and delay 100–500ms.

The current [Kyutai `lm.py`](https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py)
confirms the mechanics: `forward_text` sums each codebook embedding and the
text embedding before temporal attention. `forward_depformer_training` adds
projected temporal state to within-frame prefix embeddings. `LMGen._step`
places externally supplied user codes into delayed cache slots. `dep_q`
determines which audio channels have prediction heads; channel count alone
does not establish the supervised outputs. Training returns already aligned
logits and validity masks after reversing the delays, so applying an extra
generic one-token shift to those returned logits would be wrong.

Implementation implication: Moshi's depth axis is neither a flattened list of
all stream tokens nor an attention mask separating two independent language
models. Removing assistant embeddings from temporal context would change a
pretrained model's interface and discard its own conversational history.

## BayLing: block interleaving, not summed embedding fusion

The [paper, section 2](https://arxiv.org/html/2606.14528v1#S2) starts from
GLM-4-Voice's 9B LLM, speech tokenizer and decoder. Its flattened blocks contain
10 user-audio tokens, five assistant text/state tokens and 10 assistant-audio
tokens, at 12.5 audio tokens/second. Each block spans 0.8s; assistant outputs
are shifted ahead one block to avoid observing future user speech. Four state
tokens encode silence, reply onset, text completion and speech completion.
User tokens are conditioning-only: supervised loss covers assistant text and
audio, with weights balancing frequent silence against rare boundaries.
Timing preference optimization adds an SFT retention term. Inference restricts
the vocabulary by channel, and a mid-block interruption is handled after the
current block completes.

The [released README](https://github.com/BayLing-Models/BayLing-Duplex/blob/main/README.md)
provides `interleave_ratio="10:5:10"` and model/tokenizer/decoder loading paths.
It requires the BayLing weights plus `zai-org/glm-4-voice-tokenizer` and
`zai-org/glm-4-voice-decoder`. A public model tree is stronger availability
evidence than the README's outdated conditional private-repository wording.

Our inference: useful transfers are explicit conditioning-versus-target labels,
boundary weighting, causal timing tests and retention examples. Replacing
Lychee's native layout with these blocks would require new alignment, model
training and serving validation. A 0.8s block is also a different latency
envelope from our existing 400ms arrival chunks; reported success percentages
do not establish performance under our interruption and result-arrival tests.

## Context Spanning: directly relevant, but different from our prefix refresh

The [paper, section 3.2](https://arxiv.org/html/2609.33443v1#S3.SS2)
inserts returned text between dedicated span delimiters into an ongoing Moshi
timeline. The model predicts a retrieval marker; the backend runs
asynchronously. Injected text is prefilled rather than sampled, with assistant
silence/user 440Hz placeholder audio. Injected spans have no training loss.
Spans are inserted after the acoustic-delay transformation. Training starts
from PersonaPlex and includes result-arrival delays. The proposed benefit is
direct access to text rather than compressing it into user-audio embeddings.

The [released sequence convention](https://github.com/mindlogic-ai/Context-Spanning/blob/main/contextspan/model/sequence_convention.py)
specifies 17 rows: text, eight agent codebooks and eight user codebooks.
Retrieval/open/close IDs are 4/12/13. When inserting a span, the preceding
frame's delayed acoustic codes move to its final column; their original audio
loss is masked. This handles the real-audio boundary rather than simply
padding over incoming speech. Prefix and span masks are separate from causal
attention. The [training documentation](https://github.com/mindlogic-ai/Context-Spanning/blob/main/docs/TRAINING.md)
requires stereo dialogues and word times; the reference arrival is constrained
to precede the grounded body of an utterance. Its published sampler is
lead-in-dependent, so do not silently substitute a uniform delay recipe.

Current repository [`refresh_session`](../../evals/lychee_context.py) instead
replaces Lychee's system prefix, closes the old request, restores the original
body/audio/cache mappings/clock/processors and audits the next real native
request. Sampling RNG is checked unchanged. Our native semantic inputs mask all
past loss and supervise only future result words/EOT after the actual receipt.
These preserve context and causality, but **are not Context Spanning**: no
retrieval/open/close span tokens, virtual frame insertion, trained span
distribution or incremental span-only KV update has been adopted.

Our inference: an append-only, trained result event may reduce full-history
rebuild cost and provide a clearer learned boundary. It does not itself solve
the current 15-second wait for an eligible result utterance. Measure ready,
injection, first sampled fact token, first audible fact and interruption
separately. Compare actual-worker behavior, not just a fixed prompt containing
the answer from the start.

## Reproducible Context Spanning audition

Public Git HEAD resolved with `git ls-remote` during this review:
`79df302fa2820cae10c61bb2b7f302c52edb6ed2`. Pin and hash this source plus all
downloaded artifacts before running. These are proposed steps, not work already
executed.

The [weight loader](https://github.com/mindlogic-ai/Context-Spanning/blob/main/contextspan/model/weights.py)
first loads `nvidia/personaplex-7b-v1/model.safetensors`,
`tokenizer-e351c8d8-checkpoint125.safetensors` and
`tokenizer_spm_32k_3.model`; then it loads
`mindlogicinc/context-spanning-7b/context_spanning_7b.pt` with strict
`load_state_dict`. LoRA metadata is supported. `CS_BASE_DIR` and
`CS_WEIGHTS_DIR` select local artifact directories. Use `CS_VOICES_DIR` for
local `voices/main.pt` or a selected released voice; voice lookup deliberately
does not follow `CS_WEIGHTS_DIR`. The existing PersonaPlex assets match these
filenames; reusing them still requires hash verification.

The [engine](https://github.com/mindlogic-ai/Context-Spanning/blob/main/contextspan/model/engine.py)
exposes `Engine(checkpoint, device, temp, temp_text)`,
`set_persona(prompt, voice_codes)`, `step(user_pcm)` and
`inject_context_span(reference)`. At 24kHz/12.5Hz, `step` accepts 1920 float32
samples and returns `agent_pcm`, `text_token`, `is_ret`. Injection reports
consumed virtual frames and `last_prefill_ms`. It warms both step and span
paths. `clone_voice` creates eight-codebook prompts from PCM; the current
accepted synthetic reference is a possible controlled comparison after native
voice audition. Use this project's vendored `contextspan.moshi`, not our
installed `moshi` package.

Proposed bounded order:

1. Create an isolated source checkout and dependency overlay. Reuse the working
   container Torch; do not run an unconstrained install over accepted overlays.
   [`pyproject.toml`](https://github.com/mindlogic-ai/Context-Spanning/blob/main/pyproject.toml)
   requests Torch>=2.2, standard audio packages, `sphn`, and `mcp>=1.9,<2`.
   Python/package/native ARM compatibility remains untested. Its core engine
   does not require our custom Lychee vLLM wheel.
2. Check strict checkpoint loading, actual `dep_q`, all key shapes, Mimi state,
   finite output, native per-frame timing and span timing on GH200. The vendored
   [base loader](https://github.com/mindlogic-ai/Context-Spanning/blob/main/contextspan/moshi/models/loaders.py)
   uses `strict=False` plus compatibility copying before the final checkpoint
   load; record and reject unexpected fallback/missing base weights rather than
   treating absence of an exception as complete loading proof.
3. Run a controlled direct-injection component test: the same user audio and
   actual receipt schedule, with two fresh private values. Sample all spoken
   text/audio. Record these as externally scheduled injections, not learned
   retrieval. Include no-result and irrelevant-result controls.
4. Run native `<ret>` routing separately with real ASR/backend; classify trigger,
   tool, argument, result grounding and latency failures independently. Avoid
   conflating a scripted event with learned delegation.
5. Reproduce concurrent PCM and interruption tests, then independent per-utterance
   ASR and listening. Retain Lychee unless the alternative improves the actual
   task while retaining accepted voice quality.

The supplied [CLI](https://github.com/mindlogic-ai/Context-Spanning/blob/main/main.py)
supports `main.py infer --input-wav ... --output-wav ... --output-text ...
--checkpoint ... --voice ... --seed ...`. It automatically creates/prewarms
backend and ASR clients and checks tool discovery. Therefore use the core
`Engine` for the first deterministic component test; configuring only the
checkpoint is insufficient to run the complete CLI end to end. Default input
lead/tail silence is four/eight seconds and must be disclosed in latency tests.

## AEC: implement the right separation

Three separate interventions need distinct names and evidence:

| Intervention | Purpose | Position here |
| --- | --- | --- |
| Waveform acoustic echo cancellation | Remove playback leakage from microphone input using the actual render reference | Useful for real speaker/microphone trials; not yet evidence for fixing synthetic clean-channel failures |
| Loss ownership mask | Exclude supplied observations/injected context and corrupt target regions from prediction loss | Audit and enforce against channel provenance; preserve valid assistant text/audio supervision |
| Attention/context removal | Hide assistant history from later predictions | Do not adopt as a generic AEC fix; changes native factorization and removes useful self-history |

Our own training code establishes the immediate scope:

- [`lychee_training_inputs.py`](../../evals/lychee_training_inputs.py), lines
  137–154, initializes every label to -100 and enables only future semantic
  words/EOT. [`lychee_semantic_train.py`](../../evals/lychee_semantic_train.py),
  lines 122–126, computes CE only at those positions. User audio is continuous
  conditioning, not a predicted target. An additional user-stream loss mask is
  currently a no-op.
- Historical [`personaplex_train.py`](../../evals/personaplex_train.py), lines
  181–201, supervises assistant text plus the first eight audio heads. Do not
  infer a user loss merely because its input tensor contains 17 channels.
- The released Context Spanning
  [`finetune.loss_fn`](https://github.com/mindlogic-ai/Context-Spanning/blob/main/contextspan/training/finetune.py)
  selects audio targets using the actual output-head count and masks prefix/span
  columns; the vendored PersonaPlex loader infers 16 predicted audio heads. This
  differs from our eight-head historical recipe. Before further training,
  inspect the loaded checkpoint and count per-channel weighted labels explicitly.

Our proposed extension, if training includes a user prediction objective: retain
a clean user target alongside the possibly echo-contaminated microphone input.
Represent echo-only, genuine-user, double-talk, uncertain and padding regions
with trusted data provenance. Mask echo-only/invalid **user targets**; keep
genuine user supervision during overlap and keep assistant supervision. Never
mask every frame in which the assistant speaks: that would remove exactly the
barge-in examples needed for duplex behavior. Codec token identity is not a
speaker label, and nonlinear codec mixtures cannot be separated by subtracting
token IDs. For a conditioning-only user stream, there is no such target loss;
use waveform AEC and echo augmentation plus false-trigger/barge-in tests instead.

Required checks before calling any AEC implementation successful: assistant-only
playback must not trigger a user interruption or delegation; real user overlap
must still interrupt; real repetition of the assistant's words must remain
audible; delayed echo, gain changes and partial cancellation must be covered.
Record the actual render reference, microphone/AEC signals and timeline. A loss
mask unit check establishes ownership only; it does not establish acoustic echo
cancellation or full-duplex task success.
