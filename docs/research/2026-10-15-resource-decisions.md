# Decisions from the owner's duplex resources — 15 October 2026

Plain words: the immediate new candidate is Context Spanning, because it releases
a PersonaPlex-derived checkpoint specifically trained to receive asynchronous
backend text. Keep the smooth Lychee recording as the audio reference while
we audition that candidate. BayLing's supervision and timing ideas are useful,
but its flattened block layout is a different interface, not a small fusion
layer to add to Lychee. No new model is qualified by this review.

## Coverage of all ten links

| Owner resource | Finding and decision |
| --- | --- |
| [PersonaPlex paper](https://arxiv.org/html/2602.06053v1) | Hybrid voice/role conditioning builds on Moshi. Already auditioned here; keep the accepted synthetic voice as a comparator. Its paper does not supply the missing asynchronous delegation behavior. |
| [Codacus video](https://www.youtube.com/watch?v=shJLycXjhVY) | Title and first-party description verified; full captions were unavailable. Its described micro-turn cascade/top-layer adaptation is a separate comparator, not proof that our native acoustic-token model should use that recipe. |
| [DuplexCascade paper](https://arxiv.org/abs/2603.09180) | Borrow distinct interruption, backchannel and waiting targets. A 0.6s text micro-turn cascade differs from a native duplex speech model. |
| [DuplexCascade code](https://github.com/sbintuitions/DuplexCascade) | Actual inference/weights exist, with gated EU-excluded model metadata. Preserve access conditions; no access workaround is proposed. |
| [Kyutai Moshi code](https://github.com/kyutai-labs/moshi) | Preserve native temporal/depth factorization, separate stream identities, native delays and real observed user audio. |
| [FLM-Audio](https://arxiv.org/abs/2509.02521) | Released native model; useful contiguous-text, ASR/TTS curriculum and leakage-augmentation comparison. A separate runtime/codec audit is needed before any fine-tuning. |
| [BayLing-Duplex](https://arxiv.org/html/2606.14528v1) | Flattened user:text:assistant blocks of 10:5:10, one-block causal offset, assistant-only loss and timing-preference training. Do not transplant the 0.8s block layout into Lychee unchanged. |
| [Context Spanning](https://arxiv.org/html/2609.33443v1) | Prioritize a released-checkpoint audition with actual late receipts, then native retrieval choice. Its trained virtual text spans differ from our system-prefix refresh. |
| [Moshi paper](https://arxiv.org/abs/2410.00037) | Summed stream embeddings across time plus a smaller within-frame depth model. It also describes delayed assistant-waveform echo augmentation. |
| [Duplex survey](https://arxiv.org/html/2606.19453v1) | Extend acceptance beyond concurrency: genuine interruption, supportive backchannel, unfinished thought, background speech, worker return and sustained overlap. |

Detailed primary-source and code evidence lives in the
[Moshi/BayLing/Context Spanning review](2026-10-15-moshi-bayling-architecture.md),
[alternative-model review](2026-10-15-duplex-alternatives.md) and
[AEC/loss review](2026-10-15-aec-loss-review.md). Those notes distinguish paper
claims, released artifacts and experiments actually reproduced here.

## Moshi and BayLing mechanics

Plain words: Moshi packs simultaneous channels into one temporal step; BayLing
serializes a small block of each channel into one ordinary token sequence.

| Mechanism | Moshi | BayLing-Duplex | Current native Lychee experiment |
| --- | --- | --- | --- |
| Stream combination | Separate embedding tables summed before temporal attention; depth model predicts within-frame tokens | User audio, text/state, assistant audio appended as ordered blocks | Continuous user-audio encoder plus native text/speech/control channels and branched decoder |
| Audio clock | 80ms Mimi frames | 80ms tokens grouped into 0.8s blocks | 40ms S3/native slots, arrived input supplied every 400ms |
| Causality | Native channel delays and earlier frames | Shift assistant output by one block; use the same offset in playback | Preserve actual arrived audio/cache and receipt times; no future input |
| User target loss | Depends on architecture/head configuration; joint paper objective differs from our historical assistant-only trainers | None: user tokens condition predictions | None in the semantic pilot: only future assistant text/EOT is supervised |
| Immediate transfer | Delay/stream ownership checks; speaker leakage augmentation | Explicit target ownership; boundary weighting; timing preferences with retention | Enforce assistant-only future targets and retain frozen speech/control paths |

These are distinct pretrained interfaces. A full conversion requires matching
training data, codec, inference grammar and latency validation. The allocation
supports adaptation and comparison; recreating foundation-scale speech training
is not the useful next experiment.

## AEC implementation decision

Plain words: prevent mislabeled speaker targets, but do not hide the model's own
history or discard the user's speech while the assistant is talking.

The proposed user-stream loss mask would currently be a no-op: Lychee has no user
prediction loss. Implemented `evals/lychee_loss_scope.py` now verifies every
unmasked label against independently sourced assistant text/EOT, rejects history
or targets before actual result arrival, and reports zero user/audio targets.
The trainer runs this check before optimization. Inputs, attention and existing
valid labels are unchanged; no echo-removal improvement is claimed.

For a later user-prediction objective, source attribution must come from clean
speaker stems and encoder support intervals, not repeated codec token IDs.
Echo-corrupted microphone input should be paired with clean user targets where
available, preserving genuine double-talk and clean silence. Supplied context,
voice prompts and invalid targets remain excluded independently. If only mixed,
ambiguous targets exist, reject or explicitly exclude them; do not label the
assistant's echo as user speech.

Actual acoustic echo cancellation belongs at the capture/render boundary, with
played PCM as its far-end reference. Generated but canceled/unplayed tokens are
not a loudspeaker signal. The current concurrent WAV is a causal playout
simulation, not that device boundary. A real microphone/speaker trial and
near-end/far-end/double-talk tests are still required.

## Execution order and current evidence

1. Preserve Lychee baseline 7452017, which the owner reports sounds smooth.
   Disabled-adapter 7454004 matches all three original token streams exactly.
2. Continue the bounded Lychee result experiment: four native development runs
   7454007/09/10/11 produce correct actual record sentences. Initial access denial
   and 14.8–16.4s result-reply activation delay remain failures.
3. Complete concurrent actual-worker speech audits. New baseline 7454087 has no
   gaps; worker captures 7454091/93 have 4564/3380 gap samples across earlier
   replies. Their result utterances have zero measured gaps. Keep the capture
   continuity failures visible while independently transcribing complete speech;
   a correct value does not waive the gaps or the earlier misleading reply.
4. Pin/download Context Spanning source and weights, then require strict native
   loading, finite speech, timed span ingestion and a cold controlled actual-
   receipt comparison. No synthetic scheduled injection is a native retrieval
   pass. Preserve the accepted synthetic voice in a controlled follow-up.
5. Only after these comparisons, allocate larger adaptation with ordinary-answer,
   pending/no-result, correction, backchannel, genuine interruption and echo
   cases. Keep sealed test material excluded; current repeated validation scene
   is development evidence, not a fresh held-out qualification.

The target guard is committed. CPU audit 7454206 verifies all 15 examples and
170 assistant-only targets without modifying inputs or labels. Speech audits
7454207/08/09 confirm both complete baseline utterances at WER 0; the yellow
result has WER 0.1111 (fails 0.08), purple WER 0. Earlier playback gaps remain.
Context Spanning assets are downloading in 7454205; the native probe is prepared,
with no inference result claimed yet.
