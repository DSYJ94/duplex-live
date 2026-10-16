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
Context Spanning assets 7454205 are complete: source commit and public model
revision are pinned, and checkpoint SHA matches the publisher's LFS digest.
Native probes 7454283/84 test two actual private worlds; 7454285 uses the accepted
synthetic reference voice. All three strictly load 16 native audio heads and generate the actual private
value without a forced response start, with p95 29.7–34.0ms per 80ms frame and
zero causal-playout underruns. First-frame stalls and incorrect follow-up physics
remain. The next probe connects native retrieval markers to actual arrived-audio
recognition and real backend execution; backend tool choice stays external.
No new training or complete-task qualification follows from these component runs.


## 2026-10-15 — Context voice accepted; native result use still fails

Plain words: the owner reports Context Spanning's synthetic-reference voice is
“as good or better” than the accepted smooth Lychee recording. This selects a
voice candidate only. Scheduled-dispatch probes 7454283–85 say their actual
private values. Native-RET, actual-ASR and actual-backend probes 7454442–44
(warmed) and 7454577–79 (cold) all recognize the project and obtain the right
actual tool result, but only 7454578 samples the correct value. Other replies
invent/omit it and ordinary physics answers remain unreliable. All speech words,
starts and acoustic tokens are sampled; specific tool choice remains external.

Full-stream independent audits 7454749–58 yield WER respectively 0, .0208, 0,
.0517, 0, .0182, .0667, .0635, 0 and 1.7568. The final synthetic-voice native
run repeats audible material absent from its text; it fails speech behavior.
Other intelligible recordings do not pass factual/delegation gates. Full LM+PCM
runs together at p95 28.5–35ms per 80ms frame. Cold released first-frame stalls
remain roughly 2.8–3s; cold synthetic maximum backlog is about .505s. Warming
full Engine.step removes most cold cost but does not consistently preserve
continuity or behavior: warmed native purple/synthetic captures contain 406/2052
causal-playout underrun samples. Cold native controls have zero underruns.
First cold/warm text divergence precedes actual result insertion; different
nodes/caches/numerical execution remain confounds. No proven RNG-only bug or
quality-preserving warmup fix is claimed.

The next implementation is a bounded eight-update TRAIN-only Context adapter
proof: native 16-head aligned loss, four final temporal layers plus text head,
rank eight with A nonzero/B zero, base frozen, sampling RNG preserved during
construction. Reuse approved clean native caches after every source approval,
fold, artifact, word-timing and receipt check. No arbitrary crop or held-out
failure becomes supervision. User loss targets the clean separate user stem,
retaining genuine overlap; prefix/span targets and invalid IDs are excluded
before CE. This implements an ownership-mask seam, not waveform AEC or measured
echo suppression. No larger run is authorized by component fit alone; require
finite gradients, unchanged frozen hashes, fresh reload and native free-running
retention checks.


## Execution update — 16 October

Plain words: Context Spanning is now an owner-accepted voice, while result use
and control remain unqualified. Native 16-head training has passed finite
gradient, frozen-weight, exact fresh-reload and teacher-forced input/temporal
checks. The earlier128 pilots fail actual generation and must not be scaled.
Native request labels are now separated from actual worker dispatch clocks,
with explicit TRAIN-only causal guards; the eight-update variant still fails
private result use. Actual native student histories are being collected as
unapproved conditioning, with all past labels excluded from future supervision.
See HANDOFF.md and EXPERIMENTS.md for job IDs, measured results and the current
Isambard access block. The loss ownership mask is implemented; waveform AEC
and microphone double-talk performance are still untested.


## 16 October, late — Isambard remains the priority

Plain words: the owner explicitly prioritises a good Isambard result and rejects
additional compute as an assumed source of gains. Cloud GPU validation is on
hold; the previously approved $25 cap remains for later validation only. No
cloud resources have been launched. Keep the export, but focus experiments on
observed failure mechanisms within the remaining allocation.

Plain words: both paired13-case continuations (7480819 teacher acoustics,
7480820 own acoustics) reach136 updates, preserve frozen weights and reload
prediction/optimizer/RNG state exactly. Their training losses fall from
0.981 to0.684 and1.260 to0.881 respectively, but neither qualifies on native
conversation controls. Correct text after a prior invented private answer does
not constitute success. Do not expand these recipes on training loss alone.

Plain words: actual-worker evidence identifies a separate interface failure.
In native7480841 the real receipt contains Alder North's `teal star`, but the
backend prose answers only the later physics question. The runtime previously
injected that prose alone, omitting the successful record value. Source3e42038
preserves verified, current, actually executed lookup facts alongside backend
prose and records the exact injected span. Cancelled/replaced/unrecorded
receipts are rejected; errors remain failure notices. All21 focused worker,
failure and payload tests pass. Native improvement is not established yet.
This implements the Context Spanning principle of supplying external information
explicitly; it is not additional model training or forced spoken output.

Plain words: the practical benefits from the links are the owner-accepted
Context Spanning voice candidate, native delay/span conventions, explicit
speaker/target ownership and timing/retention evaluation. BayLing's 10:5:10
layout and timing-preference optimisation have NOT been reproduced here.
The current adaptation supervises only future assistant text; user/audio loss
is zero. The earlier16-head trainer has a clean-user ownership mask. Neither
is evidence of waveform echo suppression or tested microphone double-talk.
Full video captions remained inaccessible; only metadata/description was
reviewed, so the video is not claimed as a fully reproduced recipe.


## 17 October — measured pending-state result and remaining control gap


Plain words: early17 October — targeted state/payload tests finished. Native
pending-status runs7485621/22 produce the correct orange square/teal star;
independent CPU speech audit7485697 confirms both complete values (word error
against sampled text0.0606/0.0270). No spoken text/audio/start is forced. Without
pending status7485619/20 still invent D138/six. Receipt-only changes do not
establish safe waiting. This is improved private-result behavior, not complete
conversation qualification: TRAIN000 still omits the intervening physics answer.

Plain words: broader pending-state controls remain failures.7485698 gives correct
pencil/graphite text;7485699 is incoherent about evaporation and7485700 ignores
cancellation. Actual cancel-scene backend requests see only audio BEFORE the
cancellation, so no cancellation tool is dispatched. Zero-adapter baselines
7485702/7485755 also fail the private cases; ASR/backend timing differs, so this
does not isolate the contribution of training.7485701 was rejected for missing
TRAIN-scene authorization flag and is not a model result;7485755 is its retry.

Plain words: do not expand the failed private-only recipe. Its13 new supervised
continuations are private facts; other behavior kinds are base-distribution
retention, not corrective supervision. Next training work must explicitly
approve ordinary/clarification/cancellation request/timing targets from TRAIN
sources and actual arrived user speech, with separate eligibility rules for
non-factual actions. The fact-only receipt gate must not be weakened to smuggle
in unsupported control labels. In parallel, measure request delay, pre-receipt
speech and post-receipt value use separately. No qualified native model exists.
