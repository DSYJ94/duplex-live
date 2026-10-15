# Duplex recovery — active from 13 October 2026


## Current execution — 15 October, native Lychee and actual audio

Plain words: no complete model qualifies. Continue the owner's authorised work
until actual input is needed; the allocation expires in about six days. The
accepted development voices are PersonaPlex synthetic marin and Lychee. Use
[docs/LYCHEE_NEXT_STAGE.md](docs/LYCHEE_NEXT_STAGE.md) for the next actions.

The released Qwen-based Lychee model runs through its native custom-vLLM backend
with all 982 checkpoint tensors accounted for. Warm-input baseline 7451709 keeps
maximum input backlog to 18.5ms; every text, speech and control token equals the
original 7441742 baseline. Concurrent two-GPU capture 7452017 completes 70s of
paced input in 70.540s with p95 model compute 299ms/400ms, maximum backlog 72.7ms,
1258 codec units accounted for and zero within-utterance gaps under fixed 240ms
per-utterance causal playout. It actually generates LM tokens and PCM together;
the final WAV simulates playout. Browser/audio-device/microphone use is untested.
The baseline still makes factual errors and is not a qualified voice assistant.

Static fact controls 7450548/50 say both private values. Actual late-return
7451814/15 and idle/quiet delivery 7451816/17 omit both values despite verified
fresh prefixes at the actual engine request boundary. Quiet delivery adds about
20s and is rejected. More prompt/timing-only variants are not the next step.

Native capture 7452224 and HF comparison 7452225 compare all three raw heads
at three fresh prefills using identical embeddings and correct next-text merge
conditioning. All nine raw top-one predictions match; RMS logit differences are
0.0379–0.1219, KL 0.000153–0.00863. This is evidence of similar graph behavior,
not numerical identity, actual masked control agreement or incremental-KV proof.

Native S3 conversion 7452500 yields 125 segments from 48 approved TRAIN scenes;
audit 7452502 accepts 106 and excludes 19. Actual student histories 7452624–31
support seven causal native semantic examples (7452831), with all past loss
masked and private text targets after actual results. The eighth lacks exact
native gold transcript agreement and is excluded.

First semantic proofs 7452899/7452921 reject NaN gradients before updates on the
default attention path. Matched no-cuDNN 7452937 and math 7452938 pass eight
updates: 2,883,584 trainable parameters in main layers 24–27, unchanged frozen
hashes, zero same-input speech/control drift and exact fresh-adapter reload.
Checkpoint 7452937 SHA 5c238a774042eee2596350b53fd04afd83ea689b8ed470475001e7636adf7965.
Paired TRAIN histories 7452946–58 and causal assembly 7453119 complete
15 usable examples. Both 64/128-step pilots 7453182/83 preserve frozen branches
but their cold actual-worker replays fail: they invent room temperatures, and
one value echo describes lightning instead of the record. Ordinary answers also
drift. Do not scale this recipe. Fixed-state TRAIN continuations 7453748–50
now show exact native input IDs and embeddings (maximum error 0). Both adapters
return all three supplied private values correctly; base returns one of three.
These teacher-started TRAIN component checks do not qualify live behavior.

Next diagnosis keeps the base weights during ordinary conversation, activating
the checked adapter only for one application-started result reply after a real
receipt, native listening state and two arrived quiet chunks. Every switch
aborts the old KV request, preserves history/audio/clock/RNG and rebuilds it.
Disabled-adapter 7454004 now reproduces every baseline token exactly.
Actual-worker 7454007/09/10/11 produce the correct room-marker sentence in all
four counterfactual replays, but initial access denial and 14.8–16.4s activation
delay remain. Next: concurrent two-GPU audio plus independent speech audit. This is a disclosed application policy, not
learned turn choice or delegation. Keep held-out data excluded from training.

The owner listened to concurrent capture 7452017 and said "Sounds smooth" on
15 October. Retain it as the development audio baseline; factual accuracy,
worker behavior and microphone acceptance remain separate unresolved gates.

New owner resources are reviewed in
[the decision note](research/2026-10-15-resource-decisions.md).
CPU loss-scope audit 7454206 passes all 15 examples; there is no user-audio loss
in the current pilot. Concurrent worker speech audits confirm the actual values,
but yellow WER 0.1111 and earlier playback gaps still fail the gates. Context
Spanning assets 7454205 and a bounded native span audition are next; this is a
new trained backend-text mechanism, not our older system-prefix refresh.

Code moves through Git only. New jobs use clean immutable worktrees and exact
commits. Fetch full logs before local filtering. Preserve unrelated pyc, roadmap
and cluster files. Earlier notes below record historical stages and may be stale.

## Historical continuation notes

**Plain words — 15 October, active continuation:** custom PersonaPlex and
Lychee voices remain accepted for development quality only; no complete model
passes. Generic-role self-history policy **7436033** reloads exactly and its
native replays **7436134/37/39/40** demonstrate actual lookup, simple no-call,
clarification-without-private-read, and real cancellation receipts. Spoken facts,
waiting, and continuation still fail. SHA
`11ce28b70c936be367e0cde24b85a2093e34412cb5e03feb3ae0f68fc1a85c8a`.
The matching records-role head **7437285** also reloads exactly but does not
qualify behavior. Direct worker-text forcing, untrained native text memory, and
quiet handoff **7437911/12/13/14** do not fix unsupported completion/facts. All
remain explicitly experimental; backend context is a causal measured-word oracle,
not live ASR.

Current native experiment **7439916** warm-starts 7436033, freezes language and
action weights, and trains causal client-state embeddings plus the reference
bridge on the 48 approved TRAIN scenes. Prefix state is fixed zero; a selected
action enters only the next frame, and results enter only at actual receipt time.
Eight updates have run with finite nonzero state gradients; preservation/reload
and actual free-running behavior are required. Historical teacher cancel label
020 is now derived correctly from its bounded two-call chronology (v2); approved
source data is immutable and older checkpoints retain their original lineage.

Lychee official custom-vLLM build **7437283** compiles/links all 116 CUDA targets
with the recorded isolated CUDA13 operation-name migration, then fails on a
missing generated Python package directory. **7439917** resumes those verified
compiled sources with the directory correction. Native kernel numerical tests,
complete checkpoint loading, paced tokens, and actual streamed PCM are still
required; compilation does not establish a viable real-time runtime. No owner
input is needed now; Clifton renewal was verified and existing access continues.

**Plain words — 15 October, behavior failure and next controlled tests:** native
128-update **7435379** saves/reloads with text/audio/action errors 0.0 and frozen
base hashes unchanged; SHA `44cb8b5b64838afd370f93fa2d2e9fc73bace43e5c971027941bf039ac9a547f`.
Its five free-running development checks **7435517/18/19/20/22 all fail** in
model text/action evidence: missing reads, invented/incomplete facts, silence,
and no successful cancellation. Audio audit is still required; no success claim.
Eight-update native replay 7435380 also fails with no dispatch. Native action
choices are real; workers still use causal measured transcript context, not ASR.
Do not scale the language-adapter recipe. Next tests compare zero-adapter speech,
a real scheduled-worker diagnostic (explicitly not native routing), and a model
with language weights frozen, training only reference integration/action choice.
The frozen-language run must prove reference-free text/audio unchanged exactly.
Revised control batch 7435049 (resumed by 7435472) completes seven raw scenes;
audit 7435484 accepts only two. Conversion 7435530 and native encoding 7435538
verify both, at most 100ms word error. They may join the separately hashed 13
paired teachers and original 33 TRAIN scenes (48 total); held-out folds unchanged.
Backend component 7435106 passes 8/8 actual API/tool cases, not speech routing.

**Plain words — 15 October, native joint training cleared:** proof **7434702**
passes with text/audio errors 0.0000763 / 0.000124, exact delayed inputs/targets,
zero adapter effect and finite nonzero gradients in all 25 intended output
adapters. The batched-only depth mask matches the pinned streaming cache's
16-slot boundary; the source and streaming model stay unchanged. Prior failures
7434416/7434638 remain recorded. This proves implementation, not task quality.
Paired-private collection 7434240 completes 32 raw recordings. Audit 7434696
rejects 19 and accepts 13; conversion **7434749** verifies all 13 under unchanged
speech and 200ms timing bars. Encoding/native grid checks remain required.
Silent teacher responses now become explicit failed records, allowing the rest
of the batch to be checked; the old aborted audit 7434249 is preserved.
A bounded eight-update native PersonaPlex pilot is prepared: final-eight-layer
LoRA, a zero-initialized reference bridge and separate idle/delegate/cancel head;
Mimi, depth weights and raw QKV remain frozen. It must save and reload exactly
before any generated-speech conclusions. Extra paired data can enter only after
its native code checks pass. No model qualifies; no owner input currently needed.

**Plain words — 15 October 11:43 Berlin, access restored:** renewed Clifton
certificate is verified by actual SSH and valid until 23:42 Berlin. Paired-private
teacher job **7434240** is running (32 raw TRAIN-only sessions, four worlds per
question); independent strict audit **7434249** is queued after success. Neither
raw generation nor audit completion alone approves targets for training.

Native history proof **7429698 passes exactly** on another TRAIN scene: every
prepared native input agrees, including the 162-frame accepted voice/system
prefix (shape 1 x 17 x 402). Template SHA
`8b34f2acd3b43da0d9c3eb6f5c84cab0506aeec1134d6603c978db5467a04cd0`.
Control-teacher audit **7429624 rejects all seven** scenes. Clarification failures
include missing questions/premature or wrong record reads; cancellations lack real
cancel receipts and disclose the supposedly cancelled result. Keep all seven out.
Inspect complete event logs before changing the client/worker contract.

**Plain words — 15 October, current engineering stage:** accepted custom
PersonaPlex voice is the leading integration candidate; Lychee remains a viable
voice alternative. Neither is product qualified. Paced PersonaPlex 7415855 has
32.46 ms p95 compute per 80 ms frame, with one 50.71 ms gap after buffering.
Reference proof 7427733 passes with the exact accepted voice/system prefix.
Bridge pilot **7428894** trains eight updates on one approved TRAIN phrase only:
frozen speech hashes unchanged; streaming/full-forward error 0.0001847;
zero bridge and reference-free prefix errors 0.0; fresh-process reload error 0.0.
Bridge SHA `76db170d7f15085ba1241d127f67e48a5b92f4b973f183dc3ab1901b7077f919`.
The phrase loss changes from 9.2622 at step 1 to 5.2089 at step 8; this is an
engineering result, not generalization or spoken task success. Native token 4
is a byte token; separate routing must be trained rather than assuming RAG support.

**Completed/next checks:** native HF incremental Lychee **7428971** measures token
processing with only already-arrived 400 ms audio chunks; vocoding is afterward,
so it cannot establish live audio latency. PersonaPlex actual-worker replay
**7429089** fails a manifest assumption (ARC has no standalone config); corrected
**7429163** uses real API/record-tool receipts and frozen ARC rows through the
trained bridge, with a scheduled diagnostic dispatch and causal measured words.
It explicitly does **not** test native delegation choice or live ASR. Interactive
submissions while another reservation job runs can hit QOSMaxSubmitJobPerUserLimit;
normal workq accepted 7429089 without changing QoS or cancelling running work.

### Recording fault found; perceptual cause still under investigation

The availability-aligned recorder starts each packet as soon as possible without
a startup buffer. The old Moshi recording can be reproduced sample-for-sample
from its packet ledger: it inserts two underruns (silence when the next packet
is late), totaling 122.54 ms, largest 82.96 ms. A causal 160 ms startup buffer
preserves every generated sample and removes both underruns. Originals and
original latency evidence stay intact. This does not prove every click is fixed.
PersonaPlex's official offline exporter already concatenates frames continuously,
so this fault cannot explain its artifacts. The gallery serves byte-identical
WAVs; PersonaPlex and GPT-Live have no clipped samples or sample-rate mismatch.

Fresh GPT-Live replay now writes original, buffered and raw-packet diagnostic WAVs
plus exact packet availability/underrun receipts. Raw packet concatenation is a
voice diagnostic, never a wall-clock timing claim. Future native recovery replays
also preserve both views. New buffering has three regression tests, including
the requirement to retain real stalls rather than hiding them.

Eight-codebook Mimi reconstruction 7414595 preserves the 10-second reference's
240,000 samples with zero padding. CPU decoder control 7415051 decodes the exact
same tokens without compilation: RMS difference from compiled GPU 0.0000311,
maximum 0.002305. This is numerical evidence, not a perceptual verdict. GPU
uncompiled control and custom PersonaPlex prompt complete in 7415052. GPU
uncompiled RMS difference is 0.00001817, maximum 0.0003555. These differences do
not establish perceptual transparency, but do not suggest a large decoder
compilation divergence. The codec
control uses the original arrival-aligned GPT-Live recording, which may itself
contain recording gaps; do not treat it as a pristine voice-quality ceiling.

Listening: http://localhost:8766/audio/voice-diagnostics/index.html . Builder:
`python -m evals.build_voice_diagnostics --quality-job 7414595 ...` . Only completed
local recordings appear. NATF1/NATM1 auditions in 7414595 finish, but still give
incorrect bread-staling answers; changing voice does not fix intelligence.
Custom-prompt attempts expose missing `pyloudnorm`, then its `future` dependency;
the job-local overlay now supplies both and checks imports before model loading.
7415052 produces a real custom synthetic-marin audition; its factual answers
remain poor. The owner accepts its voice quality; factual answers remain a failure.
Failed and completed conditions remain separate, immutable artifacts.

### Training supervision corrected, without claiming a trained improvement

The previous result-weight limitation below is now fixed. Private and clarify
scenes can weight exactly the checked phrase's actual native subwords, verified
against independent measured words and the result-ready time. Missing/reordered
tokens or premature phrases fail closed. Cancellation keeps its last-utterance
acknowledgement scope. `--result-scope last-utterance` explicitly reproduces the
historical objective; the new default is `measured-phrase`.

The regression reproduces the old whole-stream mistake before the fix, then passes.
CPU audit 7414927 checks all 23 eligible TRAIN/validation scenes, excluding sealed
test labels. The six natural private teachers now select 2–3 checked subwords,
rather than weighting their entire 34–55-token responses. A matched 128-update pilot 7415494 has now completed with this revised objective.
Fresh-process reload text/audio errors are 0.0; adapter SHA
`0263c20d64a7649c9efd43a659e4fb31503e36c553071128531f78738577dc9f`.
Development replays: private 7415517, simple 7415519, clarify 7415521, cancel 7415522.
No generated-behavior improvement is inferred from training/reload. Current relevant
local suite: 36 tests pass.

### Lychee native voice accepted; speed and delegation remain open

Lychee-FD releases a 13B native duplex checkpoint plus a separate Step-Audio-2-mini
vocoder and training code. See the [official model card](https://huggingface.co/HIT-TMG/Lychee-FD)
and [source](https://github.com/HITsz-TMG/Lychee-FD). Asset job 7414609 pins model
`ac06087bef70b33d4dfebf63a3eadadb4e024f33`, vocoder
`e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36` and source
`7bb1068c59b578e1f8e65fb762cd26708c14cc6d`, recording every file hash.
The packaged HF auto-map points at the simplex class without the checkpoint's
acoustic/control/merge branches; its offline runtime also imports an absent
module. Our explicit bridge selects the released native training implementation
and rejects every missing, unexpected or mismatched weight. No silent fallback.

Environment attempts 7414719/7414866 fail on an unused optional TorchAO API, then
missing datasets. A fresh isolated overlay explicitly disables only optional
quantization discovery for ordinary BF16 loading, adds dataset dependencies and
checks actual native source imports on CPU before releasing the GPU dependency.
7415053 passes all dependency and actual native-source imports. CPU preflight
7415211 checks all 982 checkpoint tensor names/shapes, with no GPU/weight loading;
every one matches. GPU 7415227 strictly loads with no missing, unexpected or
mismatched weights, but produces no speech. This is not a valid model-quality
comparison: the wrapper inherited `align_audio_input=False`, while the published
app explicitly uses `True`, alternating audio-patch/pad slots. The real CPU clock
regression 7415324 must reject five input slots per 400 ms and accept ten, matching
the 25 Hz control clock. CPU check 7415324 passes the real native preprocessing red/green check. Corrected
replay 7415325 completes, retaining all native text/speech/control token traces.
All 982 tensors load exactly. It speaks four answers, getting bread staling wrong;
the owner accepts its voice quality as matching the GPT-Live reference. Generation
including the separate vocoder takes 229.47 s for 70 s of input. This offline
result does not pass real-time or tool-selection qualification.

Fresh GPT-Live 7415050 completes with 753 packets, zero API errors/delegations,
and correct science answers in its transcript. The unbuffered capture has 76
underruns / 1.312 s; 160 ms startup buffering leaves 66 / 1.152 s. Thus the simple
startup buffer does **not** fix this network stream. Its exact PCM packet stream
is also exported separately for voice-only audition, excluding transport gaps
without claiming real-time playback. The focused page now starts with that
reference, original PersonaPlex NATF2 and the custom PersonaPlex condition;
technical controls/older candidates are collapsed below.

MiniCPM custom-voice replay 7415165 answers only the first science question in
its generated text. Compute p50/p95 0.452/1.836 s per one-second input;
maximum backlog 6.229 s. Source capture is reproduced exactly from packet
receipts; buffering leaves eight underruns. This is neither a content nor a
latency pass, regardless of its voice audition.

### Retrieved sampling results and protected backups

Both previously unread temperature-.2 jobs completed. Audit 7414528 confirms
7412588 says only “yellow”, omitting “circle” and the science answer (WER 0).
7412590 has bad spoken content (WER 0.2174, valid time bounds), premature private
claims and a late native cancellation request at 22.58 s after the user's cancel
ends at 19.42 s. Actual cancellation eventually succeeds in the application, but
after the private reference was already queued at 22.48 s. Neither replay passes.

All 48 mixed code/audio payloads remain backed up. Previously interrupted parent
metadata retrieval is complete; all three parent conversion ingestion hashes
match their cache receipts. Canonical cluster paths/hashes remain unchanged.
Accounting snapshot through 7415227: 89 terminal jobs with allocated GPUs,
8,652 physical GPU-seconds / 2.403333 hours. It excludes CPU allocation billing,
interactive premiums and the pending corrected replay, and is not a remaining
balance. [Official accounting](https://docs.isambard.ac.uk/user-documentation/guides/accounting/)
uses node hours (four GPU-hours per node hour) and exposes actual project credits
in the Awards Portal. That separate portal is not signed in; the owner's reported
remaining allocation/expiry is still the planning input, not a newly verified balance.
No additional authorization is needed for this work. Normal bulk/training stays
on workq; the interactive reservation is only bounded diagnosis/audition work.

## Earlier execution — 14 October, native teacher pilot evaluated

**Current verdict:** no product model qualifies. Native training and exact
serialization now work; generated speech still fails task selection, waiting,
clarification and cancellation. Do not promote teacher-forced loss or component
accuracy to model acceptance.

### Data and checkpoints

Repaired TTS verification 7407554 approves 38/40 scenes: 23 train, 8 validation,
7 test, retaining all four behaviors in every fold. Original rejected scenes
remain excluded. Frozen native Mimi/ARC cache 7408223 preserves every word.
Read-only audit `recovery-native-code-timing-base-20261014` checks actual first
subword frames for all 38 scenes: maximum offset 80 ms against the unchanged
200 ms bar. This supplements the separate measured-word grid audit.

GPT-Live teacher collection 7408766, resumed in 7409126, records twelve approved
TRAIN stimuli. Independent CPU audit 7410766 passes ten recordings (six private,
four simple); two fail exact spoken-word position matching and remain excluded.
Immutable conversion/composition 7410969 gives 33 train / 8 validation / 7 test.
Validation/test content and membership remain unchanged. Raw recordings stay
unapproved; only derived rows passing every source, word, fact and timing guard
enter training. Recorded input is checked against approved audio plus silent tail.

Teacher encoding 7411223 fails before codec loading: a second CUDA process cannot
open the GPU. Single-context retry 7411513 completes on a GPU reporting
`Exclusive_Process`, while retaining the actual HTTP reference endpoint. It
encodes only the ten new recordings and reuses all 38 hash-verified base codes.
Every new native first-subword clock and control-marker collision check passes.

| Checkpoint | Training | Generated development result |
|---|---|---|
| 7408323 | 32 updates, unweighted control/result objective | 7408416 omits returned private fact |
| 7410830 | Matched 32 updates, RAG weight 10/result utterance weight 3 | 7411280 guesses white and invents a book |
| 7408653 | 128 updates on original approved train fold, same weights | 7410641 uses yellow circle later, guesses red early and repeats |
| 7411514 | 128 updates on approved teacher mixture, same weights | 7411691 omits fact; corrected-context 7411889 uses it but adds unsupported claims |

All four save/reload exactly (maximum canonical text/audio logit errors 0.0),
change 674 adapter tensors and preserve frozen weights. Latest adapter SHA256:
`104bcd641351e03f8bf7da6f00477bf63c9d8a4f369a75c5d1b9bd7247753093`.
These are rank-8 native adapters (24,242,176 trainable parameters), not pretraining
or a new Qwen row/layer campaign. Candidate outcomes vary; single replays do not
establish causal learning gains.

**Objective limitation:** result weight currently covers the last assistant
utterance. Natural teachers use one measured event for their whole assistant
stream, so all their spoken words receive that weight. Their reported
`result_word_nll` is not private-fact accuracy. Refine this scope before enlarging
or increasing the objective.

### Real worker and generated behavior

Actual `gpt-6-luna` backend regression 7411885 passes eight policy/tool contracts,
with three real read-only record calls. This is backend development evidence.
Native model RAG requests still require a stronger semantic worker to select
operations/arguments; do not call this a native-only intelligence result.

Cancellation 7411020 exposes a real application bug: the API worker cancels its
own newly reserved dispatch, leaving older lookups deliverable. Current dispatch
is now excluded from selectable tasks and rejected as a cancellation target.
Batch cancellation validates every selected existing ID before changing state;
related duplicates can be cancelled together while unrelated work is preserved.
The captured self-cancellation regression fails before the fix and passes after.
Actual API cancellation component 7411442 passes. Native replay 7411443 never
requests cancellation, so the repaired application cannot honour that request.

Worker context previously flattened all user turns into one message, producing
repeated private reads when a science question followed. Separate chronological
utterances and already-generated assistant chunks now preserve the latest user
request. Ready-phase metadata is exposed without leaking private values early.
The helper excludes future words and scripted assistant answers. Context remains
a measured transcript oracle, not real streaming ASR or microphone acceptance.

| Latest teacher checkpoint, corrected context | Evidence | Verdict |
|---|---|---|
| Private 7411889 | One actual private read; yellow circle spoken after ready; unsupported private claims and extra request | Fail overall |
| Simple 7411886 | Two unnecessary native work requests; unclear salt answer | Fail |
| Clarify 7411887 | Omits missing-project clarification; invents book discussion; later reads correct record | Fail |
| Cancel 7411888 | No cancellation dispatch; premature checked/private claims; result delivered after cancel request | Fail |

The four replays finish on normal/interactive nodes, including two simultaneous
jobs on nid010336. Each HTTP server now owns an ephemeral loopback port, preventing
cross-job endpoint collisions. Single-context voice execution supports exclusive
GPUs. Compute p95 is 22.36–24.35 ms per 80 ms frame; maximum backlog across this
matrix is 33.4 ms. This measures software replay throughput, not real microphone
or general interaction acceptance.

Independent speech audit 7412064 confirms the simple/clarify/cancel speech with
valid time bounds. Private 7411889 has an invalid hallucinated tail, so discard
its aggregate WER and do not promote that ASR result to quality evidence. Its
bounded early word intervals show yellow circle at 10.48–11.08 s after the
actual result was ready at 8.55 s; no whole-task pass follows. GPT-Live development
reference 7408224 uses the actual tool, answers science and returns yellow circle.
It is one successful development scene, not general qualification.

Captured-inference proof 7410675 passes with deterministic math SDPA: repeated
control/prefix errors 0.0, post-reference difference 5.6953. Earlier paired and
single-slot fused-attention tests are inconclusive. Production attention remains
unchanged; this isolates reference transport, not free-running task success.
27 current local causal/context/cancellation/alignment regressions pass.

### Earlier access interruption and continuation — superseded above

Listening: http://localhost:8766/audio/voice-comparison/index.html . Top row is
GPT-Live / PersonaPlex / corrected MoshiRAG on the same four science questions,
assistant output only. Owner voice preference remains pending.

Clifton certificate expired at 19:30:50 UTC / 21:30:50 Amsterdam on 14 October.
SSH now fails `Permission denied (publickey)`. Renewal was launched with
`clifton auth --open-browser true --show-qr false --write-config false`; the
Chrome Isambard page awaits the owner's identity-provider sign-in. If that flow
expires, restart `clifton auth`. Do not infer authentication from elapsed time.

Inference-only text-temperature .2 probes 7412588 (private) and 7412590 (cancel)
were submitted before access failed. Their status/results are unread. After
reauthentication, check existing jobs, fetch COMPLETE logs/artifacts, then audit
speech before comparing. Do not submit duplicates merely because access lapsed.
No new compute permission is needed; the owner's authorization persists.

Local backups contain all 48 code tensors and all 48 audio/sidecar scenes with
matching content hashes, plus checkpoints, raw teacher audio and evaluations.
Parent approval/cache metadata backup is incomplete because SSH expired during
retrieval. See `runs/recovery-20261014/local-backup-report.json`; retrieve parent
metadata after login. Preserve canonical source hashes and recorded cluster paths.
Latest complete allocation snapshot through 7412064: 80 terminal GPU jobs,
7,692 allocated GPU-seconds / 2.136667 GPU-hours, including failures/cancellations.
This excludes the last two probes, billing premium, CPU/API costs and does not
verify the remaining project balance or expiry date.

Next priorities: choose voice backbone using the audition; broaden real grounded
waiting/clarification/cancellation examples; test temporal jitter/counterfactual
values against recurring late requests; refine private-result supervision; then
replace transcript oracle with actual streaming ASR and fresh full-task gates.
Keep test episodes sealed. Do not scale solely from loss or intent-head accuracy.
Normal bulk/training remains on workq; interactive is bounded development only.

## Earlier execution — 14 October, after native optimizer recovery

The conditioned four-update engineering checkpoint is now verified. Fresh
process 7407067 reloads all native adapters while preserving the speaker
conditioner's mixed precision; text and audio logits both match exactly
(maximum absolute error 0.0). Prefix-aware native tests pass all six cases in
7406692. The earlier optimizer job 7406181 failed at reload, not during its four
updates; the repaired reload uses that saved checkpoint without retraining it.
This is serialization evidence, not audible quality improvement.

Fresh synthetic corpus generation completed in resumed CPU job 7407148, reusing
181 unfinished-batch clips and generating one missing clip. The corpus has
40 conversations, 182 utterances and 733.76 seconds of conversation canvas.
Source is pinned `gpt-4o-mini-tts-2025-12-15`; assistant voice marin is shared
across folds, while script families and user voices are split in advance.
The source directory is `runs/recovery-curriculum-7406244` on Isambard. None of
this corpus is approved merely because rendering completed.

Verification job 7407250 was submitted to workq. It checks every user and
assistant clip with Whisper timestamps, requires native-conversion WER <=0.08,
and requires exact assistant word-position agreement before assigning measured
onsets. It replays the actual emission grid and retains the existing 200 ms
timing bar. Rejected conversations cannot enter conversion or training. Approved
stereo is assistant left / user right. The conversion retains sealed train,
validation and test membership and records source/artifact hashes.

Next: frozen Mimi/ARC encoding, then a bounded 32-update native LoRA pilot from
the released RAG base with zero-output adapter initialization. All reference
rows enter only after their recorded result arrival. Include general answers,
clarification, private facts, pending conversation and cancellation. Require
source approval/hash checks again at startup, unchanged frozen weights, finite
gradients and fresh-process reload. Teacher-forced validation is diagnostic;
test episodes stay outside training/selection. Qualify generated speech and
actual semantic dispatch separately before enlarging the corpus or GPU campaign.

Normal training and corpus jobs remain on workq. Interactive reservation is for
bounded debugging/exploration, not bulk generation or production training.
GPT-Live access and speech replay are verified (7400621/7400991); its worker path
still needs an actual delegation call. Qwen's corrected strict routing probe
7402864 achieved 11/20 manually labelled development cases, with 8 invalid
outputs and 6 missed required delegations. It is not a qualified semantic router.
PersonaPlex 7402673 produced an audition but answered bread staling incorrectly.
The backbone decision remains conditional on audible and task-quality evidence.

## Architecture and training direction — clarified 14 October 2026

**Decision:** adapt a released, pretrained full-duplex speech model and connect
it to a separate reasoning/tool worker. This is supervised fine-tuning, not
pretraining from scratch. MoshiRAG is the working reference-integration candidate;
the final voice backbone is still conditional on audible quality and interaction
comparisons, including PersonaPlex. Do not assume their conditioning interfaces
or checkpoints are interchangeable.

The old Qwen path trained new control-token embedding/output rows while freezing
the language core. It established timing behavior but not useful self-generated
spoken content. Preserve it as a baseline; it is not the main scale-up recipe.
Qwen remains a candidate backend/teacher and an unchanged speech/content
reference. No new Qwen layer-training campaign has been started in recovery.

The completed 64-step adaptation pilot trained LoRA adapters on plain Moshi,
using 16 reused conversations. It verified gradients, frozen-weight preservation
and serialization, not quality improvement. The corrected reference replay used
released MoshiRAG weights with inference-harness repairs, not newly trained
MoshiRAG weights. A causally timed training adapter now passes native CPU tests;
released-weight backward and native conditioned optimizer/save/reload now pass.
Generated task behavior remains unqualified; the current execution section above
records the fresh-data and teacher pilots.

After backbone selection, start with adapters inside the pretrained transformer.
Expand to selected pretrained blocks or broader fine-tuning only when evaluations
show adapters are insufficient. These remain fine-tuning approaches. The target
is useful speech plus waiting, interruption, cancellation, and returning to late
results without fabricated early answers. Do not use lower training loss as
acceptance or scale the tiny reused pilot corpus.

The owner authorized GPT-Live as the reference on 14 October. Benchmark GPT-Live
itself, separately from Realtime API measurements. Use matching conversational
tasks and, where possible, the same client-delegation worker/facts/delays to
isolate voice and interaction quality. API/account access is not yet verified.
Official architecture and client delegation:
https://developers.openai.com/api/docs/guides/live
https://developers.openai.com/api/docs/guides/live-delegation

Owner listening feedback on the corrected replay: “sounds okay not great”.
Record this as one informal audition, not a completed blind voice comparison.

Immediate sequence: (1) GPT-Live reference plus voice-backbone comparison;
(2) actual context-based backend requests and results with task IDs/revisions;
(3) fresh split conversations with timed result supervision;
(4) verified timed-reference backward/save/reload and a bounded adaptation;
(5) joint content/voice/duplex/delegation evaluation, then larger training.

## What the first recovery checks established

### Base intelligence and delegation selection — 14 October

**Plain words:** released duplex weights exist, but sufficient judgment for our
product has not been established. Moshi has an official adaptation recipe;
PersonaPlex demonstrates role following and conversational knowledge, while its
release does not establish a matching turnkey fine-tuning recipe. MoshiRAG
already accepts asynchronously arriving reference text. This makes it an
integration candidate, not a proven intelligence or voice winner.

Delegation selection is now a separate gate from successful result delivery.
Evaluate four kinds of requests: simple answers available to the model; private
or current facts requiring a tool; multi-step reasoning requiring a worker;
and ambiguous requests requiring clarification. Include follow-up corrections,
negation, and requests to cancel. Report missed delegation, needless delegation,
unsupported answers, clarification quality, and complete task success separately.
Do not use reported model confidence as proof it knows an answer.

Train the selection decision together with the spoken behavior: acknowledge the
request, ask for missing inputs, continue hearing, wait without inventing a result,
and incorporate or correct the answer when evidence returns. Application task
IDs, revisions and result guards complement learned selection; they do not prove
the speech model learned it. A separate semantic router is a possible bridge,
but any such result must be labelled a hybrid rather than native model delegation.

Qwen's existing control experiment remains a timing baseline. Native Qwen's
coherent answers justify keeping it as a backend/content candidate. Restoring its
language and speech components does not itself solve continuous incoming audio,
causal training, or concurrent task results. A deeper Qwen adaptation is a valid
alternative if the native duplex candidates fail joint gates; the previous
new-token-row recipe alone is not that alternative.

Cluster access restored and commit `ad2c474` synced with matching SHA256. New
comparison scripts are committed at `9cfa3af`; access launcher repaired at
`7e41781`. CPU job 7400572 stopped before making an API request because of a
malformed shell line; replacement CPU job 7400621 submitted. PersonaPlex job
7400573 is capped at one GPU for 15 minutes and uses pinned existing source and
weights, approved `dense_dialog-001` audio plus ten seconds of silence. This is
an offline development audition, not a fresh holdout or live latency benchmark.
Results and job accounting will be appended after complete logs are retrieved.

**First GPT-Live reference completed:** access job 7400621 verified an actual
`gpt-live-1` session. Replay 7400991 finalized successfully with 77 seconds of
session usage, 752 returned audio packets, zero delegation requests and zero
API errors. All four user questions appear in the received transcript; the
assistant transcript gives substantive explanations of lightning, refrigeration,
starch retrogradation and noise cancellation. This development transcript
contrasts with MoshiRAG's wrong bread explanation, but does not constitute blind
voice grading or a fresh general intelligence benchmark. No backend request
occurred, so this run does not validate the new worker path or delegation judgment
on tasks requiring help. There was no supplied answer bank or injected reference.

Original PCM/WAV, transcript fragments, complete timestamped session events and
usage receipt are local and on cluster under `runs/gpt-live-replay-7400991/`.
The restored listening gallery exposes both tracks:
http://localhost:8766/batch/gpt-live-replay-7400991 .
PersonaPlex 7400573 remained pending for resources at the latest check; do not
claim its refreshed comparison has run. No additional GPU allocation has yet
been recorded for today's probes. API seconds are separate from GPU accounting.

**Training seam inspected:** pinned MoshiRAG `LMModel.forward_text` accepts
streaming reference conditioning only when both the condition and sequence
length are one. Passing its inference conditioner unchanged into whole-sequence
training will assert. Implement and compare a causally aligned whole-sequence
condition against one-frame execution before gradient/save/reload checks;
never broadcast a late result across earlier training frames. The native CPU
proof and queued full-weight test described below now address this seam; there
is still no completed optimizer run with timed references.

Sources: [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/),
[MoshiRAG release](https://github.com/kyutai-labs/moshi-rag),
[GPT-Live client delegation](https://developers.openai.com/api/docs/guides/live-delegation).

| Check | Evidence | Meaning |
|---|---|---|
| Native Qwen content and voice | Job 7377200: three questions, each in text and audio; all six produce coherent generated text and nonempty speech. Job 7377265: ASR word error 0.0 for all six. | The untouched base can answer and speak. These development questions are a component check, not a generalization benchmark or live acceptance. Human audition remains pending. |
| Native MiniCPM duplex | Job 7377673: 60-second paced replay, four generated answers; processing p50 0.477 s and p95 1.768 s per one-second input chunk; max backlog 3.095 s | Native words and speech work on the new container, but this is not real-time acceptance. Bread-staling answer incorrectly attributes it to drying. Human audition remains pending. |
| Controlled delayed context | Job 7378025: 10 s worker wait, seven processed packets (six speaking) during wait; result injected 0.467 s after ready; bread answer uses supplied mechanism | Positive reference-delivery/use check on one example. No learned dispatch, actual research or stale-result rejection tested. Max input backlog 3.913 s remains unacceptable. |
| Native Qwen speed | 12.29–17.94 seconds to finish each 3.98–8.54 second audio answer; Transformers 5.17 / NGC 25.10 / GH200 | This batch generation path is not ready for live use. These are full-generation times, not first-packet latency measurements. |
| Existing microphone clock | `python -m evals.live_contract_probe`: actual `m2.live.main`, fake model/ear, numbered input ticks; first 49 ticks have no audio feature, subsequent features are 3,920 ms old | The 80 ms decision cadence conceals four-second input buffering. This synthetic control-flow result does not measure encoder receptive field or actual microphone hardware. |
| Delegation comparison | Full logs of jobs 7333494 and 7356783, plus executable launcher regression | Both scripts lost `--lessons`, `--init-ckpt`, and `--out`. Both consumed the default merged shard selection and saved to `runs/p1-v1`. The claimed mixture comparison and held-out designation do not establish those hypotheses. |
| Launcher repair | `python -m unittest evals.harness.test_slurm_launch_contract` failed before the repair and passes after | The tracked launcher now passes selection/init/output arguments and finds prefixed held-out IDs. It has not been used to retrain; the old negative results have not been magically corrected. |
| Demo input/playback repair | `python -m unittest evals.harness.test_live_inputs`: red before, green after | Unverified fixtures now stop before model loading; answer clips retain their real length instead of one billion padded samples (about 17 hours at 16 kHz). The clock probe still correctly fails after these repairs. |
| Environment | Job 7377265: clean container CUDA works; xformers absent | Qwen's overlay is working. MoshiRAG needs a separate dependency setup and ARC attention-kernel check. |
| Data availability | Primary-source review in the linked research note | DuplexChat is metadata and a reconstruction pipeline, not a downloadable collection of prepared two-channel audio. |

Local evidence: `runs/recovery-20261013/`, `runs/native-content-7377200/`,
`runs/minicpm-recovery-7377673/`, `runs/minicpm-recovery-7378025/`.
Generated speech is exposed by `python -m viz.listen --runs runs --port 8766` at
`http://localhost:8766/batch/native-content-7377200` and
`http://localhost:8766/batch/minicpm-recovery-7377673`; delayed-context comparison:
`http://localhost:8766/batch/minicpm-recovery-7378025`.

Eight bounded jobs consumed approximately **0.143 allocated GPU-hours**
(514 GPU-seconds), including failed environment attempts. Jobs 7377148, 7377200,
7377265, 7377413, 7377469, 7377588, 7377673 and 7378025 are all terminal.
This is Slurm allocation time for these jobs, not the project's billing balance.
The initial eight jobs are terminal. Current follow-on job states are tracked
below; their time is not included in this initial 0.143 GPU-h figure.

## Direction and decision gates

**Plain words:** preserve working speech from an existing duplex base, then teach
the missing interaction and delegation behavior. Keep the current Qwen timing
checkpoint as a reference. Do not spend the allocation scaling the old recipe
before showing that the new recipe can produce meaningful speech.

1. **Finish native baselines.** Qwen establishes content and speech separately.
   MiniCPM-o 4.5 gets a paced replay on the same verified user audio, with actual
   input-consumption and output-availability times. MoshiRAG is the first
   delegation candidate because released weights already accept background text.
   Its live path, not its offline retrieval loop, must prove concurrency.
2. **Select by complete behavior.** Run fresh, family-disjoint content,
   interruption, backchannel, thinking-pause and background-speech scenarios.
   Archive input/output PCM, generated text, scheduling trace and task events.
   Preserve model and source revision IDs and include cold and warm runs.
3. **Prove adaptation at small scale.** One optimizer step with finite gradients,
   then fit 16 verified conversations, reload the checkpoint, and evaluate with
   its own generated history. Report exactly which parameters train. Native
   speech, content and interruption behavior must survive before scaling.
4. **Expand the right data.** Add varied conversations and timed background
   results only after the baseline exposes its weaknesses. Split by source
   conversation, script family and speaker before synthesis or augmentation;
   hash duplicates across splits. Timing derived from the encoder's actual
   availability must match training and inference. Existing quality stamps stay
   mandatory; no unverified reconstruction enters training.
5. **Scale supervised adaptation.** Use measured frames/second, memory and
   learning curves to choose the GPU count and data amount. Compare small
   backbone adaptation, staged behavior training and separate control output
   where appropriate. Preserve an unchanged reference in each evaluation.
6. **Integrate real work and evaluate jointly.** Background requests carry a
   conversation snapshot, task ID and revision. Late results from cancelled or
   superseded tasks cannot become spoken facts. Test actual bounded tools,
   missing inputs, failures, corrections and streamed partial results while
   speech continues. Provider integration follows the same interface.

Reinforcement learning is conditional on coherent supervised behavior and a
reward that scores meaning as well as timing. Training from scratch is not the
default response to this prototype's failures.

### Provisional engineering acceptance bars

**Plain words:** these are targets to preregister for the next suite, not scores
already achieved or claims of parity with commercial systems.

- Content: at least 16/20 fresh spoken questions answered substantively and
  audibly without supplied answer text; report semantic errors separately from
  speech intelligibility. Human listening accompanies ASR.
- Duplex: at least 16/20 mid-speech corrections incorporated correctly; initial
  interruption-stop target p95 <= 500 ms. Report tolerated backchannels,
  premature replies and ignored interruptions separately.
- Serving: ten-minute sessions with no increasing input/playback backlog;
  warm response-start target p50 <= 700 ms and p95 <= 1,500 ms. Measure at the
  audio boundary, not only at the model's forward call.
- Delegation: 20 tasks with results delayed 0.5/2/5/15 seconds; at least 18 correct
  incorporations, zero fabricated early completions and zero applications of
  cancelled/superseded results. Measure each delay group and continued speech.
- Release evaluation: expand beyond these small engineering probes; seal a new
  test set before candidate tuning. Previously evaluated and accidentally
  trained-on lessons cannot serve as a fresh reserve.

## Compute and calendar

### Seven-day execution window — owner confirmed 14 October

**Plain words:** the owner confirmed seven days remain and authorized all
available compute needed for the best result. Treat 21 October as the working
end date; the precise cluster expiry time and authoritative remaining balance
are still not verified. No further approval is needed for ordinary bounded
experiments, preparation or useful scale-up within that authorization.

Use independent jobs in parallel. Increase training GPU count from measured
throughput and data volume once the conditioned training and checkpoint path
pass; allocation size alone is not evidence that repeating the old corpus will
help. Preserve time to export the best checkpoint, dependencies, logs and demo
before cluster access expires.

| Window | Intended work | Required evidence |
|---|---|---|
| 14–15 October | Backbone comparison, causal training proof, actual worker and fresh-data preparation | Audible outputs, native conditioning parity, real-weight finite gradients, saved/reloaded adaptation, provenance and split manifests |
| 15–17 October | Timed-result and delegation-selection supervised pilots | Fresh improvement in answers, waiting, correction, cancellation and routing; no lost duplex capability |
| 17–19 October | Parallel selected training and data expansion | Useful learning curves, sufficient varied approved data, distributed execution correctness and measured scaling |
| 19–20 October | Independent joint evaluation and serving integration | Task completion, audible quality, interruption and backlog measurements with failures retained |
| Before expiry | Freeze and export best validated result and reproducible demo | Checkpoint hashes, standalone environment, complete evidence and copies outside Isambard |

These are execution targets, not completed milestones or promised quality scores.
At today's queue check there were no idle nodes; many nodes were down/reserved.
Three independent single-GPU probes are queued: PersonaPlex 7400573, actual
MoshiRAG conditioned backward 7401863, and native Qwen policy routing 7401909.
Do not count pending jobs as allocated GPU consumption.

**Causal training implementation:** `evals/moshi_timed_reference.py` maps encoder
rows to the first eligible model frame, clears on an explicit cancellation time,
and rejects ambiguous overlapping deliveries. The training wrapper adds the
timed rows before the temporal transformer without changing checkpoint keys.
CPU job 7401774 passed all three tests on the actual pinned MoshiRAG model code:
whole-sequence versus one-frame conditioning, future-result non-leakage, finite
gradients including checkpoint recomputation and saved/reloaded tiny-model logits.
Initial CPU job 7401729 failed because its tiny depth model used an unsupported
one-codebook configuration; the corrected model uses two. Strengthened effect
checks passed in rerun 7401911, including unchanged earlier logits and changed
later logits when reference rows arrive. Tiny-model proof is not full-weight adaptation.

GPU proof 7401863 uses the actual audited ARC HTTP encoder and released speech
weights with a short approved development code sample. It checks backward
gradients, performs no optimizer step and saves no candidate. Native Qwen
7401909 tests 20 manually specified text-policy cases (answer/clarify/delegate/
cancel), not spoken routing or task completion. Both scopes must remain explicit.

**Plain words:** the owner reports more than 16,000 GPU-hours available. Treat
16,000 as a working ceiling, not a target to burn. The recorded allocation expiry
is 21 October; expiry and remaining project balance still need current portal
confirmation. SSH, account `brics.u6si`, CUDA, and job submission were verified
live on 13 October. Slurm exposes this user's jobs, not an authoritative remaining
project balance.

| Maximum provisional envelope | Work | Release condition |
|---|---|---|
| 100 GPU-h | Baselines, environment and profiling | Bounded individual jobs, useful content and honest timing evidence |
| 400 GPU-h | Data verification and train/inference alignment | Provenance, timing checks and disjoint split manifest |
| 1,000 GPU-h | Small adaptation pilots and controlled comparisons | Reloaded free-run improvement without loss of baseline abilities |
| 10,000 GPU-h | Selected supervised adaptation | Measured throughput, positive scaling signal, fixed validation protocol |
| 1,500 GPU-h | Robustness, independent evaluation and serving optimization | Successful content, duplex and delegation integration |
| 3,000 GPU-h | Reserve | Specific evidence-backed bottleneck; otherwise leave unused |

These envelopes do not submit jobs. Eight full days at 16,000 GPU-hours would
require about 84 GPUs continuously; queue capacity, distributed training support
and enough diverse data are separate constraints. A deadline does not justify
silently repeating a tiny corpus or launching an unverified distributed run.

## Execution tracker

### Interactive reservation follow-up — 14 October

Plain words: the faster short-work route is verified live. See
[interactive scheduling research](research/2026-10-14-isambard-interactive-scheduling.md).
Native CPU checks still pass, but full-weight training has exposed another seam:
the released first-speaker prepend condition is an embedding, while the native
whole-sequence forward attempts to concatenate it with integer codes. Job
7402644 failed on a 4096-versus-128 dimension mismatch before backward. Do not
claim full-weight gradients or omit this prefix just to pass. A prefix-aware
training implementation must be checked against serving execution.

PersonaPlex 7400573 failed after 27 GPU-seconds during compiler warmup because
Triton could not find the runtime CUDA linker library. The existing job-local
driver-linker helper is now wired into this probe; interactive repair 7402673
completed in 67 allocated GPU-seconds. It generated 70 seconds of audio in
45.095 seconds including model load. This is offline throughput, not live
latency. Its bread answer incorrectly attributes staling to moisture loss;
voice quality requires audition. Both tracks are available at
http://localhost:8766/batch/personaplex-recovery-7402673 .
Original pending proof 7401863 was held, then cancelled after its replacement
started. Qwen replacement 7402713 was accepted in the interactive reservation;
its held workq duplicate 7401909 was cancelled. Monitor hidden jobs with
`squeue --me --all`. Qwen 7402713 subsequently hit the host-memory limit while
loading weights, before any routing result. Its 64 GiB request is now 110 GiB,
matching the larger memory needed by the native baseline; retry uses the same
one-GPU diagnostic, not a larger training run.

The interactive pool is for short development work, not bulk training. Long
training stays on workq. Official documentation states a 50% node-hour premium;
track actual Slurm allocation and billing fields separately, not as an assumed
remaining GPU-hour balance.

| Work | State | Next evidence |
|---|---|---|
| Current handoff, source and job-log audit | Done for initial scope | Continue to challenge claims against actual commands/results |
| Native Qwen speech diagnostic | Complete, job 7377200 | Human audition; broader fresh suite only if pursuing this candidate |
| Generated-speech ASR check | Complete, job 7377265 | Six samples at WER 0.0; no claim of human voice-quality review |
| Microphone schedule repro | Complete; known failure retained | Replace the delayed stream contract in the selected native path |
| Delegation launcher repair | Regression passes | Do not reuse the historical ratio verdict as evidence |
| MiniCPM native duplex replay | Complete, job 7377673; earlier jobs exposed missing decoder dependencies | Tested process-local PCM WAV reader; TorchAudio features and container Torch unchanged. Improve backlog and assess interruptions before acceptance. |
| MiniCPM delayed context injection | Complete, job 7378025; reference used, processing continues | One fixed result only. Add real dispatch, delayed/cancelled results and fresh tasks; reduce backlog. |
| MoshiRAG live delegation baseline | Pinned weights downloaded (7378305); environment 7379661 passed actual ARC CUDA attention (max SDPA difference 0.0). Official Llama tokenizer access remains denied | Authorize account; verify tokenizer, then live delayed-result replay |
| Training dataset/split and pilot | Data 7379026 passed. Pilot 7379340 completed four finite optimizer steps, changed 674 adapter tensors and preserved frozen weights; reload comparison failed its 0.05 tolerance | Exact-weight/mode diagnostic 7379832 and native speech comparison 7379833. These 16/4 folds are diagnostics, not fresh generalization. |
| Broad training | Not started | Only after small reloaded free-run pilot passes |

Source-backed selection and execution details:
[recovery base assessment](research/2026-10-13-recovery-base-selection.md).
The Thinking Machines reference describes the behavior target; reproducing its
276B-parameter system is not a credible use of this allocation.
[Primary reference](https://thinkingmachines.ai/blog/interaction-models/).

### Exact current continuation commands

**Plain words:** these commands validate local repairs and reproduce bounded
cluster probes. Inspect the latest job and ledger before submitting duplicates.

```sh
.venv-api/bin/python -m unittest evals.harness.test_live_inputs \
  evals.harness.test_slurm_launch_contract evals.harness.test_reference_wav
.venv-api/bin/python -m evals.live_contract_probe
# The clock command intentionally exits 1 until the delayed path is replaced.
python3 -m viz.listen --runs runs --port 8766
```

Deploy only through local commit/push, cluster `git pull --ff-only`, and checksum
comparison. Cluster working directory: `/projects/u6si/duplex-live/repo`.

```sh
sbatch evals/native-content-probe.slurm          # 1 GPU, 45 minute cap
sbatch evals/minicpm-recovery-probe.slurm        # 1 GPU, 30 minute cap
sbatch --export=ALL,RESULT_FIXTURE=/projects/u6si/duplex-live/repo/evals/fixtures/bread_background_result.json \
  evals/minicpm-recovery-probe.slurm            # controlled delayed context
sbatch --export=ALL,PROBE_DIR=/projects/u6si/duplex-live/runs/native-content-JOB \
  evals/recovery-environment.slurm              # 1 GPU, 15 minute cap
```

Fetch complete logs before filtering. Probe directories use job IDs and refuse
overwrite. The MiniCPM launcher creates a job-specific dependency overlay and
retains the base Torch/CUDA packages. Its WAV-reader accommodation is necessary
because [TorchAudio 2.9 switched its load function to TorchCodec](https://docs.pytorch.org/audio/stable/generated/torchaudio.load.html);
the attempted [matching TorchCodec version](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)
had no ARM CPU wheel in the queried index (job 7377588). Package import is
insufficient: actual reference decoding, model generation and output audition
are the checks.

## Immediate continuation

**Plain words:** the next independent work is the MoshiRAG live baseline and a
small native Moshi training smoke. MiniCPM now provides a working comparison,
not a final architecture selection. Its general fine-tuning scripts do not
yet establish timed duplex training support.

1. Build an isolated MoshiRAG environment; verify ARC xformers on the actual
   GH200 before downloading more than the required pinned weights. Run the
   live path with delayed local references; do not treat its blocking offline
   retrieval example as concurrent delegation.
2. Prove the official native Moshi training path on 16 verified conversations:
   validate assistant-left/user-right audio and word alignments, perform a
   backward step, save/reload, then generate with its own history. Report
   missing checkpoint keys rather than accepting a partial load silently.
3. Prepare a fresh split across scenario families and speakers before candidate
   tuning. Audit the existing synthetic material for suitability; the prior
   515-example collection is not an untouched evaluation reserve.
4. Profile MiniCPM speaking versus listening compute and playback buffering.
   One-second input chunks and the measured 3–4 second backlog currently miss
   the target; do not label median compute alone as conversational latency.

See the research note for exact official sources, data schemas and environment
constraints. Human audition of all new audio remains open.

## Native Moshi continuation — verified training, quality remains open

**Plain words:** native Moshi now trains, saves and reloads correctly on GH200.
A 64-step pilot reduced diagnostic loss, but generated answers remain weak.
Do not scale this corpus or select a model on this loss result.

- Model acquisition 7378305 pinned and hashed native Moshi, MoshiRAG, ARC and
  local ASR: `runs/moshi-models-7378305/model-manifest.json`. Meta's official
  Llama 3.2 tokenizer returned `GatedRepoError`; reference encoding awaits
  owner-approved access. Native Moshi does not need that tokenizer.
- Conversion 7379026 accepted 20 conversations (16 train / four diagnostic
  evaluation), totaling 2,237.36 seconds. Highest retained utterance WER: 0.05.
  Twenty-two candidates were rejected. Actual durations are 104.84–116.20 s;
  native segments round to 120 s. WhisperTimestamped supplies assistant word
  intervals. Source approval, source/converted hashes and word audits are
  rechecked before training. These reused examples are not a fresh family- or
  voice-disjoint generalization test. Human audition remains pending.
- Native trainer commit `2acc879fe7c48f885a18f6cc9548bccb2674d87b` and Moshi
  `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` run without replacing container
  Torch/CUDA. LoRA rank eight trains 24,242,176 parameters / 674 tensors.
  Frozen base weights remain byte-identical; adapter gradients are finite and
  nonzero. Training peak allocation: 23,544,080,896 bytes.
- Four-step job 7379340 trained and saved, then failed its provisional BF16
  logit comparison (text 0.125 / audio 0.375). Diagnostic 7379832 established
  exact saved/reloaded adapter and base values. The numerical discrepancy's
  cause is not fully established. No passing stamp was assigned to that job.
- Jobs 7380342 (four steps) and 7380505 (64 steps, learning rate 1e-5) passed
  exact saved-adapter hashes and fresh-process canonical logit comparison:
  float32, TF32 disabled, checkpointing disabled, compilation disabled;
  maximum text/audio difference 0.0 against a 0.001 bound. This verifies
  serialization under that execution mode, not BF16 logit equivalence.
  Voice generation still uses BF16. Saved adapters are not optimizer-resumable
  training checkpoints.
- Job 7380505 diagnostic loss was 1.9321053 versus 2.20383549 after four steps.
  First/last epoch mean training losses: 2.082494 / 1.813906. Sixteen passes
  over sixteen examples do not establish generalization. Its spoken answers
  contain clear content errors; no quality improvement is established.
- Its 40-second paced replay processed 80 ms frames at 0.0220 s median /
  0.0236 s p95, with maximum input backlog 0.000196 s. Those are frame compute
  times, not response-start or end-to-end conversational latency. Microphone,
  interruption and network acceptance remain untested. Actual speech audit
  7380455 confirms intelligibility on the earlier four-step comparison (WER
  0.0737 base / 0.0206 adapter), not semantic correctness.
- The latest replay code freezes inference parameters and resets sampling
  seed after warmup. Earlier saved speech comparisons predate that correction;
  they are exploratory rather than strictly paired sampling comparisons.

MoshiRAG environment 7379661 built xformers from pinned source on ARM/GH200
and verified the actual BF16 ARC attention kernel against SDPA (max error 0.0).
Pinned upstream source is imported directly because its Hatch package rejects
`../README.md`. Websockets was subsequently installed in the isolated runtime;
container Torch remains unchanged. Full ARC encoding and reference use have
not run yet.

The actual upstream reference callback reproduced an old encoding overwriting
its replacement. `evals/moshi_rag_runtime.py` now rejects superseded, cancelled
and reused-session results. Four tests against the actual upstream channel and
manager pass on the cluster, including revision propagation into the encoding
child task. These are delivery safeguards, not evidence of learned dispatch
or natural-language cancellation.

The owner completed access and cluster browser authentication on 2026-10-13.
Actual pinned tokenizer files downloaded successfully. Credentials remain in
the cluster model cache; no tokens are recorded in chat or the repository.

Reference probe 7386266 passed complete inference-weight coverage and actual
localhost HTTP encoding. All 239 required keys are present; two released ARC
tensors (`embedder.cont_tok.0.weight`, `embedder.rec_tok.0.weight`, each [1,3072])
are absent from and unused by the pinned inference source. Only those explicit
extras are omitted. Empty references intentionally emit zero vectors; nonempty
references must be finite and nonzero. Seven-, twenty- and 361-token references
returned [1,2,4096], [1,5,4096], [1,91,4096] tensors in 51–64 ms. Repeated text
produced identical values. Peak allocation: 14,932,377,600 bytes. This is an
encoder/transport test, not spoken use or delegation acceptance. Earlier jobs
7385889/7386158 failed overly strict probe assertions and remain recorded.

**Correction to all early reference-use verdicts:** the replay queued embeddings
with `update_streaming_sum_tensors`, then called `step` without the upstream
batch runner's explicit `apply_pending_streaming_sum_condition` operation.
Consequently jobs 7386447, 7387021, 7387196, 7387230 and 7387611 never applied
those reference rows to the language-model input. Their worker timing and
speech artifacts remain real, but their failure to repeat supplied facts is
not evidence against the conditioned model or encoder. Do not carry those
negative reference-use verdicts forward.

`evals/moshi_rag_reference_step.py` now applies and verifies each row before
generation, checks clearing after the reference ends, and the replay requires
queued-row and applied-frame counts to match. The regression uses the real
upstream queue/consume methods on CPU and detects an omitted application.

Separately, seeded BF16 histories varied before worker launch: longer warmup
7387021, fresh processes 7387196, and no CUDA graphs/compilation 7387230 did not
establish token-identical prefixes. In 7387230, the first 100 user-code and
traced CUDA RNG hashes matched, yet generated tokens differed from frame 72.
The cause remains unestablished. Eager processing p50 was 0.132–0.145 seconds
per 80 ms frame and backlog reached 39–49 seconds; that mode fails live use.
The corrected CPU causal audit uses the actual worker-start frame, rather
than confusing wall-clock and consumed-input time, and still rejects the
known fresh-process and eager histories. A seed alone is not a causal control.

Request-aligned 7387611 queued three fixture references after native RAG tokens
at frames 47, 231 and 439; its ten-second bread result was queued at 45.280 s.
Actual audio transcription agrees with text (WER 0.0084), but this run predates
the application fix and cannot establish reference use. The corrected run and its scope are recorded below.
These are reviewed fixed facts selected by known request order, not actual
research, semantic tool routing or generalized learned delegation.

Audio audit 7386830 transcribed the generated WAVs, confirming the core wrong
answers, but the full-clip reference transcription also hallucinated repetitive
words with timestamps beyond the recording length over a near-silent tail.
Its large WER is not proof of actual repeated speech. Original audio/report
remain intact; a separate tail-trimmed ASR diagnostic now records the inspected
input duration and timestamp validity. Human listening remains open. The WAVs
are available in the local gallery, e.g.
`http://localhost:8766/batch/moshi-rag-speech-7386447-delayed-reference`.

### Corrected request-aligned speech — application passes, late answer fails

Job **7387777 completed**. Four native retrieval-request tokens launched workers
for known questions; fixtures supply reviewed facts by request order, not actual
research or semantic tool routing. The replay verifies all **33 queued rows
were applied** to the active conditioning input before generation. Two CPU
regressions against real upstream methods pass (0.041 s), including omitted
application detection and clearing after the last row.

Median/p95 frame processing: **0.0165 / 0.0186 s** for 80 ms input frames.
Maximum input backlog: **0.0781 s**. This is paced frame processing, not
microphone/network/playback acceptance or response-start latency.

The bread worker started at 35.218 s and returned at 45.271 s; its condition
was queued at 45.280 s and applied in frame 565. The model had already given a wrong drying/gas
explanation. After arrival it only appended “I suppose”, with no correction
before the next question. **Long-delay result incorporation still fails this
case.** It must learn to defer unsupported answers and return to completed work.
Earlier negative reference-use claims from the missing-application harness are
invalid; this corrected negative result has actual row-consumption evidence.

Audio audit **7387786 completed**. Core answer words agree with text, including
the wrong bread mechanism, but the last truncated answer provokes ASR
repetitions/timestamps beyond recording length. Its full-clip WER is therefore
not valid speech-quality evidence (`asr_time_bounds_valid=false`). Original
WAVs and reports remain intact. Human listening is pending; corrected audio:
`http://localhost:8766/batch/moshi-rag-speech-7387777-delayed-reference`.
No causal paired-seed improvement or generalized delegation claim is made.

All recovery jobs are terminal as of this update. Initial probes consumed
514 allocated GPU-seconds; native continuation 2,076; reference continuation
1,346. Combined: **3,936 seconds / 1.093 GPU-hours**, across 39 allocated
single-GPU jobs. Failed jobs count; one CPU model download and two unallocated
dependency cancellations do not. This is job allocation time, not a verified
remaining project balance. No large training run has started.

Next: introduce fresh held-out reference facts (including facts unavailable
from pretraining), replace request-order fixtures with an actual background
worker, and implement/verify causal timed-reference training. Prioritize
waiting, interruption/cancellation and post-result correction, then evaluate
multiple samples with reviewed labels. Seed equality alone is not an inference
control in this setup. Do not scale the reused 16-example native corpus or
select a production model on diagnostic loss. Keep the native baselines and
all failed evidence; microphone/playback and human voice acceptance remain open.


### 15 October continuation — accepted voice paced; bridge engineering proof

**Plain words:** accepted custom PersonaPlex runs at 30.47/32.46 ms median/95th
percentile per 80 ms input frame (7415855). Maximum input backlog is 161.12 ms;
a 160 ms startup buffer leaves one 50.71 ms playback gap. This is promising
compute speed, not clean microphone acceptance. Native science answers still fail.
The replay changes no native model, prompt, sampling or generated-frame export;
additional WAVs retain real packet availability and an untimed PCM diagnostic.

Revised MoshiRAG objective pilot 7415494 still fails all four development tasks:
7415517 gives only yellow, invents/repeats room claims and rereads the record;
7415519 answers salt dissolution but dispatches two unnecessary workers;
7415521 skips clarification and invents marker facts; 7415522 never cancels.
Independent audit 7415856 has valid measured word bounds and respective WER
0.03125 / 0.02326 / 0.06667 / 0.0. Intelligibility does not rescue task failures.
Do not scale this recipe based on lower loss or exact reload.

PersonaPlex reference proof 7415929 passes on actual native single-slot generation
with CUDA graphs and identical teacher-forced histories: repeat, zero-initialized
projection and causal prefix errors all 0.0; manually enabled positive projection
changes post-result logits by 1.25. The rank-64 bridge maps frozen ARC rows into
PersonaPlex's temporal input. This proves a compatible causal data path, not learned
result use, speech quality, or native delegation. It assumes no native RAG token.
That first proof omits the voice prompt; a further run checks the exact accepted
synthetic voice and system prefix before any larger fine-tuning.


**Plain words:** native bridge proof 7427733 repeats the check with the exact
accepted synthetic voice and system prompt. Repeat, zero-projection and causal
prefix errors remain 0.0; the deliberately enabled positive projection changes
post-result logits by 1.1875. Native token 4 is `<0x00>`, not a tool command.
PersonaPlex and the approved cache have identical tokenizer and Mimi hashes.
The next bounded engineering pilot trains only the rank-64 reference bridge on
one independently verified TRAIN result phrase, with the actual native delayed
voice/system prefix captured from LMGen. It replaces old RAG markers with native
padding, checks streaming/whole-sequence equivalence before gradients, freezes
and hashes the speech weights, and requires exact fresh-process bridge reload.
No routing, generalization or free-running speech acceptance follows from that pilot.


### 15 October — delivered result still not used; new data prepared

**Plain words:** actual worker 7429163 reads yellow circle with a real receipt,
finishes ARC encoding at 10.294 s, then applies all three rows at 10.323, 10.400
and 10.480 s. Generated text omits yellow circle and instead asks the user about
the colour. Therefore the eight-step projection is not a useful result-use model.
Independent audio transcription remains to be run. Native routing is explicitly
untested: the dispatch is diagnostic/scheduled. p95 compute 31.65 ms/80 ms frame,
maximum backlog 117.19 ms, one buffered gap 7.75 ms. Voice remains audible locally.

Lychee incremental 7428971: 243.26 s / 70 s input, 174.63 s maximum backlog,
p95 1.781 s per 400 ms chunk. Encoder-memo experiment 7429187 avoids 1,279
repeated encodes, but still takes 214.04 s with 143.96 s backlog. All three native
token streams differ from baseline; pure repeated encoder/logit parity was not
established. Do not adopt the cache or claim unchanged behavior. Both HF paths
fail live speed. Accepted Lychee voice is retained as an alternative audition.

Control teachers 7429356 collect all seven approved clarify/cancel TRAIN scenes;
independent word/timing/receipt audit 7429624 was queued, result not yet retrieved.
New checks require a clarification before identification, no premature lookup,
actual cancellation receipts before acknowledgements and no cancelled fact delivery.
A separate 32-session paired-private batch is ready: four different declared
private worlds per identical approved TRAIN user question, two concurrent API
sessions, no held-out stimuli. All worlds/receipts/audio hashes are checked and
all targets remain unapproved until the same strict speech, fact and clock gates
pass. Resume preserves completed recordings and archives incomplete attempts.
Launch `evals/counterfactual-teacher.slurm` after authentication; no job ID exists.

Native delayed-history proof 7429698 compares the reusable prefix construction
against actual LMGen on a different TRAIN scene. Retrieve its full result before
using fast histories for multi-example training. Local related suite is verified
separately below; exact tests and results remain evidence, not model qualification.

**Plain words — local verification:** 41 related tests pass, including the new
control-teacher causality and counterfactual-world lineage guards. No local tests
are claimed as native voice, intelligence or complete-task acceptance.
