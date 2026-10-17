# 17 October: native control recovery

Plain words: the accepted Context Spanning synthetic voice remains the candidate.
There is still no qualified complete native model. The136-update balanced model
7500125 is the retained reference because independent speech recognition confirms
its three requested private values. More updates alone made conversation worse.
The new stage adds narrowly approved controls on actual histories and starts
from those136-update weights. It is fine-tuning a pretrained model, not foundation
pretraining or a return to the old Qwen experiment.

## Evidence and decisions

| Run | Observed result | Decision |
| --- | --- | --- |
| Matched14 balanced136,7500125 | Native7500126–28 speak orange square, silver hexagon and teal star; independent speech7500290 confirms. Unrelated speech and missed controls persist. | Retain as reference; unqualified. |
| TRAIN ordinary7500312/13/14 | Delegates all three ordinary training questions. | Failure is not solely generalisation beyond training questions. |
| Unchanged648,7500366 | Reloads exactly and lowers text loss, but native7500380–91 produces wrong explanations, long repetitions and missing private answers. | Reject; no further unchanged expansion. |
| Cancellation mixed15,520,7500371 | TRAIN control labels reach loss0.007/top-choice accuracy1. TRAIN0207500396 still misses cancellation. Validation0077500397 triggers at19.68s after cancellation ends19.42s; backend actually executes cancel_lookup. Spoken response falsely denies authority. | Request behavior improves on one validation case; full task still fails. |
| Exact released-default persona,7500553–60 and corrected zero pairs7500586/88/90/92/94/96/97/98 | Completed comparisons do not repair private-result use or basic explanations. | Upstream persona-distribution warning is useful, but this ablation does not justify adopting that prompt. |
| Native-audio first question7500610 | Exact27-frame prefix and independent temporal audit7500656 pass. Speech7501118 hears two different words (word error0.2222). | Exclude this rendering from training. |
| Teacher-acoustic first question7501230 | Same independently eligible first question; complete51-frame capture, exact prefix, verified original teacher acoustics. Later project, requests and results excluded. | Include only after its own component/temporal audit. Never call forced text/audio native success. |
| Current-rollout cancellation7501220 | Independent CPU approval binds actual TRAIN020 history7500396 to collector7500371 and its exact checkpoint/state/protocol. Three controls only; no words/audio/user/fact labels. | Include only after independent mathematical audit. |

These are reused development scenes and training-scene diagnostics. They are not
a fresh sealed qualification set. Native speech model request timing is measured;
tool selection still belongs to the external reasoning backend.

## Corrective stage

Plain words: source3223510 adds an explicit weights-only start for a new data
stage. Its fresh optimizer and step count are recorded separately from the
parent136 updates. Every continuation must carry the same warm-start identity,
new data/audit identity, optimizer and random-generator state. No old optimizer
state is silently reused across changed data.

The17 components are three ordinary answers,11 post-result private sentences,
one first clarification question and two cancellation request windows. Each has
its own provenance validator. Sampling cycles equally across verified behavior
kinds and then examples within each kind. Past/span/audio/user losses remain
zero; real assistant PAD3 targets keep half weight. Frozen speech/base weights
are hashed. Training touches the existing four final temporal adapter layers
and text head, with unchanged learning rate and retention losses.

Independent17-component audit7501359 passes with maximum error0.0001449585
and zero causal-prefix error. Eight-update warm proof7501360 passes finite
gradients, unchanged frozen weights and exact prediction/optimizer/random-state
reload. The warm source is7500125; its136 update count is not added to the new
optimizer count. Pending originals7501242/43 were held and replaced with matching
new dependencies/preflight IDs; zero-update audit used the short debug reservation.
Native7501763/65/66/67/68 remain on workq; interactive pencil7501844 replaces
held/cancelled7501764 and produces a refusal in sampled text. No longer update
budget is submitted yet.

Plain words: source9c90970 adds a separate return-to-listening target from four
measured quiet seconds after the teacher finishes its ordinary answer. Actual
repetitive TRAIN0087500380 supplies conditioning, never factual labels. CPU1751
passes exact collector/voice/state bindings, same user PCM, quiet teacher/user
stems and50 control-only labels. This is not permission to treat the rejected
648 model's words as truth. Independent18-component temporal audit7501865 gates
any use of this additional component.

## Delayed-result and ASR diagnostics

Plain words: source26ffd8b supplies an explicit inference-only minimum transport
delay for actual lookup results. It holds only executed lookup receipts, leaves
ordinary/cancellation responses undelayed, keeps worker threads free, and rejects
cancelled late results. It does not force speech or request tokens.

Twenty-five-second comparisons7500652/53/54 still expose wrong independent
physics explanations. Delayed cancellation7501186 used the interactive debug
reservation but failed on invalid ASR word bounds. No result from invalid
recognition is authorised. Source3b787dc turns only that typed recognition
rejection into an explicit repeat-request notice with no backend API/tool call,
so the audio stream continues. It never clamps timestamps or uses reference
words. Repaired debug7501264 completes but still misses cancellation; zero ASR rejection
notices occur in that run, so live recovery of the invalid-timestamp path is not
demonstrated by it. Its listening/physics/private-result behavior remains unqualified.

The reservation starts short debug probes quickly. Normal training remains on
workq. Question7501143 was held and replaced by7501230 after verified effective
reservation settings; the original was cancelled before duplicate execution.
A second simultaneous interactive submission hit the site's submit limit and
was not accepted;7501264 was submitted after the question became terminal.

## Delivery and preservation

Plain words: keep19 October delivery,20 October recovery buffer and18 October
18:00Berlin training cutoff. The17 October12:00Berlin recipe freeze was missed
without a qualified recipe. Cloud validation remains on hold and no cloud
resources were created.

The accepted public weights, earlier states and new corrective dependencies
are backed locally in runs/cloud-export-20261016. Five earlier checkpoint/state
pairs and all15 earlier component tensor hashes match their reports. Newer
checkpoint/data transfers and current source bundles must be verified before
final packaging. The old16 October global SHA index is stale. CPU portability
checks do not establish a destination-GPU resume.


## Independent checks completed after the initial record

Plain words: speech1857 confirms the cancellation mismatch from actual PCM,
with word error0.0521 for7500397 and0.0638 for delayed7501264. Both have valid
recognition time/word bounds. The backend cancellation in7500397 is real; the
spoken denial is also real. These are separate requirements.

Plain words: all18 components pass temporal audit7501865 (maximum error
0.0001449585, causal-prefix error0). Workq proof7501930 uses the same verified
136-update weights and a fresh optimizer, with a12-minute walltime. No128- or
512-update continuation is submitted yet. Five17-component native checks are
still pending in the normal queue at the last read; the isolated pencil check
7501844 refuses despite a correct returned explanation.


## Bounded18-component continuation

Plain words: proof7501930 passes eight new updates, unchanged frozen weights,
and exact prediction/optimizer/random-state reload. Continuation7502015 adds128
updates with the same approved data, optimizer state and warm-source identity.
Its stage counter will be136; the original warm weights also had136 updates.
This is one new corrective comparison, not an unchanged-recipe scale-up.

Dependent native cases7502018–27 cover all eight validation scenes and TRAIN019/020.
Their declared worlds come from approved source metadata: validation000 is yellow
circle and001 green oval. Earlier136 reference tests used declared counterfactual
worlds silver hexagon/teal star; do not treat their value-dependent outcomes as
matched comparisons. Extra changed-world7502028/29 use copper diamond/amber
triangle. No value is supplied to the speech prompt before an actual lookup.
All are development evidence; none is a fresh sealed qualification.

## Completed eighteen-component results and response-boundary experiment

Plain words:7502015 completes128 additional updates, for136 updates in this
fresh optimizer stage, and reloads predictions and optimizer/random state
exactly. Its native development checks7502018–29 all fail complete tasks.
Do not expand this recipe unchanged. Validation000/002/003 include the correct
private value in sampled text but also refusals or wrong physics. Validation001
omits part of green oval. Salt requests a backend and describes chloride atoms
as positively charged; pencil requests a backend and initially refuses.
Clarification asks no project question; cancellation produces no successful
cancellation. Changed worlds copper diamond/amber triangle also fail. These
are sampled-text observations; independent speech audit7502649 checks four
recordings separately. Checkpoint SHA is
b159ad9e18d20697a1304e44c6f047c117b55556fb178737435980091d654be3;
optimizer state SHA is
f116fb4159c6a16fe40013d402935d1d17b19e5cf7b4f261fe361fa811445676.
Both are already exported and hash-verified locally.

Plain words:7502129/30 isolate startup warm-up without changing weights or
conversation. Sixteen silent frames run before resetting all streams and the
conversation seed. Generated packet-stream audio, user audio, persona prefix,
voice file and sampled text-token sequence match exactly. Packet audio SHA is
70894d0fe591d4c4c62481b16d61e56d4e3994c1ab276d4932496061fa9d4c71.
Maximum input backlog falls from3.1745792s to0.3938067s;95th-percentile compute
is0.04279s versus0.04341s. This one paired result supports runtime warm-up, not
better answers. Neither capture is approved for training.

Plain words:source05d7b28 explicitly composes the complete approved ordinary
TRAIN answer at the model's first observed EPAD/RET response boundary. The
previous composition kept later teacher timing, after the native model usually
requested a backend. This timing hypothesis is unproven. All preceding actual
conditioning and user audio remain fixed; the complete teacher answer shifts
with its word spacing retained, after the full user question. No backend result
is hidden, no held-out example is used, and forced text is not native success.
Six new pure timing guards and thirteen existing ordinary/clarification guards
pass. CPU7502645 approves all four plans and four legacy private regressions;
eligibility SHA91775c9474ae01f9b3492371da8c54d1d6d8bafd4d63b7baa8d24188ff91ce58.
Native own-audio capture, independent transcription and mathematical checks are
still required before any optimizer sees these components.

## Rejected first boundary plan; verified prediction/readout correction

Plain words: the first response-boundary experiment is REJECTED despite passing
its mathematical audit7502672 (18 components, maximum error0.0001139641) and
exact eight-update reload7502676. The engine returns text one frame after
predicting it. The plan used the returned EPAD/RET rows to select the prediction
step, but then repeated EPAD even though it was already in the exact prefix
cache. This is a bug in the new boundary composition, not proof of the cause of
all earlier failures. No continuation of7502676 is submitted or permitted for
adoption. Captures7502654/55/56 remain historical evidence only. Triangle7502653
also rendered Ice instead of Three and was excluded before that proof.

Plain words: saved native states establish the clock exactly in three original
TRAIN traces7486102/04/05. Their RET readout frames50/40/35 were predicted at
49/39/34, after EPAD was already sampled at48/38/33. At the RET prediction step,
EPAD probability is3.70e-9/8.79e-10/4.51e-8; RET probability is
0.99843/0.99987/0.99909. Exact cache identities, rather than sampler probabilities,
prove the one-frame delay. This rules out sampling chance or batched numerical
error as an explanation of that alignment discrepancy.

Plain words: source787410a uses the teacher's first word at that actual RET
prediction step, reuses the preceding EPAD from the verified actual cache, and
records prediction and readout clocks separately. Every returned future text
token must equal the previous forced prediction; capture and independent
validation both enforce this identity. The full teacher reply remains retained,
and user audio/past remain unchanged. The regression failed before the fix and
passes afterwards. Selectorbd70fc8 also rejects old policy versions or changed
planner hashes, preventing an old CPU approval from silently selecting a new
composition. All25 timing, selector and ordinary/clarification guard tests pass.
CPU7502998 independently replans the four corrected candidates and reruns the
four existing private regressions. These checks do not establish native quality.

## Corrected nineteen-component result and decision-weight comparison

Plain words: corrected CPU7502998, captures7503072/73/74/71 and per-frame
readout checks pass. Speech7503079 hears Three correctly. Day/night own-audio
7503074 is excluded because ASR reads240 rather than24 hours, even though its
aggregate word error0.0755 is below0.15. Teacher-acoustic replacement7503101
uses the verified original teacher. Lightning7503072 error0.1429, evaporation
7503073 error0 and triangle7503071 error0 are retained as explicitly forced-text
training conditioning. Nineteen-component mathematical audit7503111 passes;
8-update7503114 and128-continuation7503303 restore exactly. Stage136 checkpoint
SHA is d99f2fa33863f3ef7b9de3d1a198b8af78f9eb076de32b1eae239a490b095322;
state SHA is a6735c9e9ae089f6e5c80f1cc1146dc0ee9dab9d168316339ec6c9b48a428fea.
Both are exported locally. Native validation7503369–78 fails full tasks.
Initial TRAIN invocations7503362/63/64/66/67/68 failed the required recorder
startup guard before model execution; they are not model failures. Corrected
TRAIN3464–69 also fail complete tasks. The original source world for
clarification006 is brown square; these checks now use it explicitly.

Plain words:7503469 has a real TRAIN cancellation receipt, delivered21.9038s.
Speech7503524 confirms the later false denial, word error0.06 with valid bounds.
The model really requested cancellation and the external backend executed it;
tool choice is still external. Source323b086/da79343 adds explicit read-only
trained-collector replay, verifying checkpoint/state/assets/voice/protocol before
native graphs. Replay7503670 reproduces the actual TRAIN020 model/cache/random
state exactly, with no new backend calls or supervision. This is a foundation
for correctly scoped acknowledgment data, not a new quality-qualified model.

Plain words:sourceebf4d27 gives only the independently verified first ordinary
word16 times the token-loss weight; other kinds and target ownership remain
unchanged. Eleven real Torch gradient/resume/calibration tests pass. Proof
7503477 and136-stage7503528 restore exactly. The same nineteen components and
parent7500125 weights are used, with a fresh optimizer for this new objective.
Native7503551/52/53 answer three ordinary TRAIN questions without a backend;
evaporation and day/night are plausible, lightning remains wrong. Triangle
7503554 still requests a backend. Private validation0007503557 and its changed
world7503565 both invent7.22 without a lookup. Other private/control checks also
fail. REJECT the complete model despite the routing change. Checkpoint SHA is
22851e6b40b2dc0a4bc36a914f8ce107f114e278dec45c68bca24bca41b28e8c.
The outcome exposes missing positive first-lookup supervision: the eleven late
private components supervise words after results, while retention alone does
not preserve that decision under stronger ordinary-answer training.

Plain words:constant live RET-logit biases0/-2/-4 were compared on unchanged
7500125 weights in7503275–80. All four calibration-specific Torch tests pass;
nonzero bias covers exactly all377/500 live predictions and changes no other
logit or hidden state. Bias-2 still delegates the ordinary question. Bias-4
removes its request but also removes the private lookup and hallucinates180.
REJECT both calibrations; no wider sweep is justified.

Plain words:the old complete private captures7467981/7468102/7468611/7479546
share the accepted voice and contain a verified first lookup policy. They also
contain a later backend answer span without a matching second teacher request.
Do not treat those whole futures as native routing evidence or reintroduce the
answer-bearing tail to solve this failure. A separately revalidated prefix,
ending at the first approved lookup and before every future receipt or spoken
word, can provide positive control supervision without that later dependency.
The native-audio7479546 duplicates teacher-acoustic7468102; do not double-count
the same question/world merely because the conditioning differs.


Plain words:lookup-prefix CPU7504055/56/57 all pass:17/13/16 conditioning
frames, exactly one RET label each, no answer words, future receipts, audio or
user labels. They derive from7467981/7479546/7468611 respectively; do not
include the duplicate7468102. Twenty-two-component mathematical audit7504247
passes all components (maximum absolute streamed/batched difference0.000151635,
causal error0). Data SHA15b81402c6a4fb4e9fab90c81a527a011be9ec8d02f9aad0c181f481f7b72446.
Fresh-optimizer proof7504403 starts from7500125 weights;136-stage7504434 passes
exact optimizer/random-state and prediction reload. Ordinary first-word weight16
and balanced-kind sampling remain explicit. Checkpoint SHA
16055329c24f19621f3cc958c753e981f8b0dc873b45c2c53fd4b60e4ec532ca.
Native4494–4509 restore requests for help but regress ordinary direct answers;
private-result use, clarification and cancellation still fail. Do not continue
this recipe unchanged. These are development scenes, not sealed-test results.

Plain words:source2b8d6f5 tests a bounded actual-ASR question plus validated
backend result. It never supplies reference words or forces spoken tokens.
Paired7504409/13 private,4411/15 clarification,4412/16 TRAIN cancellation and
4425/26 ordinary salt remain unqualified. Startup4410/14 reject the unused
one-word CLI value before model execution;4425/26 are the corrected replacements.
Query anchoring sometimes supplies a value but the model reads context labels
aloud, contradicts facts and misses cancellation. It is not adopted. Cold-run
between-step pauses reach multiple seconds and confound latency comparisons;
result-prefill timings themselves are about60–72ms. Do not attribute every pause
to the speech network or call these cold samples a runtime qualification.

Plain words:independent speech7504102 confirms weighted19 native evaporation3552
and day/night3553 at word error0, and confirms lightning3551's wrong explanation
at word error0. Private3557 audibly invents7.22; word error0.477 also exposes
mismatch between text and spoken repetitions. Thirty-eight new native recordings
are backed up locally and match their report audio SHA256s. The listening page
has five native examples, each HTTP200 verified, with explicit failure labels.
The Git bundle now includes complete history through2b8d6f5; cloud remains paused.

Plain words:the next separately typed data path uses only the approved teacher's
terminal “It's cancelled” sentence after an actual, successfully delivered cancel
receipt. It requires the exact verified trained collector, matching TRAIN user
PCM/world, a quiet boundary after all receipts, and unchanged word spacing.
Earlier model denials stay conditioning without labels. This teaches a late
correction; it does not qualify the first acknowledgment or cancellation timing.
Other trained future generation remains blocked. Capture, an independent
component audit and a mathematical audit must pass before optimization.


Plain words:cancellation CPU7504574 passes. Capture7504669 restarts307 real
frames exactly on actual TRAIN7503469 with verified collector7503303 and no new
backend calls, then composes11 future frames of the approved terminal “It's
cancelled” sentence. Independent CPU7504688 passes all scope, provenance, native
input, target and collector checks. Six text targets include five word tokens;
past, user, audio and private-fact labels stay zero. The selected boundary is
24.56s, after the already observed denial: this teaches a late correction, not
an earlier truthful first acknowledgment. Mathematical validation and native
behavior tests are still required; neither replay nor teacher forcing qualifies
a model.

Plain words:actual7504434 optimizer steps reveal a retention pairing confound.
With six supervision kinds and four retention references, step-modulo-four
pairs each kind with only two references. Ordinary updates see only simple and
cancel; lookup updates see only private and clarify. A regression reproduces
this exact arithmetic (red), and the explicit per-kind rotation passes13 guard
checks (green). Default legacy behavior is unchanged, including old checkpoint
resume; the new schedule is separately selected and bound into recipe metadata.
This is a falsifiable training hypothesis, not an established cause of the
native regressions. Compare the same22 data/root/weight/update budget before
claiming a gain. Separately evaluate late cancellation-acknowledgment supervision
from the actual3303 collector.

Plain words:runtime sourcef897a7f adds measured immediate log I/O and an explicit
inference-only deferred mode. The original cold traces have11.05/11.17/11.31s
between-step pauses despite sub0.05s typical model steps; the cause was not
measured then. Warmed repeated diagnostics7504674/75/76/77/79/80/81/82 compare
immediate/deferred writes on unchanged3303 weights and actual-ASR input. Await
actual write timings before blaming shared storage. No quality qualification
or runtime improvement is inferred from this instrumentation alone.


Plain words:per-kind retention proof7504743 restores exactly. Learning-curve
jobs7504805 and7504806 resume that same saved state for128/512 additional updates
(136/520 total), with the same22 data, root7500125, weight16 and new schedule.
The earlier128-update fit took36.24s and exposed each ordinary example only5–6
times. This bounded comparison checks underfitting rather than assuming more
compute cures bad supervision. Numerical fit and native task behavior remain
separate gates. Do not promote either checkpoint from loss alone.

Plain words:the20-component cancellation stage uses the19 corrected components
plus4669, without the three new initial-lookup prefixes. Audit7504744 and
warm-start proof7504816 are submitted, with collector3303 as the parent weights,
a fresh optimizer, ordinary first-word weight1 and explicit per-kind retention.
This tests receipt-conditioned late acknowledgment separately from the22-data
routing experiment. Continue only after mathematical and exact reload gates.

Plain words:warmed immediate/deferred diagnostics4674–82 all complete. Immediate
writes peak0.0003–0.0005s, not the earlier11s. Input backlog is0.277–0.364s across
all eight. Deferred mode has four zero-underrun runs except one76-sample gap;
immediate has one zero-underrun run, others176/452/581 samples. The old stall did
not reproduce, so do not claim storage caused it or that these few samples
qualify a latency fix. No language/task improvement is established.

Plain words:the portable launcher now carries mixed-component mode, warm-start
identity, actual and retention sampling, first-word weight and saved-state
resume. A local integration check verifies exact Docker arguments and data-mode
rejection using a mocked Docker executable; no instance or GPU was created.
The local Docker daemon is currently stopped, so no new amd64 data-restore result
is claimed. Isambard training remains the priority and cloud validation is held.


Plain words:7504805/06 both restore exactly at136/520 updates. At520 the first
ordinary target probabilities become0.239/0.893/0.379/0.020, while initial lookup
probabilities stay0.999. Native4899–4917 confirms direct evaporation/day-night
answers at520 but wrong lightning, missing or contradicted private results,
clarification and cancellation. These ordinary successes are sampled-text
observations pending independent speech checks, not a qualified full model.
The retention change at136 does not resolve the failures. More fitting improves
some decisions but does not make teacher-conditioned accuracy native success.

Plain words:20-data136/520 checkpoints7504893/96 restore exactly. The final
late-acknowledgment fit reaches top-one on all six supervised targets at520,
but this is not evidence of actual cancellation or truthful free speech. Their
native screen5114–5129 is running; available results already show private lookup
collapse, incorrect physics and repetitive speech. Do not extend this recipe
unchanged. The source-state correction must also preserve positive lookup
supervision and independently check actual cancellation receipts.

Plain words:source81b9269 adds explicitly scoped trained-state corrections,
without weakening the original zero-collector/v2 paths. TRAIN-only captures
4965–4969 observe the520 model on ordinary and clarification audio with backend
execution disabled. Native-ASR TRAIN private captures4970–4972 use real declared
orange-square/purple-triangle/orange-triangle worlds. No evaluation input becomes
training. CPU5110 verifies all eight plans, exact trained checkpoints and user
PCM, passes four legacy component regressions, and runs eleven real Torch/state
and boundary tests. Thirty-eight local guard tests also pass. The new first
response replaces a sampled word or RET at its prediction frame, reuses the
already cached EPAD, retains measured word spacing and requires a complete
arrived question. Clarification ends before the later project utterance. Private
correction still requires a real delivered result, quiet post-result state and
unchanged source/world. No generated model claim supplies a gold label.

Plain words:trained future captures5207/5214/5216/5217/5218 (ordinary/question)
and5219/5220/5221 (private) all complete successfully. Exact restart lengths are
51/40/34/49/30/201/176/146 real frames. All five early captures verify actual
one-frame text readout; own sampled speech remains forced-text training data.
The three private captures force approved teacher acoustics for conditioning
only. Speech auditing is submitted for the five own-speech captures, followed
by independent component/mathematical approval before any new optimization.
A potential26-data recipe replaces the old four ordinary/question captures,
adds three trained private corrections and the cancel acknowledgment, and
retains all eleven late-private lessons plus explicit positive lookup controls.
It is not approved or submitted as training yet.


Plain words:speech7505283 passes all five trained-prefix early captures at the
unchanged0.15 word-error bar, with valid word and segment times. WERs are
0.0357/0/0.0566/0/0.1111. Critical content is preserved: light faster than sound,
liquid becomes gas, Earth rotation and24 hours, three sides, and a missing-project
question for the locker label. The lightning rendering substitutes “survive”
for “travel” once; the question adds a preposition. These remain forced-text
captures, not native task passes. Proposed selection is saved with speech and
plan hashes in `context-trained26-selection-20261017.json` on Isambard.

Plain words:26-component audit7505317 and dependent eight-update proof7505318
are submitted from immutable81b9269, whose fourteen changed file MD5s match local
source. No optimizer starts unless the full matching mathematical audit passes.
New-stage weights start from4806, optimizer restarts, first-response word weight16
now explicitly covers the verified trained clarification boundary as well as
ordinary boundaries, and retention rotates per kind. Three trained private
corrections and acknowledgment4669 supplement the retained eleven private
lessons, three positive lookup prefixes and original cancellation/listening
controls. Qualification is pending. Cloud remains paused.

Plain words:26-component audit5317 and proof5318 pass.136/520 checkpoints5421/22
restore exactly and preserve frozen weights. Their32 free-running development
runs5474–5505 remain unqualified: evaporation and some rotation answers work;
5422 asks the right first question on TRAIN019 and validation006 but never
requests the lookup after the project reply. Some changed private values are
used then contradicted with stale pending claims. Cancellation is missed.
No sealed test was read; independent speech5897 confirms correct question
wording at WER0, and also confirms the incorrect lightning explanation at WER0.
ASR checks support words spoken, not factual correctness.

Plain words:eight capability comparisons5524–5538 complete. Explicit cancellation
capability on4806 executes cancel_lookup in5528 and eventually says it cancelled,
but the whole conversation still fails.5422 misses cancellation even with that
instruction. This is a component clue, not an adopted prompt or qualified model.
Three latest checkpoint/state pairs and40 native recordings are now exported
and hash-verified locally in context-trained26-and-persona-local-verified.json.

Plain words:owner reviews the report and authorizes faster continuation. The
intelligence question remains unresolved: wrong everyday answers cannot be
excused as only task control. Zero-adapter paired native runs5883–5888 retain
the same voice/layout/attention but all six request help and several mishandle
correct supplied answers. The latest adaptation is not the sole cause. Eight
publisher-persona comparisons5899–5906 now test the documented upstream persona
instead of inventing a generic instruction that upstream warns is unsupported.
These are inference-only reused development scenes, never new training labels.
Six earlier voice-argument startup failures are preserved, explicitly excluded
from behavior evidence and corrected to accepted-synthetic before model runs.

Plain words:source8a1c201 introduces explicitly typed trained-history control
roll-in. Only three independent expert PAD/EPAD/RET controls receive loss; a
fourth unforced frame proves the delayed RET readout and gets no loss. Existing
past words/user/audio remain conditioning. CPU plans5869/5870, captures5893/5895
and independent audits5894/5896 pass with four legacy regressions. Actual replay
is133/228 frames respectively; readout[3,3,0,4] is verified. No tool is executed
by a forced capture. Audit5910 and dependent eight-update5911 use these two plus
the original26, same4806 parent, balanced kinds, per-kind retention and weight16.
Only after the complete mathematical/reload gates may the bounded matched
continuation run. The intended outcome is complete-task improvement without
ordinary/private regression, not merely fitted control-token probability.

Plain words:28-data mathematical audit5910 passes every component, maximum
streamed/batched absolute error0.0001516342163, causal prefix error0. DataSHA is
de92099057ded15a1af0e387c96b457cdf4c1731c26594dcdca211d4c442fc1e.
Eight-update5911 is running; dependent5957 adds512 updates, same parent and loss
settings. Sixteen native runs are in context-trained28-native-full-20261017.jsonl;
independent speech5984 waits for them. All stages require predecessor success
and revalidate artifacts on load. No new native behavior is qualified yet.
Adding clarification_lookup makes eight supervision kinds versus seven for26;
same balanced-kind policy changes exposure, so this is a bounded new-mixture
comparison rather than a pure one-label causal estimate.

Plain words:publisher-persona5899–5906 fails to restore reliable behavior. Base
5899 corrupts a correct backend salt explanation;5900 gives a plausible supplied
pencil explanation. Adapted5903/5904 answer directly but incorrectly;5906 claims
cancellation despite only a lookup receipt. The issue is neither proved to be
only control nor proved to be only base knowledge. Exact source histories5483/85,
control tensors5893/95 and14 new native comparison WAVs have local readback
verification in context-transition-local-closure-verified.json. New native
question/answer/failure players are available through viz.listen at
/audio/voice-diagnostics/response-boundary.html; four WAV URLs return HTTP200.

Plain words:5957 passes520-update training and fresh-process exact reload.
Full16-native matrix completes. The new control targets have an actual effect
in some scenes:5968 requests cancellation at19.12s input clock after full17.29s
cancel audio, and the successful cancel receipt is supplied23.117s.5975 performs
a real lookup after a correct clarification question. TRAIN0195967 still makes
no request. The complete model remains unqualified: private/status words
are contradictory, ordinary lightning/pencil answers are wrong, cancellation
acknowledgment fails. These development scenes are reused, not fresh holdouts.

Plain words:speech5984 completes19 recordings.5968,5976 and publisher5906 have
invalid word timing and substantial text/audio mismatch; do not pass them.
The incorrect lightning and pencil explanations are audible at WER0. Correct
sampled copper diamond5977 is rendered Copper Dive by ASR, so value-use speech
is not verified. Model5957 and source histories have portable backups; all28
component/source files (119 total) pass readback SHA256. Static scan finds no
missing referenced path; no new destination-GPU restore is claimed.

Plain words:next comparison changes the retention objective instead of simply
extending5957. Existing four references only teach KL agreement with the poor
base and hidden-state preservation. New explicit weight1 adds CE against their
independently approved measured teacher assistant text. Prefix, supplied spans,
user and audio get no new targets. CPU6263 passes all four source/mask/label
checks and real Torch gradient, excluded-logit and resume tests. Same data,
parent, sequence inputs, existing KL/MSE and update budget are retained; source
b0f1beb binds the new weight/scope audit in exact-resume metadata. Mathematical
5910 is reused because actual-component inputs and forward math are unchanged.
Short6461 must pass finite gradients/frozen weights/fresh reload before512-more
6462 and16 native6463–6478; speech6479 follows. Hypotheses remain retention
conflict, insufficient actual result-state coverage, or a backbone limit. No
new-objective gain is inferred from its implementation or CPU checks.

Plain words:teacher-retention proof6461 PASSES eight updates and exact optimizer/
RNG/prediction reload. Actual logged teacher CEs are nonzero for all four kinds
(approximately0.60 private,0.13 simple,0.39 clarify,0.45 cancel initially).
6462 is confirmed running. No native improvement is inferred. The previous
5957 matrix has p95 compute0.0365–0.0448s/frame but cold backlog3.14–3.84s and
12/16 runs with some causal-playout underruns. This is not latency or microphone
acceptance. Source5957 logs are preserved from its authoritative home stdout.
Its checkpoint,16 native WAVs and two raw histories pass22 further local hashes.

Plain words:teacher-text-retention6462 PASSES520 updates and exact fresh reload,
but its16 completed native development conversations do not qualify. Ordinary
lightning adds a false explanation after the correct speed comparison; triangle
fails despite the correct supplied backend answer. TRAIN clarification retrieves
brown triangle but then mixes it repeatedly with evaporation. Validation
clarification6475 drops the required question/lookup. Private replies remain
contradictory or absent; independent speech6479 is running. Reject this recipe.

Plain words:the next bounded comparison changes future acoustic conditioning
only for three already verified trained-history private corrections5219/20/21.
It keeps4806's exact raw prefix4970/71/72, approved teacher sentence, measured
word spacing, actual prior receipt, user codes, text labels and sparse loss mask.
Engine.step samples its own future audio while text is forced for TRAIN capture;
this is not native task success. New independent CPU validation compares labels,
mask, first input and all user inputs against each original teacher-audio
component. Past/span/user/audio loss remains zero. Any capture must pass exact
restart, CPU validation, independent ASR including private words and mathematical
stream/batch audit before optimization. No added teacher-retention objective.

Plain words:own-audio captures6926/6930/6932 PASS exact native replay, returned
text's one-frame delay and native acoustic ownership. CPU6934 independently
PASSES all three against original5219/20/21: identical labels/masks, first input,
all user codes and actual receipt. Text target counts12/23/21 are unchanged.
Seven legacy regressions pass. Independent speech6936 has word error0/0/0.1,
valid segment/word bounds and all three private phrases correct: orange square,
purple triangle, orange triangle. Nine local artifact hashes pass and all three
WAVs return HTTP200 through viz.listen. These are forced-text TRAIN captures.
Numerical audit6969 is running; gated proof6976,512-update continuation6978,
16 unforced native runs6979–6994 and independent speech6995 are submitted. No
native improvement is established yet. Keep teacher-retention weight0.


Plain words:math6969 PASSES28 components, maximum streamed/batched error
0.00015163421630859375 and zero causal-prefix error. Eight-update proof6976 is
confirmed running. Inference-only, zero-adapter publisher-persona comparison
7079–7082 tests original versus accepted-synthetic voice on reused salt/pencil
questions; speech7083 follows. This tests voice-reference confounding, not a new
model, new holdout or voice-quality promotion. No result claimed yet.


Plain words:voice-reference7079/7080 terminate at the exact-training-voice guard,
before conversation; these are engineering failures, not native task failures.
7081/82 with accepted reference complete, but do not form a matched voice pair.
Dependent speech7083 is cancelled. Corrected7115–7118 use the existing adapter-
free path for BOTH voices, validated layout7463825 and MATH attention, with the
publisher persona; speech7119 follows. No source/guard changes. Own-audio6976's
fit report now PASSES; its separate exact reload is pending. Main continuation
6978, native6979–6994, speech6995 remain gated. Do not duplicate submissions.

Plain words:continuation receipts are context-native-private-trial-20261017.jsonl,
context-native-private-native-20261017.jsonl and
context-native-voice-conditioning-bare-20261017.jsonl under cluster runs.
Read-only local status script /private/tmp/duplex-native-private-model-progress-bounded.py
retrieves full reports with bounded Slurm calls. All three new capture tensors,
restarts/audio, CPU audits and math6969 are exported locally;34 audit-referenced
paths have no missing export paths. Sourceb42a6e4 restores as a clean independent
clone from the Git bundle. This is source/data preservation, not cloud GPU resume.


Plain words:proof6976 PASSES exact fresh-process reload and four local artifact
hashes. Continuation6978's520-update fit report PASSES; separate reload is pending
at the last snapshot. Voice ablation7115–7118 and speech7119 complete. Both voices
produce a reasonable graphite explanation after actual backend text, while salt
explanations are poorer. ASR WER is0.1176/0.0294/0.0588/0.0294, all segment/word
bounds valid. Do not infer broad intelligence, voice neutrality or a preferred
backbone from two reused questions. No voice switch is adopted; four WAV hashes
pass locally. Original7079/80 guard failures remain excluded.

Plain words:zero-adapter FP32 conditioning diagnostics on the three matched late
components are mixed. First-fact probabilities change0.7189→0.9530,
0.6377→0.5649 and0.8778→0.6627 under own audio. This does not support a universal
teacher-acoustics shortcut explanation. Native behavior remains decisive.
Read-only cache/conversion metadata inventory finds25 GPT-Live TRAIN teachers:
19private,4simple,1clarify,1cancel (15.23minutes including silence). The rejected
teacher-text trial used four longest histories, one per kind. Broader coverage is
a distinct unresolved hypothesis, not evidence that larger compute would help.


Plain words:6978 PASSES exact fresh-process reload and four local model/state/
data/voice hashes. The first completed native runs reject full qualification:
6981 says night comes from the Sun moving away,6983 misses lookup after its
project question,6984 executes real cancellation but mixes answers and repeats,
6985 gives yellow circle followed by false eclipse physics. No native gain is
established by the acoustic-conditioning change; remaining cases and speech6995
are still pending. All-teacher audit now has a distinct schema rejected by the
unchanged four-reference optimizer policy. It audits all independent TRAIN
labels/masks and code/source hashes without a model forward or optimizer.


Plain words:all16 native6979–6994 complete. Validation clarification6991 now
asks the project, actually retrieves brown square and includes it in sampled
text; it omits the subsequent requested evaporation answer. This is a component
gain, not full-task qualification. Changed6993 includes copper diamond but wrong
physics;6994 has no text/lookup. Cancellation6984 really cancels but repeats and
mixes responses;6992 misses cancellation. Ordinary6981 says the Sun moves away
from Earth. Sixteen WAV hashes pass locally, raw TRAIN6983/84 and full stdout
are preserved. ASR6995 is still running at the last snapshot.

Plain words:these16 runs have compute p95s0.0362–0.0436s per80ms frame, cold
input backlog3.1256–3.3296s and11 recordings with causal-playout underrun samples.
No microphone acceptance or broad low-latency qualification. Teacher CPU7449
PASSES all25 histories:11,426 real text targets,993 spoken targets, zero prefix/
span/user/audio targets, with19private/4simple/1clarify/1cancel. Corpus source
and all17-row/native real-to-virtual maps are independently checked. The new
schema cannot authorize the unchanged four-reference optimizer; an explicit
local rejection guard passes. A broader objective needs its own recipe, sampler,
short proof and native evaluation; no broader optimizer has been submitted.


Plain words:ASR6995 completes all16 recordings in5minutes.6983 has invalid
segment/word timing.6986/6994 give only You.6981's wrong Sun-moves-away speech
has word error0.6985's ASR says yellow screen, so correct model text does not
verify the private spoken value.6991's ASR contains brown square with valid
timing, but full-clip word error0.278 and omitted follow-up prevent acceptance.
Changed6993's copper diamond is heard (WER0.1014), followed by wrong/repetitive
physics. Whole trial remains REJECTED; no model promotion, fresh holdout or
microphone qualification. Full per-case review joins actual receipts, sampled
text, independent ASR and runtime limits in
runs/recovery-continuation-20261016/context-native-private-native-review.json.

Plain words:before broader teacher optimization, verify its complete prepared
teacher inputs against the native Engine at prefix/span boundaries. CPU7449
proves target provenance and loss masks; it does not prove live/training input
parity or broader gradients. Keep four-reference KL/hidden retention separate
from a proposed25-teacher supervised term, record per-kind teacher exposure and
exact resume identity, then require a new short proof/native comparison. Existing
four-teacher optimizer rejects the all-teacher inventory schema;19 related local
guards pass. No broader optimizer is launched and no compute-gain assumption.

Plain words:the next audit executes complete approved teacher recordings through
Engine.set_persona, Engine.inject_context_span and Engine.step, including both
native codecs. Only independently approved teacher text/agent acoustics are
forced. It compares all consumed17-row inputs against the current prepared
teacher sequence, separately compares all eight streamed user codec rows, and
checks every real target's ownership and real/virtual mapping. No optimization
or new target generation. Native runtime CuDNN-on/MATH settings are explicit;
eager execution audits integer inputs, not production latency. Initial cases
cover all four task kinds plus the longest private reference, before the full25.

Plain words:normal audit jobs7800–7804 are pending on Resources. Official
interactive policy was rechecked17 October: one short debugging audit is allowed,
production batch work stays on workq.7802 was held before a held replacement was
submitted; the new job's effective partition/reservation were verified, then7802
was cancelled and7852 released.7852 is confirmed RUNNING; remaining four stay in
workq. No duplicate work ran. Source4650eea contains the audit; wrapper source
deb2662 has byte-identical audit code. All changed source MD5s match.

Plain words:codec7456232 had already established small offline/streaming user
code differences. A separate CPU verifier is prepared to distinguish those from
any further token-layout error. It reassembles the oracle with recorded actual
streamed user codes only, requires identical teacher words/agent codes/masks/
clocks, and checks all current forced targets and every prefix/span/live call.
The original strict offline-cache failure remains a recorded result; the native
conditioning variant is a separate data declaration. No optimizer approval or
claim that small codec changes explain behavior is made.

Plain words:all five complete GPU traces finish. Legacy offline-cache comparison
fails only in user rows; text/agent rows are exact. Independent CPU8370 PASSES
all17 consumed rows after substituting the actual streamed user codes, with
unchanged teacher targets/masks, current forced-target ownership and every span/
frame checked. No further layout mismatch found in these five cases. Remaining
20 teacher traces8408–8427 run on workq, at most8 GPUs; CPU8428 independently
validates the complete25 afterward. This does not qualify native conversations.

Plain words:the prepared numerical preflight requires ALL approved teachers,
rejects a partial or duplicate inventory, streams their complete native call
ledgers and compares the batched forward in FP32 MATH (threshold0.001). Changing
the final input must leave every earlier prediction unchanged. It binds actual
native input tensors, original labels/masks and source hashes; it performs zero
optimizer steps. No broader supervised optimizer is implemented or submitted yet.
