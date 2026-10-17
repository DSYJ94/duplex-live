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
