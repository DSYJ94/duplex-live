# HANDOFF — where we are (updated 2026-10-17 for active recovery)

## Current execution — 17 October, mixed ordinary/private correction

Plain words: at11:43UTC/13:43Berlin, renewed SSH access is needed. The certificate
expired11:47Berlin; SSH now rejects public-key authentication. `clifton auth`
is running locally and a sign-in tab was opened; the owner has been asked to
complete it. Do not duplicate any jobs on an observation failure. Retrieve
7495630 (BayLing assets),7495650 (runtime retry),7495652 (independent late audit)
and7495653 (dependent remaining late captures) after renewal. Their current
terminal states are UNVERIFIED because access expired. Latest source is
be534a3 before this documentation update. The planned12:00Berlin recipe freeze
has passed without a qualified recipe; do not claim it was met. Keep19 October
delivery/20 October recovery buffer and18 October18:00 training cutoff visible.

Plain words: current07:48UTC status supersedes the mixed-stage submission note
below. Mixed16 checkpoint7495189 reaches136 updates and exact reload, but native
7495483/84/85/88/89 plus retries7495506/07 and changed-value7495508/09 do NOT
qualify it. Ordinary questions are refused, available values are omitted, and
clarification/cancellation fail. CPU speech7495544 confirms the wording in eight
recordings (mean WER0.04068). Cancellation7495489 has invalid ASR timestamps;
do not interpret its repeated ASR words as verified repeated model speech.

Plain words: source4242668 fixes exact recorded-payload replay and records
pending-state virtual boundaries. Matching pending/canonical TRAIN array7495543
has19 captures. CPU audit7495583 validates all19 raw histories and four old
private components, but approves ZERO full early continuations: pending state
would precede the teacher request, or no quiet early cut exists. Twelve separate
post-result corrections are feasible. Never move request/receipt clocks to force
an early pass. Eligibility SHA256 is
`a20ee13355ab00d9e214caef149fd03aad32564de6efa30aeeb946c85a846ebc`.
Map: runs/context-pending19-capture-map-20261017.json on Isambard (local copy in
the continuation directory). Source2bd64cd enables explicit hash-bound late
selection. Pilot7495632 restarts all206 prefix frames exactly and captures a
32-frame orange-square teacher-text/acoustic component. It is not native success.
Independent temporal/ownership audit7495652 gates remaining11 captures7495653.

Plain words: bounded BayLing baseline audition is being prepared while these
checks run. Public source is pinned41669ac04bbcbfaf5b1be3a2cc563e4b3752d041;
asset pins are evals/assets/bayling-audition-pins.json. CPU download7495630 is
pending retrieval. CPU runtime7495638 failed because TorchAudio was absent;
7495650 retries with isolated TorchAudio2.9.0 and unchanged NGC Torch2.9.
Current launcher source04dd62e has not yet produced a BayLing inference result.
Its probe is explicitly OFFLINE (whole audio encoded first, wave decoded after
generation), checks exact loading, and cannot qualify streaming or delegation.
Do not change the backbone based on source availability. Preserve upstream
GLM-4-Voice license/notice and review commercial registration before adoption.

Plain words: active goal is Isambard model improvement and an honest qualification
result by19 October; recipe freeze17 October12:00Berlin, stop new training
18 October18:00. Cloud GPU validation remains on hold. No qualified complete
native model exists. The owner-approved Context Spanning synthetic voice remains
the candidate; ordinary answering, clarification and cancellation are still gaps.

Plain words: source2e1c352 adds separately gated ordinary-answer futures. Raw
observations7486102/04/05/06 pass integrity7486128; eligibility7489479 approves
four exact-user complete teacher replies without backend context. Future array
7490241 yields7490242/43/44/41. Independent CPU speech7492289 recovers the first
three explanations (WER0.0357/0/0.0377), but arithmetic7490241 is silent and is
EXCLUDED. These are forced-text rendering diagnostics, not native task success.

Plain words: mixed16 stage orders the three ordinary futures before the13
existing own-acoustic private futures7480443/44/45/46/47/49/54/55/63/64/65/66,
7480417. Zero-update temporal audit7493024 passes all16 (maximum error0.00027275,
causal-prefix error0). Fresh8-update proof7494393 passes frozen-weight and exact
prediction/optimizer/RNG reload checks. Bounded128-update continuation7495189
also reloads exactly but fails the native checks above. Use legacy-epad0 plus
native-prelead-epad for approved source
conversion; future real-frame PAD3 half-weight supervision is independently
audited.7492288 rejected an incompatible launch convention before updates.

Plain words: previous136-update checkpoints7480819/20 are now backed up locally
under runs/cloud-export-20261016/tree/runs with adapter and training-state SHA
matches against report.json. They are preserved baselines, not qualified models.
Full new logs and audit reports are in runs/recovery-continuation-20261016.

### Earlier execution notes — current statuses above supersede these


Plain words: corrective ordinary-answer preparation is now sourcec35f8fb.
Observe-only mode records natural simple TRAIN histories without backend
execution or answer-bearing spans; it cannot qualify a task or approve labels.
Captures7486102/04/05/06 correspond to TRAIN008/009/010/011, with exact zero-proof
voice/layout7463825 and expected-kind=simple. First validate all four with
`evals.context_history_validate` at collector commit
`c35f8fbbcfa402ad43683d474ab66a1ac39ce292`, then add separate ordinary-answer
future eligibility. Private receipt checks must remain unchanged. Metadata
feasibility of the old four raw histories found answer-bearing spans during
three teacher replies; report is `simple-control-feasibility.json` in the local
continuation directory. New captures are raw conditioning only, not training.


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


Plain words: late16 October — owner explicitly prioritises Isambard model gains,
not extra compute. Cloud validation is ON HOLD; $25 remains approved for later
validation only, with zero cloud resources launched. Clifton was renewed and
SSH works. Both136-update13-case continuations7480819/20 complete and reload
exactly but do not pass native controls; do not expand on text loss alone.

Plain words: a concrete backend interface bug hid actual successful receipts
when worker prose answered only a later question (7480841: lookup=`teal star`,
injected prose=physics only). Source3e42038 now injects verified actual record
facts alongside prose; stale/fabricated/cancelled receipts cannot be promoted,
and errors remain failure notices. All21 focused regression tests pass.
The incorrect numeral in7480830 completes at9.957s before real receipt injection
at10.869s: waiting and available-result use are distinct failure mechanisms.

Plain words: four targeted native tests7485619–22 keep the136-update own-audio
weights fixed. Receipt preservation alone still fails. Adding actual pending
status (no forced speech) yields correct sampled orange square/teal star text
in7485621/22 without the earlier fabricated values. This is preliminary text
behavior, not full voice/task qualification. Independent CPU speech audit7485697,
ordinary/clarify/cancel controls7485698/99/5700 and zero-adapter comparisons
7485701/02 are running/queued. Exact payload events and phrase/receipt clocks
are saved under `runs/recovery-continuation-20261016`.
See `docs/research/2026-10-15-resource-decisions.md` for what was actually adopted
from each supplied resource and what remains unimplemented.


Plain words: latest checkpoint at19:45Berlin supersedes transfer/pending claims
below. Both full public model weights are exported locally and read-back SHA
verified (33.49GB total), with `runs/cloud-export-20261016/model-verification.json`.
All25 corrected synthetic histories are backed up in the workspace/export tree;
four critical file hashes per case pass. They remain conditioning-only, not
optimizer-approved. Required zero/layout proof7463825 is included too.

Plain words: the provider-neutral amd64 image builds and runs CPU state/data
checks. Actual Isambard136-update checkpoint7475992 deserializes on amd64:
26 finite adapter tensors, finite optimizer moments, CPU/Python/NumPy random
states accepted. CUDA-state restore/forward/backward on a destination GPU is
STILL UNTESTED. No cloud resources provisioned; provider/account and spending
cap remain undecided. Use `portable/context/README.md` to restore and validate.

Plain words: matched native tests7479192–97 all fail task qualification, including
private TRAIN/validation, simple, clarification and cancellation. They use the
correct fixed voice/layout, so this is a legitimate rejection for scaling the
three-conversation136-update recipe. New loader789d3f3 restores the checkpoint's
required native layout and MATH attention automatically, exactly once;7480107
runs without explicit layout/math flags and still gives the wrong private value.
All51 saved prefix tensors match the explicit-protocol run7479192 exactly;
this verifies that prefix only, not the complete generated trajectory.

Plain words: corrected25-case collection7479198 completed. Eligibility report
`context-matched25-eligibility-9c5dc2047dc4019a6f39a5efc10894fdfe2586fa`
approves13 cases for gated fact-future capture, not for optimization. It rejects
unsupported ordinary/cancellation targets and cases without a valid silent
prefix/measured sentence/actual matching receipt. All13 teacher-audio future
captures7479955 completed. Independent temporal audit7480420 PASSES all13;
maximum streamed/batched error0.0008345 (limit0.001), causal prefix error0.
Corrected eight-update/reload proof7480562 PASSES, frozen base/speech unchanged.
Real job IDs:7480269,7480270,7480273,7480275,7480280,7480281,7480282,
7480283,7480289,7480290,7480292,7480295,7479955. Do not use the old7477252 map.

Plain words: native-audio futures7479546/47 pass independent ownership/delay
and TRAIN-only audits. Assistant audio is naturally sampled; approved teacher
TEXT is forced, and real source receipts are replayed. CPU speech audit7480258
recovers the intended Purple Triangle/orange triangle values, with word errors.
This is a rendering diagnostic, not native reasoning or live delegation success.
All13 native-audio counterparts7480417 completed with restart gates passing,
sourcee150524. Independent temporal audit7480482 PASSES all13, maximum error
0.0002728 and causal prefix error0. Corrected eight-update/reload proof7480563
PASSES exact predictions/optimizer/RNG reload, frozen base/speech unchanged.
CPU speech check7480484 recovers11/13 complete private phrases; mean word error
10.16%, median8.11%. Two Blue Circle renditions are incomplete/misrendered despite
supplied text. This is not native policy success or a human voice-quality gate.
Native real job IDs:7480443,7480444,7480445,7480446,7480447,7480449,7480454,
7480455,7480463,7480464,7480465,7480466,7480417. Interactive first-source7479666
remains pending and is not evidence of faster scheduling. Both new checkpoint
states/data/adapter hashes are verified in the local export. Proof launch7480433
was rejected before optimization because CONTEXT_ACTUAL_TEMPORAL_FIT=1 was
omitted;7480483 was cancelled pending to fix the same launch flag. Corrected
jobs7480562/63 are the valid proofs. No training source/data rule was weakened.

Plain words: paired128-update continuations7480819 (teacher acoustics) and
7480820 (own acoustics) resume their exact new eight-update parents, sourcee150524.
Native tests depend on successful completion/reload: teacher7480830/32/33/34/35/36;
own7480837/39/41/42/43/44. Both test TRAIN000, changed private validation000/001
(silver hexagon/teal star), pencil005, clarification006 and cancellation007.
These are known audio scenarios with new private worlds, not unseen audio.
There is still NO qualified native checkpoint. The paired baseline text losses
are0.716 teacher-audio versus0.921 own-audio (FP32, same13cases); this is a
conditioning diagnostic, not a proven explanation of native task failures. New data/source is a new stage, never a silent exact resume.

### Earlier 16 October notes — current statuses above supersede these

Plain words: owner confirms compute ends in5 days and requires readiness1–2
days before. Delivery target is19 October,20 October recovery/export buffer;
freeze recipe by17 October12:00Berlin and stop new training by18 October18:00.
See [dated delivery plan](docs/DELIVERY_PLAN_2026-10-19.md). Time-box small
experiments; this deadline does not waive causal-data, voice or task gates.

Plain words: IMPORTANT correction at18:15Berlin — recent native evaluations
omitted the prefix/span layout used in actual-history training. Their wrong
outputs are real, but they are NOT a matched checkpoint-quality comparison.
Source8a14d91 now refuses actual-history adapter runs without a layout audit.
Corrected native tests7479192–97 use adapter7475992, exact checkpoint voice,
and layout audit7463825. Check these before any scaling/rejection verdict.

Plain words: the first25-case array7477252 completed but used the released
unwrapped/no-layout protocol and freshly cloned voice, unlike the replay
contract. Preserve it as conditioning evidence; do not use it in the current
future trainer. Corrected array7479198 uses zero proof7463825, its exact voice
codes and layout audit7463825, sourcee591faa. A regression audit also exposed
the different newly cloned voice; regression checks now use each artifact's
own recorded voice. No optimizer approval was issued for the mismatched batch.

Plain words: capture planning now matches all25 approved GPT-Live teachers to
exact TRAIN user PCM/words and teacher worlds (19private,4simple,1clarify,1cancel).
Plan SHAa79f4309e6e4f8dcc666b3c6f65231c5ed849252e5bed84794f5053b8ddb8b9b.
Early fact selection no longer depends on a late quiet gap; its own silent
prefix and actual receipt/240ms speech-margin checks remain mandatory.
Grounded clarification can use the same checks. Explicit teacher IDs preserve
which approved trajectory is selected when multiple teachers share a world.
CPU eligibility source839bd23 predates protocol/voice-regression fixes; do not
rerun it as the final broad-corpus audit. Use the current source and corrected map.

Plain words: large direct model transfers BOTH lost their partial files on SSH
timeout. Custom data/checkpoints/bundles remain safe. New stdlib
`evals.artifact_shards` splits fixed-hash models into512MB pieces and atomically
restores only after all piece/full hashes pass. Two corruption/missing-piece
regression tests pass. Remote shard creation is running; small model assets
transfer separately. No full model-weight export is verified yet.

Plain words: corrected recipe7475992 (136 total updates) PASSES exact reload
but FAILS native private-result use, including TRAIN000. Native7476104/05/06
are wrong/missing complete values;7476107 gives the correct pencil/graphite
text;7476108 fails clarification;7476109 stops on an invalid backend record
argument. Do not scale this three-conversation recipe or call it qualified.
Source90ef8eb fixes that runtime crash by turning rejected tool arguments into
explicit failed results, never success receipts;16 regression tests pass.

Plain words: cloud export now passes all3 independent data approvals inside an
isolated amd64 Docker container, without Isambard filesystem access. Python
open tracing missed libsndfile's native reads; added286 scene-audio files
(612MB), hash/size checked, alongside845 observed files (1.24GB). Both manifests
and restored audit results live in `runs/cloud-export-20261016`. The corrected
8/136-update checkpoints, required temporal preflight, accepted voice reference,
all3 future components and raw student histories are included. Actual amd64 GPU
model resume and model-weight transfer verification remain pending.

Plain words: further raw TRAIN data must cover controls, not only private facts.
Initial captures7476271/76/79/81 completed but TRAIN004–007 are PRIVATE scenes
(fold numbering differs from validation), so they are not control evidence.
Corrected collector3005c58 adds an explicit expected-kind guard. Verified TRAIN
control captures7476737/38/39/40 are simple008/simple009/clarify016/cancel020.
They are conditioning evidence only, with no automatic optimizer approval.

Plain words: the current stage corrects a confirmed loss mismatch. Upstream
`contextspan/training/finetune.py` lines38–51 trains real PAD3 silence at half
weight; our sparse early mask excluded it. Sourcead82745 independently verifies
the original capture unchanged, then approves real text/PAD targets only;
past/span/user/audio losses stay zero. Audit7475863 PASSES all3 with257/257/240
targets, including169/181/172 silence targets, and causal prefix errors0.
Fresh base8-update7475934 PASSES separate exact prediction/optimizer/RNG reload
and is backed locally with state/adapter/data hashes checked. Do not resume old
0e0e363 states into this changed loss/source stage. The old168-update7475556
passes its engineering/reload checks but is not a behavioral success claim.

Plain words: the amd64 portable image now BUILDS with pinned NVIDIA digest
42263b2424fc237b34c4fc4a91c30d603c57eed36e37d31ff6d9a4f1f801edee.
Native Engine/training imports and both CPU optimizer/RNG resume tests PASS
inside that image. Actual amd64 GPU continuation is still unverified. All845
observed files (1.24GB) passed size/SHA checks and are saved in
`runs/cloud-export-20261016/tree`; observed-read manifest sits alongside it.
This does not cover unobserved native/subprocess IO or establish a full restore.
Exact Context/base model-weight transfers are still running into
`/private/tmp/duplex-cloud-export-20261016`. Clean source and vendor bundle clone
restore checks pass. Rebuild immutable checkouts with Git metadata before use.

Plain words: the owner requires continued training and development after expiry,
across AWS/GCP/Modal/Nebius/Runpod; destination undecided. No paid resources
provisioned. `portable/context` contains the pinned amd64 CUDA recipe; local
Docker build is in progress, not verified. Atomic optimizer/adapter/RNG state
support is in immutable f3804c6; interrupted/uninterrupted CPU tests pass.
New full-model state reload and destination GPU validation are still required.
Exact resume binds the same source/data/recipe; do not change those silently.

Plain words: corrected independent CPU audits pass all3 early captures with
88/76/68 text targets. Old audit7469288 stopped before any optimizer because
unquoted whole-value token matching disagreed with measured per-word slots;
0e0e363 uses the already verified measured sentence/word positions instead.
Corrected temporal7469530 PASSES all3 streamed/batched FP32 checks (maximum text
errors0.00007629/0.00006056/0.00008321) and causal prefix errors0.
Eight-update7474819 and resumed32-update7475099 (40 total) PASS exact separate
reload, optimizer/RNG restoration and frozen-weight checks. Both are backed
locally with state/adapter/data hashes verified. Their training commit is
0e0e3634ea258761d2d9f16df6e3347ce9150834; exact resume must retain that commit.
Native7475190 still fails the full orange-square answer after8 updates. The
40-update development batch7475396/98/5400/02/04/06/08/10 is queued/running.
Invalid voice launch7475100 was cancelled and replaced with accepted-synthetic;
no result is taken from it. The corrected7475190 started in workq, so no
interactive duplicate was submitted.

Plain words: provider-neutral `portable/context/train.sh` runs the same
audit/fit/resume commands without Slurm. Docker's first base-image download
failed in its local layer store; retry is in progress, so amd64 build/GPU
validation is still pending. Offline read inventory completed, identifying845
observed files (1.24GB), and their transfer is in progress. This is not yet a
complete dependency-closure/restore proof. A verified complete vendor Git bundle
(27.6MB) and repository-history bundle are backed locally. Reconstruct standalone
source checkouts, not linked worktree .git files pointing to the old host.

Plain words: early captures7467981/7468102/7468611 completed. First independent
CPU audit passes88 text targets, including request, with no past/span/audio loss.
Three-component temporal audit7469288 uses the short interactive reservation;
pending workq duplicate7469148 was cancelled first. Early manifests remain
quarantined; separate independent audit controls eligibility. No new native
behavior success is claimed. All8 head-only validation tests7467595–7467602
completed; their outputs reject head-only proof7466440 as a useful candidate.

Plain words: temporal7467496 PASSES exact fresh reload and is backed locally
with adapter/data/voice/packaged-audit hashes checked. Native7467590 gets actual
results (backend failures0, RET2) but omits orange square. Independent speech
audit7467658 WER0.0222 confirms intelligible faulty speech. Reject a result-use
success claim. Native7467526 failed the explicit TRAIN-history launch guard
before inference; corrected7467590 captures TRAIN history explicitly. Pending
head tests7467049/51/53/55/57/59/62/64 were cancelled because their scene settings
had the same mismatch; validation replacements7467595–7467602 are in workq.

Plain words: late corrections start at26.16/17.92/20.72s after actual receipts
at10.55/10.87/11.03s. They do not teach the earlier choice to wait for a result.
CPU early-window plans pass on all3 histories: original future dialogue clocks,
incoming real user speech, no preceding student reply/request, actual future
receipt boundaries, complete approved teacher text/request targets. Source review
adds measured first-word onset and240ms private-speech-after-receipt margins,
and checks pending assistant gold residuals across spans. New native Engine
capture7467981 (immutable553794d) uses the explicit native-prelead-epad labels.
It executes both codecs and native span-exit readout; future receipts are replayed,
not freshly dispatched. Captures remain quarantined with training_approved=false
and future_text_targets_approved=false until an independent artifact audit.

Plain words: head-only proof7466440 PASSES8 updates and exact separate-process
FP32/native-BF16 feature reload. SHA0c70dde5983d272a7c3772f23808a7c50fff6f2487642d0b573a807584ca6563.
Learning is tiny (native NLL1.819252→1.819245); first fact-token top1 was already
3/3 under previous teacher speech. Do not call this improved result use.
Native development tests7467049/51/53/55/57/59/62/64 are queued on workq:
two changed worlds each for000/001 plus ordinary004/005, clarify006/cancel007.
Earlier submissions7466976/78/80/82/84/85/86/87 used an invalid voice CLI name;
they were cancelled and replaced before model evaluation. No accepted evidence
comes from those submissions.

Plain words: temporal audit7467209 PASSES all3 components: streamed/batched
FP32 text max errors0.00004745/0.00004911/0.00005651, causal prefix errors0.
This checks actual already-delayed inputs without another shift. FP32 versus
captured BF16 hidden differences0.346/0.284/0.238 remain a separate numerical
diagnostic; exact native replay is still the original stricter gate. Interactive
reservation started this short check; pending workq copy7467202 was cancelled
only after acceptance. Audit/data/voice are backed locally.

Plain words: interactive short backward proof7467496 (immutable0859f51) starts
at10:35UTC, node nid010715. Eight updates train last4 temporal rank8 adapters
and text head on the3 approved actual-history continuations, using native BF16
batch/MATH computation. All actual past/audio targets are absent. Separate
approved TRAIN private/simple/clarify/cancel examples regularize the unchanged
base text distribution (KL) and hidden states (10x MSE). Base speech parameters
are frozen and hashed; require finite gradients and exact fresh-process reload.
The matching zero-update temporal audit is mandatory and packaged with the
proof. This engineering proof alone does not justify128-step training. Retrieve
FULL logs before deciding whether to run its unforced native comparisons.

Plain words: corrective future components7466157/60/63 PASS strict native
Engine replay through327/224/259 real frames of actual TRAIN histories7465804/06/07.
They contain32/56/35 future frames,12/23/21 text targets, zero past or audio loss
targets; each uses a complete independently approved same-user/same-world teacher
sentence. Source001 already spoke its value and is excluded. These components
and their passing restart reports are backed locally with hashes checked.
Job7466440 (source07824cd) now independently revalidates each component, source
approval, builder commit, voice and exact cut before an eight-update text-head
proof plus fresh-process reload. All temporal, speech and other adapter weights
stay frozen. Later features contain earlier teacher audio: fitting them is not
proof of result integration. Require fresh unforced native tests; no new model
qualifies yet. Eleven related local target/request-policy tests pass.

Plain words: the owner renewed Clifton and authorised retry. Normal SSH works
at07:46UTC/09:46Berlin on16 October; current certificate expires21:46:11Berlin.
The previous pre-auth access block below is historical, not a current blocker.
Prepared batch ran successfully:128 pilot7463729, TRAIN calibration7463730 and
native dependent tests7463731/33/34/35/36/37/38/39. Pilot passes exact fresh
reload; checkpoint SHA7a6894b3604209188e0446b7df68686e2d0d99ed7d2f5a79d77b5a6cd9c1e51f.
Reject7463729 for scaling: native7463731/33/34/35/36/37/38/39 have seven empty
text streams/zero RET, with only pencil005 sampling an unqualified answer. Saved preflight7457079 is now fully
backed locally, with adapter/data/voice hashes checked.

Calibration7463730 on the PAD3/pre-lead corpus has ordinary-word NLL11.339→11.206
and private-phrase NLL10.426→10.322 after8 updates; RET top1 remains0/40.
These figures must not be compared as an unchanged-label learning curve against
the old legacy-padding diagnostic. Direct accepted native captures7454283/85
contain29/46 EPAD0 tokens and both show EPAD0→RET4→first word. A new explicit
`native-prelead-epad` conversion retains ordinary EPAD cues and moves only a
lead's cue into an empty frame before its derived RET. Every source and derived
cue frame is recorded; no words/audio/receipt clocks move. Require all48 causal
checks and an integrated8-step proof before any larger run. Published convenience
preparation's PAD3-only format is not proof the checkpoint prefers that format.

Raw-history integrity checks now pass both7457161/62: actual tensor file hashes,
voice/source/model pins, source TRAIN approval and all frame/receipt accounting.
They remain unapproved conditioning, with future target approval still absent.

Plain words: EPAD-retaining preflight7463825 passes8 updates, integrated native
history/parity and exact fresh reload. Its native7463977–84 all produce text.
Independent audits7464215–22 confirm both changed values in Willow000 and
ordinary answers, but Alder001, clarification006 and cancellation007 fail.
This is partial behavior only. Bounded128 pilot7464214 completes/reloads;
native7464223–30 has four empty sampled-text streams and remaining faulty
private/control behavior. Reject for scale. Dense teacher-history training
still drifts away from useful free-running behavior. Keep the accepted base.

Plain words: new adapters7463825/7463729 and actual TRAIN v2 histories7463985/86
are backed locally with hashes checked. Strict restarts7464207 and cold7464272
fail assistant-acoustic code equality while clocks/provided flags/global RNG
and delayed returned text match. No future gold targets are approved. New v3
collectors7464403/04 save temporal hidden/text scores to locate divergence;
dependent7464405 compares actual consumed inputs before any optimization.
Do not weaken the gate or pretend this restores codec/PCM streams.

Plain words: recorded-voice controls7464929/31 reproduce the saved prefix
exactly; re-encoding voice in the replay loader had changed process-global
compiled-kernel history. Later acoustic mismatch remains. Source review
rejects an unproven extra-RNG/reset-offset explanation. Full Engine replay
now executes both codecs from lossless approved FLOAT user PCM at the actual
recorded span boundaries, without ASR/API/new dispatch. First7465002 used
incorrect codec CuDNN flags and failed; corrected7465018 restores captured
flags. Retrieve its full result before approving any future targets.

Plain words: future-target helpers reject wrong worlds, missing/failed/cancelled
actual results, changed user audio/words, premature teacher words and token
collisions. Ten related local tests pass. Successful native tool results use
route=delegate with executed receipts; plain reasoning uses route=answer.
These selectors alone approve no dataset or optimization. New proof/history
artifacts are backed locally; full source/evidence remains immutable by job ID.

Plain words: exact full Engine MATH replay7465797 PASSES all189 frames:
actual input IDs, temporal hidden/text values, sampled cache/provided values,
clocks and CPU/CUDA RNG. No tolerance or sample-equality requirement relaxed.
Recorded voice reuse plus both codec operations and matching math attention
are needed together. This qualifies the bounded history/data seam only.

Plain words: four further MATH TRAIN histories7465804–07 collect approved
worlds000 orange square/001 orange diamond/002 purple triangle/003 orange
triangle. Future-data code2c4cfd7 first plans same-user/same-world complete
teacher sentences at a quiet failed-reply boundary, then repeats strict replay
at that cut before native forced-gold future generation. Past labels stay absent;
initial residuals stay sampled. Plans/targets/optimization remain pending.
CPU-only eligibility can run with `python -m evals.context_future_component`;
it loads approved code caches and hashes but performs no model forward.

### Earlier recovery evidence — 15 October

Plain words: no complete model qualifies. The owner says Context Spanning with
the synthetic reference is “as good or better” than smooth Lychee. Keep both as
accepted development voices, not factual/delegation passes. Continue authorised
work until actual input is needed; allocation expires around 21 October.

Context Spanning is now the leading native duplex candidate. Its pinned released
checkpoint strictly loads all 475 LM keys and 16 audio heads. The first three
scheduled-dispatch/measured-word component probes 7454283–85 speak the actual
private values; they do not test native retrieval. Six subsequent native-RET →
actual arrived-audio Whisper → actual backend runs 7454442–44 and 7454577–79
recognize the right project and obtain the correct actual private value, but
only 7454578 samples that value correctly. Other foreground replies omit or
invent facts. Warmup is not the sole cause: the cold controls also fail.

Independent speech audits 7454749–58 cover ten Context captures. Nine have WER
0–0.0667; 7454579 has WER 1.7568 with repetitive audible speech absent from
its sampled text. Accepted voice quality does not waive this speech failure.
First-frame latency and some warm-run playout underruns also remain unresolved.
These are reused development scenes, not fresh qualification holdouts.

We have completed bounded native Context Spanning adapter proofs and rejected
two128-step recipes; the latest request-policy8-step model is also unqualified.
This is adaptation of released weights, not foundation pretraining. Revalidate the 48 approved TRAIN scenes across caches
7411513, 7434926 and 7435538; retain ordinary, private, clarify and cancel cases.
Use complete clean 17-row sequences, measured word times, approved result
arrival times, native context-span assembly and clean user-target ownership.
Require zero-adapter logits/RNG parity, finite nonzero gradients, frozen-base
hashes and exact fresh-process reload before any larger training. See
[training seams](docs/research/2026-10-15-context-training-seams.md) and
[resource decisions](docs/research/2026-10-15-resource-decisions.md).
Proof 7455033 stops before updates on marker-list order; 7455068 passes
48-scene provenance but fails BF16 zero-adapter identity. Corrected proof
7455157 completes eight finite updates on four complete TRAIN conversations,
1,894,400 adapter parameters, frozen base unchanged, 34.06GB peak allocated and
exact fresh-process FP32 logits. Checkpoint SHA
`d2086be61657383141510a234ae5f3b51325ffb315d7d7b048ded2845c53455b`.
Native depth q15 errors .0755–.192 become <=9.54e-6 with batch-only context15.

Actual history audit 7455458 captures native inputs through a complete voice/
persona prefix, 70 real frames and an approved receipt span. Released inputs
omit the first prefix column and lose some pending acoustic values at the span.
The explicit candidate bootstraps the prefix and preserves only delay1 codes
whose provided flags are true; all 17x316 actual input IDs then match the clean
training oracle exactly. This is teacher-forced input-layout parity, not a
production graph/logit/quality pass. Real free-running pending assistant codes
must never be copied as if observed.

Matched native-ASR tests 7455516–23 compare released, layout-only, zero-wrapper
and eight-update models for two private values using identical saved voice codes.
All have zero causal-playout underruns, but none passes the complete task.
The trained purple run says its actual value; trained yellow invents money.
One zero-purple backend executes the actual read but returns empty text, so
that is a delivery-contract failure, not evidence the foreground ignored a
supplied value. A new worker guard rejects empty backend text before delivery.
Independent audits 7455646–53 are complete; details are in EXPERIMENTS.md.

Pilot 7455712 completes 128 updates across all 48 TRAIN scenes, with exact
fresh reload and unchanged frozen weights (SHA
`12635510464a68aeceea7ab2826da10b33750377306b6ca68bf794941d3918b2`).
**Reject it for scaling:** 7455763–76 show silence/no RET on six of its eight
private/control cases, one isolated improved pencil answer and failed cancellation.
Thirteen speech audits 7455873–85 retain those failures, including silence-floor
failures/Whisper's unreliable “You” on quiet output. Backend zero-control
7455773 also aborts on an attempted read without user identification; the guard
blocks it. New error handling preserves the failure and supplies an unspoken
notice while continuing PCM. All such recovered failures remain failed cases.

Teacher calibration7455921: ordinary-word TRAIN NLL improves1.887→1.299,
private-phrase NLL .439→.430, but RET top1 is0/40 even after128 updates.
RET probability at ordinary-word slots falls .0413→.00178, suppressing the
natural native requests. Do not describe improved teacher fit as live competence.
The diagnostic's old key `eot` actually means **EPAD/end of padding**, not end
of utterance. Native Context preparation uses PAD3 without those EPAD0 tokens;
old Moshi Interleaver adds them. There is no inherent80ms word shift: the direct
check finds the same floor timing, with only collision-induced delays.

Codec audit7456232: native batch-two encoding exactly matches both cached
speakers for251 frames. Native streaming user IDs differ in3/251 semantic frames
and5–17/251 other-head frames, predominantly later/silent regions. First any-head
difference is9.44s, after this TRAIN scene's first request. No broad role/clock
shift is found; zero-shift agreement is best. Do not claim full streaming ID
identity or silently shift/re-encode targets. Audit7456078 failed before measurement
because its standalone entry omitted the known Triton driver helper;6232 fixes it.

Padding-only preflight7456094 replaces EPAD0 with PAD3, retaining every measured
word, audio code, RET position and receipt clock. It passes integrated native
history, all-gradient/frozen-base checks, eight updates and exact fresh reload.
SHA `63f7d31c0e31eed5de5d1173607263142ea3bb893a84d49468e822211cc3f8c0`;
new data SHA75c8a6dee846420b3319984fc40b041240d0ca109bbda616f967bef3568e5796.
Padding-only128 ablation7456282 passes128 finite updates across all48 scenes,
unchanged frozen hashes and exact fresh-process FP32 reload. SHA
`3e9d4011a23858e505878c43a7b2ffbf8af4cda89480ad4776ecd660d006c13c`.
Reject7456282 for scaling too: native7456656–63 all emit empty text and zero
RET across private000/001 paired worlds, simple004/005, clarify006 and cancel007.
Canonical temporal audit7456824 passes complete-prefix/first-span inputs, hidden
states (max error.0003052) and text logits (.00009883, zero top1 disagreements).
This removes a measured early temporal-computation mismatch as an explanation;
production graphs, free-running exposure and all-span parity remain unproven.
Next native-prelead conversion explicitly separates the request target from the
immutable actual teacher dispatch clock. It places RET in an unoccupied frame
before the next measured lead, checks arrived user identification/cancellation,
rejects a request after receipt, and keeps words/audio/receipt times unchanged.
No held-out example enters this derivation. Require all48 checks before updates.
Preflight7457079 validates all48 derived TRAIN sequences and completes eight
finite updates, exact fresh FP32 reload and integrated temporal/input checks.
All40 RET targets move later by.16–.80s (median.44s); source clocks stay fixed.
SHA0b20cc01db2174fb114201f81e41cdf0ddeb5373d3f72edfc3efca898ba59716.
Native7457163–70 all sample nonempty text and emit1–2 RET, but four private cases omit their
values;001 requests only the later physics answer. Clarify006 makes an early
unidentified-project read (correctly blocked) and invents M room. Cancellation
is still not followed. Simple005 gives a correct graphite explanation;004 has
an irrelevant E F T preface. All input is retained; only cancel007 has10 samples
of simulated-playout underrun. These are development failures, not qualification.

A TRAIN-only native history collector is prepared to record actual direct-prefix/span and graph
inputs, source/voice/code hashes and real-to-virtual receipt boundaries. These
are raw conditioning only, never approved target labels; every past label must
be masked before any independently checked future continuation can train.
First collectors7457161/62 complete private TRAIN000/simple008: exact contiguous
call accounting1x17x780/626,501/355 actual graph calls plus direct prefix and2/1
receipt spans. Zero playout underruns. Neither recording is approved supervision.
The access problem is now explicit: a fresh verbose SSH trace receives
**“Not allowed at this time”** from ai.login.isambard.ac.uk before key
authentication. Stop automated connection recovery. The verified certificate
is still valid until16 October10:39:50 Berlin; do not misdiagnose expiry.
The owner has been asked to test the ordinary SSH command manually. Official
[agent access guidance](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/)
says, “Do not use automation to recover from access or policy issues.”
No persistent control connection was established, and SSH configuration is unchanged.
Prepared native-policy128/calibration submissions returned no job IDs and were
refused during connection establishment; do not call them running.

After authorised access is restored, the concrete reviewed next batch is
[evals/context-native-policy-next.sh](evals/context-native-policy-next.sh):
verify pinned9b411d0 source, preflight7457079 SHA/data/voice/reload/history gates,
submit one128 pilot, one TRAIN calibration and eight afterok-dependent native
checks. It creates submission receipts and refuses a duplicate invocation.
On the cluster, review the file after `git pull --ff-only`, then run
`bash evals/context-native-policy-next.sh`. The helper itself has only local
syntax checks; it has not run on the cluster. No further owner voice vote is
needed until a behaviorally credible new candidate is available.

One explicit pending-state diagnostic7455886/87 improves acknowledgement of
waiting but still omits both results and hallucinates physics. Reject more
prompt-only tuning. Clarification-recovery7455888/89 remains behaviorally bad.
Next, if padding alone fails, separate the native request-label policy from old
actual teacher dispatch times and collect approved actual-student histories;
do not invent request/word timestamps or train on development failures.

The current Clifton public certificate was freshly checked: valid until
16 October10:39:50 Berlin (08:39:50UTC), superseding the earlier expiry note.
An SSH attempt reported “No route to host” at the login bastion; that is a
network failure, not evidence the certificate expired. Retry bounded requests;
do not ask for reauthentication while the verified certificate remains valid.

The older Lychee work below remains useful evidence and a fallback:

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
[the decision note](docs/research/2026-10-15-resource-decisions.md).
CPU loss-scope audit 7454206 passes all 15 examples; there is no user-audio loss
in the current pilot. Concurrent worker speech audits confirm the actual values,
but yellow WER 0.1111 and earlier playback gaps still fail the gates. Context
Spanning assets and actual native-ASR probes are complete as summarized above.
Its released backend-text mechanism is distinct from our older prefix refresh.

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

38 strict TTS scenes plus 10 independently approved GPT-Live TRAIN teachers make
33 train / 8 validation / 7 test. Exact source/word/grid/native-token clock checks
pass; no held-out audio trains. Original base codes are reused. Single-context
encoding 7411513 works on `Exclusive_Process` GPU; 128-update teacher training
7411514 saves/reloads with text/audio errors 0.0 and unchanged frozen weights.
Checkpoint SHA 104bcd641351e03f8bf7da6f00477bf63c9d8a4f369a75c5d1b9bd7247753093.

Corrected context gives one private read and correct value in 7411889, but adds
unsupported claims. 7411886 simple delegates unnecessarily, 7411887 skips needed
clarification/invents books, 7411888 misses cancellation. These are failures.
Audit 7412064 private tail is invalid: do not use aggregate WER. Backend API
component 7411885 passes 8/8 with 3 actual reads; this is not native intelligence.
Self-cancellation bug reproduced/fixed; native missed dispatch remains unsolved.
Causal dialogue context excludes future words/GT assistant answers; still an
oracle, not live ASR. Single-context voice and job-local HTTP ports work. Current
36 relevant local causal/context/atomic-cancel/alignment/playout tests pass. Captured reference
proof 7410675 repeat/prefix errors 0.0, post-injection change 5.6953.

Clifton renewal is verified by live SSH. Temperature-.2 probes 7412588/7412590
both fail; the latter eventually dispatches real cancellation, but too late.
Speech audit 7414528 and full recordings/logs are retrieved. Do not rerun duplicates.

Owner now accepts both **PersonaPlex with the synthetic marin reference (7415052)**
and **Lychee-FD (7415325)** as sounding like GPT-Live. Original PersonaPlex NATF2
is rejected as muffled/radio-like. This is voice-quality acceptance on development
recordings, not factual, delegation or real-time acceptance. Keep this reference
and its hashes fixed for the next comparisons; no further listening choice blocks work.
Listening: http://localhost:8766/audio/voice-diagnostics/index.html .

Recorder repair removes two old Moshi underruns with identical generated PCM;
PersonaPlex already concatenates native frames. Fresh GPT-Live 7415050 still has
66 underruns with a 160 ms startup buffer; raw packet-stream WAV is quality-only.
CPU/GPU uncompiled Mimi decoder differences are small; no large numerical
compiler divergence found. PersonaPlex custom voice gives poor factual answers.
Lychee strictly loads all 982 tensors, generates four answers with corrected native
input clock (real CPU red/green 7415324), but also gets bread staling wrong.
Its offline generation plus vocoder takes 229.47 s for 70 s input; not a paced pass.
MiniCPM 7415165 answers one question and reaches 6.229 s backlog.

Revised measured-phrase objective passes CPU audit 7414927 and now has a completed
128-update pilot **7415494** on the same 33 TRAIN scenes. Fresh-process reload:
text/audio errors exactly 0.0; adapter SHA
0263c20d64a7649c9efd43a659e4fb31503e36c553071128531f78738577dc9f.
Four dependent development checks: private 7415517, simple 7415519, clarify 7415521,
cancel 7415522. Retrieve full outputs before interpreting behavior.
Next: pace the accepted PersonaPlex setup, compare native intelligence/latency,
and prove a causal worker-result interface before larger fine-tuning.

All 48 mixed codes/audio scenes plus parent metadata are backed up; three parent
ingestion hashes match. Historical completion statements/GPU-hour snapshots below
are superseded; latest snapshot through 7415227 is 2.403333 physical GPU hours,
excluding CPU allocation billing/premiums and pending jobs. This is not remaining
balance. No new compute authorization is required.

## ACTIVE CONTINUATION — 2026-10-13 recovery

**Plain words:** the owner has reopened the work to build meaningful duplex
speech with delegation. Start at [docs/RECOVERY_PLAN.md](docs/RECOVERY_PLAN.md).
The older "COMPLETE" statement below closes the prior timing-only phase.
New checks found 3.92 s stale audio in the microphone scheduler and invalidated
the historical delegation-mixture comparison (both launchers dropped their
selection/init/output arguments). The tracked launcher is repaired with a
regression test. Native, unchanged Qwen now generates six coherent speech
samples, all ASR-roundtripping at zero word error, but this batch path is not
live. Native MiniCPM now completes a 60-second paced duplex replay on the new
container (job 7377673), but reaches 3.10 seconds of input backlog and gives a
wrong bread-staling explanation. Jobs and exact limitations are recorded in
the recovery plan. Neither candidate has passed product acceptance.

Native Moshi has a verified 64-step adaptation pilot (7380505): frozen weights
unchanged, exact saved adapters and canonical float32 reload logits. Lower loss
did not establish useful spoken content. Meta tokenizer access is now verified;
the full ARC encoder and actual HTTP path pass (7386266).

The first reference replay omitted the upstream per-frame application step;
discard its negative reference-use verdicts. Corrected 7387777 applied all
33 reference rows, processed 80 ms frames in 16.5/18.6 ms p50/p95, and reached
78 ms maximum input backlog. It still answered incorrectly before a ten-second
bread result, then failed to correct itself afterward. This is fixed-fact
request-order integration, not actual research or general task routing. Human
listening, microphone acceptance, fresh content tests and timed-reference
training remain open. ASR timestamps for the last cut-off answer are invalid;
keep original audio and do not accept its aggregate WER as a quality verdict.

Seeded BF16 histories varied even with fresh processes and eager execution;
input-code and traced RNG hashes matched in the diagnostic. Cause unestablished;
no single-run causal pair claim. Recovery consumption is **1.093 allocated
GPU-hours**, not a verified remaining balance. No large training run or
production selection. Read the recovery plan before scheduling more.

## HISTORICAL STATE — prior phase, superseded by the recovery above

**Plain summary:** duplex-live set out to build a live duplex voice
model on Qwen3-Omni-30B-A3B (80 ms tick loop, Isambard GH200s). The
project is COMPLETE for this allocation (ends 2026-10-21; all
experiments closed by 2026-10-12). WHAT SHIPPED: a model that hears in
time and takes turns correctly, live through a microphone — final
reserved evaluation on never-touched lessons: F1 0.80-0.86, recall up
to 0.99, word onsets 12/12 within 1 s, zero interruptions. WHAT DID
NOT: free-run word CONTENT (digit soup; three fixes ruled out with
evidence — needs the Talker wired or next-phase scale) and
delegation-mixing (emission learnable, but mixing classes into
training breaks turn-taking at any ratio tried). Full narrative:
docs/FINAL-REPORT.md; every experiment: EXPERIMENTS.md (dated).

## Where everything lives (cluster $DL=/projects/u6si/duplex-live)

- DELIVERED MODEL: $DL/runs/scale1-q2-pt1/p1_checkpoint.pt + its gate
  $DL/runs/scale1-q2-gate1/gate_checkpoint.pt (use gate-threshold 0.10)
- MIC DEMO: repo demo/mic-demo.sh (sox mic | ssh srun | speaker; needs
  `brew install sox`; LESSON env picks the answer bank)
- GALLERY: local `python -m viz.listen --runs <dir>` (batches must be
  <runs>/<batch-name>/manifest.jsonl — one dir level, model.wav now
  listed). Local copies under runs/ (demo-q2, live-q2, ...)
- CORPORA (all fail-closed verified, quality.jsonl in each):
  synth-scale1-q/audio (159, the training corpus), synth-scale2-api
  (45), synth-scale2-api-local (52), synth-scale3-api (178),
  synth-deleg (81). MERGED VIEW: runs/shards-merged + runs/corpus-merged
  (515 lessons, prefixed ids q2_/a1_/a2_/a3_/d_)
- SURVEY: docs/research/2026-10-08-duplex-approach-survey.md (cited,
  8 ranked recs; rec #3 DuplexChat 282kh real audio is the next-phase
  data move)

## ENVIRONMENT — the container saga (critical, current)

- The fleet drivers were updated 2026-10-06 (580.173.02) and the OLD
  container (/projects/u6si/containers/latentsprites-pytorch-25.05.sif)
  FAILS on it everywhere (CUDA Error 803). DO NOT USE IT.
- USE: $DL/pytorch-new.sif (NGC nvcr.io/nvidia/pytorch:25.10-py3,
  pulled arm64 ON A COMPUTE NODE — login-node mksquashfs dies on
  thread limits; Docker-Hub pytorch tags are amd64-only) + overlay
  $DL/pylibs-new (transformers 5.17.0, accelerate 1.15.0; the pip
  torch/nvidia/triton PACKAGES AND THEIR .dist-info were purged so the
  container's own torch 2.9 wins — leaving dist-info causes an
  AuxRequest ImportError)
- Historical GPU-path slurm scripts were retargeted (evals/*, synth-filter).
  The old minicpm render path (pylibs-minicpm) is NOT retargeted. The new
  `evals/minicpm-recovery-probe.slurm` works on the new container using an
  isolated overlay and tested reference-WAV reader; see the recovery plan.
- KEYS: $DL/keys/elevenlabs.env + openai.env (chmod 600; creator-tier
  ElevenLabs credits nearly exhausted; OpenAI key valid, gpt-5-pro
  needs the /v1/responses API — banks_openai.py handles both)
- Login-node python is 3.6 — always run inside the container.
- Node health is mixed after maintenance; if a job dies in <60s check
  for Error 803 first, then pin to a node that works (sacct NodeList
  of a surviving job).

## Standing rules (owner-set, keep them)

1. Plain language ALWAYS — every md entry leads with "Plain words:"
   answering problem/did/found/means. The owner has enforced this
   repeatedly.
2. Cluster code changes: local commit+push, cluster git pull, md5
   verify. Never scp code.
3. Full logs on cluster jobs; greps only after fetching.
4. FAIL-CLOSED DATA RULE (owner directive 2026-10-02): nothing trains
   or evaluates on unverified lessons; consumers assert quality
   stamps at startup (data/synthetic/quality.py). Keep it.
5. ADR-0003 Amended A: closed-API content PERMITTED under the owner's
   enterprise agreement; keep provenance tags; never re-raise public
   ToS.
6. Anything producing voice must be audible via viz.listen before
   being believed.
7. Git: do NOT push with Claude co-author attribution (owner global
   instruction).
8. FRESH-SET RULE: barge-115/117/119 were reserved for one final
   evaluation — IT HAS BEEN SPENT (2026-10-12, verdict above). They
   are no longer reserved.

## Open items (next phase, none blocking hand-back)

- Word content: wire the Qwen3-Omni Talker into the loop OR real-audio
  scale-up (DuplexChat) OR RL only AFTER supervised content exists
- Delegation/streaming: emission is learnable (fires 1/1 at the right
  tick) but class-mixing breaks turn-taking at 1:2 AND 1:6 ratios —
  needs staged curriculum or a separated head
- Serving wall: dense turns ~5x over the 80 ms budget; note the new
  stack's raw forward is 115 ms p50 (vs ~59 ms decisions on the old
  stack) — kernel regression worth investigating before serving claims
- VAP referee (survey rec #2): untrained; DuplexChat download never
  started

## Known gotchas (each cost real time — check here first)

- evals/deleg-round.slurm + deleg-round-balanced.slurm carry broken
  --out/--lessons lines from a sed edit (training SUCCEEDS, checkpoint
  lands in repo-relative runs/p1-v1/, then the script dies at the
  diagnostic line "--lessons: command not found"). Run the end-of-job
  diagnostic STANDALONE via layera.freerun_diag (see jobs 7260780 /
  7363080 for the exact pattern).
- Merged-corpus held-out ids are PREFIXED (q2_dense_dialog-113, not
  dense_dialog-113) — the train script's held-out finder and any
  --scenarios list must use the prefix.
- Self-history recollect mode is ~35x slower (chatty trunk rollouts);
  corrupt mode runs full speed but did not fix content.
- The chat-via-ssh --wrap pattern breaks on nested $( ) — write slurm
  files instead.

---

## THE FAIL-CLOSED DATA RULE (owner directive, 2026-10-02 — standing policy)

**Plain words:** we trained on a data set that the pipeline kept only
by dropping whole conversations (91 of 240 rendered), whose word
timing inside clips was estimated rather than measured, and whose ear
clock drifted ~4% from the tick clock. Those gaps only surfaced weeks
later, after they had shaped every result. That must never happen
again. Nothing proceeds — no training run, no evaluation, no re-roll,
no live-loop verdict — on conversations that have not been verified,
piece by piece. (Precision note from the factory map, same day: the
filter already speech-to-texts every clip — detection exists; what
was missing is REPAIR at render time, word-level timing verification,
and asserts that stop consumers loading unverified lessons.)

"Verified" means ALL of:
1. **Every clip says what the script says — and is FIXED where it
   fails.** Each rendered clip is checked by speech-to-text against
   its script line (the check already exists in the filter; it must
   run inside the render loop); a clip that mismatches is re-spoken
   immediately (a few retries), and a conversation whose clips still
   fail is REJECTED, never kept.
2. **The timing answer key matches the real audio.** For every turn:
   the audio must actually contain speech where the key says someone
   speaks, and quiet where it says quiet, measured on the model's own
   80 ms tick grid. Any turn that fails the check rejects the
   conversation.
3. **The verification is recorded.** Each conversation carries a
   quality record (per-clip results, per-turn timing results,
   pass/fail). Training and evaluation jobs ASSERT at startup that
   every lesson they load has a passing record — and abort if not.
   The batch report states how many conversations were rendered,
   passed, failed, and why. Re-rolls inside the render are automatic
   and recorded; there are no silent retries.

The live loop (M2) inherits the same rule: any result it reports only
counts on conversations with verified answer keys.

**IMPLEMENTED 2026-10-02 (later — all four pieces, in code):**
- repair-in-render: `data/synthetic/render.py --repair 2` (per-clip
  whisper judge at TTS time, re-speak on fail, REJECT the conversation
  if a clip still fails; only passing clips enter the clip cache);
- timing audit: `data/synthetic/timing_audit.py` (whisper word
  timestamps vs the answer key's word placement, per assistant clip);
- quality stamp: `data/synthetic/quality.py` (quality.jsonl, schema
  dl-quality-v1 — verified = every clip passed AND timing passed);
- startup asserts: prepare, train_p1, train_gate, train_head_dagger,
  freerun_diag, probe_p1, contract_trace, m2/parity all abort with a
  plain message on any unverified lesson (train_p1/train_gate also
  refuse stale shards). No override flags — fail-closed.
Audit of the OLD batch (evals/data_quality_audit.py): 127/240 pass the
clip bar; 39 of 99 dropped conversations had exactly one bad clip
(repair converts them); 94 of 141 "passing" ones carried over-bar
clips. Re-render chain with the judge on: evals/scale1-quality.slurm.

## The frontier right now (2026-09-23)

- **Phase-1 training is a dose-response study through v11.** The ablation
  ledger with every recipe, exam number and verdict is **EXPERIMENTS.md** —
  read it before proposing any new run. Training is ~1.1 GPU-h per dose
  (55 min train + 10 min held-out exam), and **the exam predicts the live
  gate** (validated on v10: its exam word-slot sil-steals mapped exactly to
  live muting), so the exam is the default decision instrument; rerun the
  T1–T5 gate only when the exam moves.
- **All three live-gated checkpoints (v2 unweighted, v3 w50, v10 w50+tax)
  pass exactly 2/5**, and the shared failure is UNDER-SPEAKING: silence is
  ~95% of training ticks, so every model learns a silence attractor — even
  unweighted v2 mutes on t1 (0 word beats). v3 is still the only checkpoint
  that fires a countdown (t+30s, 50% exam) and speaks live; v10 fixed the
  rare-token thief (1–2/lesson) at the cost of all timing and all speech.
- **In flight: v11, the speak-up dose** (user-approved 2026-09-21):
  w50 + thief-pen 2 + `--word-weight 3`. **RESULT (2026-09-22): the exam
  moved decisively — words 0.185–0.387 (v2-level restored), t+30s
  partially back (1/4, prob 0.426), barge-class silence kept; cost: the
  t+30s thief re-empowered on the countdown-dense lesson (251 steals).
  Live gate: PASS 2/5, beat census v2-shaped (sparse, misplaced speech;
  mute on t1/t2). Exam tracks the AMOUNT of speaking, not placement.**
- **External adversarial review landed (CODEX_PLAN.md, 2026-09-21) and
  VERIFIED accurate; the owner APPROVED the 5-point adoption plan
  (2026-09-22). Executed:** (1) ledger corrections committed (v5/v6
  exposure, freerun 159-steals, rounding); (2) **gate instruments
  repaired** — marks aligned to true stimulus positions, latencies
  anchored at utterance END, unanswered questions counted (t1 needs
  ≥8/10 answered), t2 reply from barge END, t4 talk-over = premature,
  `<sil>`-run (not single tick) ends a turn, 16 kHz/24 kHz framing bug
  fixed — all guarded by `evals/harness/test_gate_regressions.py`
  (three-tone cheat now FAILS, honest traces PASS, all green);
  (3) **decode knobs live**: `--sil-bias` + word continuation
  (`[pad w1 w2]`, the training contract for 33.6% multi-piece words)
  in `layera/duplex.py`, plumbed through the harness;
  `freerun_diag.py` rewritten free-mode-only with the same knobs;
  `layera/contract_trace.py` prints the training-vs-live token
  streams. **All five points EXECUTED and answered (2026-09-22):**
  rescored verdict table (v2/v3/v10/v11 under honest instruments —
  the old "2/5 each" was composition noise), decode sweep (monotone
  dial: bias buys onset exactness with chatter; held-back lesson 017
  unlocked 0/19→19/19), and the live gate at bias 0/4: **v11's live
  muting was substantially a decode threshold** — bias 4 takes
  question-end answers 0/10→5/10 and produces the first post-barge
  reply (1920 ms), with the predicted costs (background chatter,
  one yield). Decode is a diagnosis knob, not a fix: latency (3.1 s
  vs 1.5 s), part-2 handback, and amount/control coupling are
  training-side and now measured separately. Old gate numbers are
  start-anchored history.**
- **Unseen-wording gate is now the selection instrument (2026-09-22,
  job 6781995): v11's bias-4 speaking was script recall — 0/10
  answered on new questions, chatter through the whole background
  tape. v3 keeps the most real behaviour: 2/10 on unseen wording at
  p50 280 ms (timing circuit intact, coverage is the failure) and
  clean t3+t5. MORE weight on the same 80 lessons is the least
  supported next step; data-side or capacity (Codex F5-2/F5-3) lead.**
- **Capacity test ANSWERED (2026-09-23, jobs 6792730/6814111/6814321,
  EXPERIMENTS.md "Capacity test"): NEITHER small lever learns 4
  conversations free-run. Rows-only (~41k params): word placement is
  saturated — 98.5% exact onsets, 97.1% on HELD-OUT conversations —
  but cannot learn silence (chatter 33–88% of sil ticks). Rows+adapter
  (~1M): learns silence TOO well — fully mute on 2/4 trained lessons,
  exact onsets 26%. Complementary failures at the same join point at
  TEACHER-FORCED vs FREE-RUN exposure (the model never sees its own
  chatter in training) as the common cause — capacity is exonerated
  for placement, the adapter is not a free upgrade. Also fixed:
  freerun's ±1s metric was vacuous (window contained its own gold
  tick), adapter-vs-CPU-feats device crash. NEITHER anticipated
  branch of Codex's completion criterion fired (adapter did not
  learn) — the rows-only ladder is not the thing to end; the thing to
  change is what training exposes the model to.**
- **Self-history probe series (2026-09-23, EXPERIMENTS.md "Self-history
  probe"): the exposure lever WORKS but has a split personality.**
  Probe 1 (cap-sh, self-history 0.5, unweighted): first lever ever to
  move free-run silence to 58–96% (trained AND held-out) — but it
  swings to the MUTE pole (exact word onsets 1.5%) with a new `<t+2s>`
  thief. Probe 2 (cap-sh2, + word-weight 3, job 6820759): **speech
  PARTIALLY returns on the barge lessons (13/21 and 10/20 exact,
  word acc 0.50–0.62, silence holding ~57% — the first config where
  speaking and stopping coexist), but the t+30s/t+10s thief eats the
  timed_task lessons (silence 34%, up to 505 steals; total exact
  onsets 25/68 = 37% vs the 90% bar). FAIL, but the shape is right.**
  Probe 3 (cap-sh3, job 6827932) added `--thief-penalty 2` (v11's
  recipe + self-history): **FAILED WORSE — the tax re-muted the model
  (4/68 = 5.9% exact, 2/4 lessons fully silent) while the thief
  SURVIVED (42 t+30s steals) — the same cliff the ladder hit at v10:
  penalty buys quiet, not precision.** Probe series verdict: the
  exposure lever works and word-weight 3 revives speech, but the
  config is knife-edge dose-response, not a stable fix.
- **OWNER DECISION (2026-09-24): the next lever is the DATA side —
  word-dense lessons attacking the ~95%-silence root.** Built and
  committed: `dense_dialog` (rapid Q/A, 71–72% assistant speech, new
  hand-written 20-pair bank) and `dense_barge` (interruption teacher
  in the dense regime, 90%+ speech), both through the existing
  render→filter pipeline (`CLASSES=` env in synth-render.slurm).
  Batch `synth-dense1` (per-class 8, seed 907, chatterbox) rendered
  2026-09-24. Probe 5 (probe-2 recipe verbatim on the 4-dense-lesson
  rig, job 6849442) COMPLETED and scored 2026-09-25: **FAIL on onsets
  (17/304 = 5.6% exact vs 90% bar), PASS on chatter (7.5%, just under
  the 10% bar).** Not muted — that's new — but placement is
  uncorrelated with gold: the ±1 s hit rate (52%) is at/below what its
  own chatter would produce by chance. Right amount, wrong places.
  NEXT: owner decision — word-weight 6 probe on the dense rig
  (pre-registered next dose, ~5 h GPU), or treat the labeling contract
  (one word tick per spoken word) as the suspect and discuss the spec
  change. Audition audio for dense-sh1 → `runs/model-audio-dense-sh1`.
- **PROBE 6 (dense-sh2 = ww6, job 6865063, 2026-09-26): FAIL both
  bars — dose went the wrong way.** Exact onsets 4.6% (worse than
  ww3's 5.6%), chatter 10.6% (over bar), and the <t+5s> thief
  exploded onto silence (993 steals; silence acc 0.91 → 0.22–0.44;
  one lesson bistably went near-mute instead). Loss-side dosing is
  REFUTED; per the pre-registered tree the labeling contract (one word
  tick per spoken word, ~85% of grid ticks are silence) is now the
  prime suspect. NEXT: owner decision — per-tick speech re-labeling
  (SPEC change, rebuild shards + one probe ~5 h) vs move up the stack
  (M2) accepting Layer-A timing as-is.
- **OWNER DECISION (2026-09-26): v2 label contract — per-tick speech.**
  New `<speak>` token (tokenizer/vocab.py): every tick inside an
  assistant utterance is WORD or `<speak>`; between-utterance gaps keep
  true `<sil>`; only SIL upgrades so timing tokens/yields keep their
  slots. Realized density: speech labels 59.7% vs sil 39.9% (was
  12%/88%). word-weight back to 1.0 (it only compensated v1 sparsity).
  freerun_diag v2 metrics: run-onsets + speech P/R/F1 (unit tests in
  evals/harness/test_freerun_metrics.py, all green). Old shards (v1
  labels) stay valid for v1 comparisons; `shards-dense2` = v2 rebuild.
  NEXT: build shards-dense2 → probe 7 (dense-pt1): SH 0.5/ww 1.0 on
  the same dense rig, pass bar ≥90% exact run-onsets + <10% chatter →
  audition + matched-voice render as usual.
- **PROBE 7 VERDICT (dense-pt1 = v2 + ww 1.0, job 6885910): FAIL —
  re-muted on balanced labels.** 3 of 4 trained lessons emit `<sil>`
  on every speech tick (speak acc 0.000, speech F1 0.000);
  dense_barge-004 babbles `<speak>` (acc 0.849 but false speech on
  166 quiet ticks, word acc 0.026). Held-out F1 0.00–0.16. Dose law
  now clean across BOTH contracts: ww1 → mute (probes 1, 7), ww3 →
  speaks (probes 2, 5), ww6 → babble+thief (probe 6). Label balance
  is NOT the mute attractor — loss shape is (sil/speak = one cheap
  token; words = the whole BPE vocab). Input-evidence insight: the
  model never hears itself, so its own history is the only self-speech
  evidence — under SH a collapsed model feeds itself silence.
  Step-8 recollect: untrained model picks words everywhere (450/450);
  by end it learned sil mastery and let go of speech. NEXT: owner
  decision — probe 8 = v2 + ww 3 (the only recipe that ever spoke,
  single variable vs probe 7) vs SH 0 (always-gold history while
  learning) vs move up the stack.
- **OWNER STRATEGY (2026-09-27): bounded ladder + M2 in parallel;
  the allocation must NOT lapse.** Allocation ends 2026-10-21. Goal:
  by ~2026-10-14, a model that speaks and takes turns audibly
  correctly through the M2 live serving loop, with remaining node-hours
  spent on the full-data run that gets there. Layer A is worth
  perfecting ONLY insofar as it fixes the live duplex experience by
  ear (owner's condition); if the ladder can't get there, pivot the
  hours instead of grinding probes. Probe 8 (dense-pt2 = v2 + ww 3,
  job 6901314) is the MUTING ENDPOINT: speaks → ladder moves to
  placement; mutes → ww exhausted, next is SH 0 (teacher forcing);
  if SH 0 also fails → redesign, not more probes. Probe 7's audition
  render is on the gallery (runs/model-audio-dense-pt1, matched
  voices; mostly-quiet samples as expected, barge-004 babbles).
- **PROBE 8 VERDICT (dense-pt2 = v2 + ww3, job 6901314): FAIL pass
  bar — but the MUTING QUESTION IS ANSWERED: ww3 revives speech
  (speak acc 0.000→0.30 on dialog-001); the model cannot PLACE it
  (word acc 0.02–0.05 ~random, exact onsets 1–3 of 3–5, chatter over
  bar). Failure moved from "won't speak" to "speaks at wrong times" →
  per the owner endpoint rule the ladder SHIFTS TO PLACEMENT. Next:
  probe 9 (dense-pt3) = v2 + ww3 + SH 0 (single variable vs probe 8):
  teacher forcing gives complete self-speech context while placement
  forms; exposure later. M2 milestone 1 landed in parallel: real-time
  tick parity PASS (clock changes nothing); kernel bench: batched_mm
  stays (eager 3× slower AND 58% argmax flips); piece-loop arithmetic
  says a SPEAKING model at ~57 ms/forward cannot sustain the 80 ms
  budget during words — decode speed is the M2 problem the moment
  Layer A delivers a speaker.
- **PROBE 10 VERDICT (dense-pt4 = pt3 rows + SH 0.5, job 6906391):
  CURRICULUM FAILED — placement decayed AND chatter remained.** Exact
  onsets 5/5 5/5 4/4 3/3 -> 0/5 3/5 1/4 1/3 (trained), two lessons
  fully mute; recollection log shows the in-training model flip-
  flopping chatter <-> mute across self-exposure recollections
  (1285/1344 speech-first -> 2/1344 -> 390/450 -> 2/304), so training
  saw contradictory self-histories and lost the placement it started
  with. Per the endpoint rule: REDESIGN, not more probes. Redesign
  target = the exposure distribution: gold-schedule self-history
  (gold words at gold ticks in history, gold silences as targets)
  instead of recollected free-run. Decision brought to owner
  2026-09-27. Audition renderer fixed for chatter era (soft limiter +
  solo model.wav per scenario; dense-pt3 re-render = job 6908524).
- **PROBE 11 VERDICT (dense-pt5 = pt3 rows + corrupt-gold SH 0.5, job
  6911469): FAILED the bar, but the trade-off is now LAW.** Corruption
  taught recovery (sil acc 0.5-0.9 vs probe 9's ~0-0.36) yet exact
  placement decayed again (2/5 3/5 1/4 1/3) and chatter stayed 65-94%
  of quiet ticks. Three-point law: the placement mapping lives exactly
  in clean-gold history; ANY training-side history corruption erodes
  it ~1:1 against discipline. Bottleneck looks structural (one head,
  one delta, ww3 word pole) -> decision brought to owner 2026-09-28
  (aux activity gate vs speaking-state channel vs stopgap+scale).
  Stopgap audition LANDED: pt3 + min-gap 4 + user-energy gate (hold
  3) = all onsets within +-1s, chatter cut 3-10x, words 297-887 ->
  60-188; gallery: runs/model-audio-dense-pt3-gated. TTS overlay dir
  is pylibs-minicpm (no "o") — two jobs died on the wrong name.
- **GATE 1 VERDICT (aux activity gate on FROZEN dense-pt3, job
  6912904): a real learned muting dial, but its scores don't rank gold
  onsets above chatter — bar still FAILED.** Gate trains cleanly
  (acc 0.990; first try 6912797 OOMed: batched_mm gathers ~43 GiB at
  training seq lengths — eager experts, same as train_p1). Sweep
  (thr 0.35/0.5/0.65 trained; 0.5 held): dialog-001/dialog-002 FULLY
  MUTED at every threshold (gold onsets score <0.35 — gate P printed
  per-onset now), other lessons chatter 8.8-44% (barge-001@0.65 passes
  the chatter half alone at 8.8%) but exact onsets 0-2 per lesson
  nowhere near >=90%. Four-point law (9/10/11/gate1): the trap follows
  the training-STATE distribution, not the parameter grouping — the
  gate only saw gold + bounded synthetic-corrupt states, decode visits
  pt3's own compounded states (all-sil history at first onset,
  always labeled 0 in training, true label 1). Audition render
  submitted (dl-dgaud): gate-0.35-only + gate+stopgap-rules combos,
  full TRAIN4, -> runs/model-audio-dense-pt3-gate035 / -gate-combo.
  Candidate next (owner decision): DAgger-style gate round 2 — train
  on pt3's own free-run rollout states with gold labels (~40 min).
- **Listenable checkpoints (owner request, 2026-09-23): audition
  renders of the four capacity/self-history checkpoints free-running
  the 4 trained conversations — the model's words spoken by TTS at the
  exact ticks it chose, over the conversation audio — into
  `runs/model-audio-*` (gallery layout). First job (6827931) died on a
  MiniCPM-o TTS internal crash mid-word; renderer now skips bad words
  (retry once, then continue). rsync to the Mac, then
  `python -m viz.listen` → http://localhost:8765. The gallery now
  binds BOTH loopbacks (dual-stack socket): some browsers resolve
  localhost to ::1 first and showed ERR_CONNECTION_REFUSED against the
  old IPv4-only bind (fixed 2026-09-24).**
- **Where this goes after v11:** if the speak-up lever works, the Phase-1
  checkpoint target is "v10's quiet control + v2-level speaking + some
  timing"; if it cliffs too, the fallback is data-side (word-denser
  lessons) or accepting v3 and moving up the stack (M2 real voice, serving).

## Current status

- **Runbook 00/01: DONE (Issue #1 closed).** Agentic SSH via
  `ssh u6si.aip2.isambard` (Clifton cert window is 12 h and does NOT
  auto-renew on attempts — corrected 2026-09-26 after an expired cert
  blocked all ssh for 5 h while jobs ran fine on the nodes. On
  `Permission denied (publickey)`: check
  `ssh-keygen -L -f ~/Library/Caches/clifton/u6si.aip2.isambard-cert.pub | grep Valid`;
  renewal needs the owner to run `clifton auth` and approve the Keycloak
  device URL in a browser within the code's few-minute lifetime). ARM
  env verified on GH200; Qwen3-Omni thinker smoke passed. Env
  definition incl. all gotchas: `training/environment.md` — read before any
  GPU work.
- **Runbook 02: DONE (Issue #4 closed).** Both references measured on one
  scripted T1–T5 protocol (`evals/baselines/`): PersonaPlex passes the 3 s
  thinking pause, MiniCPM-o passes backchannels; **both fail addressee
  detection (T5)** → factory target. Layer-A code notes:
  `docs/notes-minicpm-layera.md`. Recordings on Lustre (`runs/bakeoff*`) =
  data seed.
- **Live demos: DONE (2026-09-16), then stopped by user decision.** Both
  served bare-metal from SLURM jobs (`pplex-live.slurm`, `minicpm-live.slurm`
  — MiniCPM demo = OpenBMB 3-stage gateway/worker/backend ported to Apptainer,
  overlay `pylibs-mcpmdemo`, dep pins in the slurm script). Relaunch:
  `sbatch <script>` + `ssh -L <port>:<node>:<port> u6si.aip2.isambard`.
  **Live-test findings (user):** MiniCPM-o live = very slow/unresponsive
  (expected: PyTorch demo, no torch.compile, ~0.9–2 s/unit on A100-class,
  worse untuned); PersonaPlex live = genuinely duplex but hard to interrupt
  with the default persona (scripted T2 replay yielded ≤0.5 s). Conclusion:
  neither reference is a product — they are the behavioral bar + tooling.
- **Synthetic factory v0.2 (Issue #3, in progress — 2026-09-17 session).**
  v0.1 scale render (32 scenarios, 16.2 min) verified on Lustre
  (`runs/synth/`), then three v0.2 upgrades landed:
  1. **Quality filter** (`data/synthetic/filter.py`, `synth-filter.slurm`):
     rule checks (clipping, same-channel collisions, barge-in cut placement)
     + whisper-large-v3-turbo per-utterance round-trip WER (MIT, staged in
     hf-cache; runs under `pylibs`). Short refs (≤4 words: backchannels,
     spelled syllables) are duration-gated — ASR tokenization can't judge
     them. Old batch: 19/32 pass; all fails were generator bugs (below) or
     stale-audio artifacts.
  2. **Generator scheduling fix**: v0.1 placed same-channel starts from
     65 ms/char estimates; measured rate is 72–105 ms/char + lead-in.
     `_est_ms` (800 ms + 80 ms/char) now schedules all same-channel starts;
     local check: 0 collisions at per_class=8.
  3. **Voice pool** (`data/synthetic/voices.py`): 39 LibriSpeech
     test-clean speakers (CC-BY-4.0, staged at `$DL/models/voices/`) + 3
     MiniCPM anchors; `render.py --voices` assigns one distinct voice per
     channel, deterministic per scenario_id, `--anchor-prob 0.5`. Content
     banks expanded (15 explain / 10 stories / 12 hard questions / 8 timed
     topics / 9 barge pairs; paired barge/yield texts; timed-task requests
     match delivered cadence).
  **v0.2 VALIDATED (2026-09-17)**: batch `runs/synth-v02/` (32 scenarios,
  17.3 min, seed 41) → **28/32 pass (87.5%)**, 0 same-channel collisions
  (7 TTS overruns auto-repaired at mix time), voice pool validated (39
  LibriSpeech voices ≤ 0.074 weighted WER vs anchors 0.042–0.105, no
  pruning needed). Remaining fails = genuine defects the gate caught
  (backchannel TTS babble on 1–2-word prompts, marginal WER outliers) —
  attrition covered by over-render. **Throughput: ~0.7 GPU-h per audio-hour
  → 50 h ≈ 35 GPU-h; compute is not the bottleneck.**
  **Fidelity policies (2026-09-17, from user audition of banks-smoke):**
  (1) render order is cut → barge-yield push → same-channel repair → mix
  (repair-before-cut shipped 12–21 s reply gaps — user caught a 21,120 ms
  one); (2) **barge-yield policy**: assistant speech never starts during a
  realized `barge_in` — pushed to barge actual end +240 ms (human
  turn-taking offset; trains clean yield, never talk-over-interrupter);
  (3) filter gates `yield_latency` (>2500 ms) AND `yield_overlap`
  (<−160 ms) so neither failure ships unheard; (4) TIMED_TOPICS bank items
  must be numerically consistent (count nouns == callout count, "for N
  seconds" == span) — banks.json pruned 24→22 timed topics.
  **Audit by ear**: `python -m viz.listen` → http://localhost:8765 —
  serves locally synced batches (mixed/stems/per-utterance players joined
  with WER + whisper hypotheses) + voice pool; rsync recipes in its
  docstring; refresh after each sync.
  **ACCELERATED (2026-09-19, user call):** multi-worker GPU rendering
  (`--slice` shards + manifest merge + atomic cache publishes; chatterbox
  4 workers/GPU, minicpm 2) ≈3.5× batch throughput; **big7–10 queued in
  parallel** (seeds 707-710). Barge-in audition follow-up measured: pre-
  barge assistant silence median 0 ms / p90 0 ms across 96 scenarios in
  big5/2/1 — barges land mid-speech, not after predictive pauses; the
  perceived pause = prosodic dip/outlier, NOT a pipeline bug, arguably
  naturalistic. Layer-A (#7) starts EARLY (~Sep 22) instead of Sep 28 —
  all dependencies (tokenizer, factory data, eval gate, delegation brain)
  are at v0-done.
  **Layer-A Phase-1 status (2026-09-21): shards + trainer built; first
  gradient chain in flight.** M1 driver done (audio flows during
  generation, append-only KV; feed+decode p50 144 ms vs 80 ms budget —
  polish deferred to serving, off critical path). `layera/prepare.py`
  shards verified: ~1.04 audio tokens/tick (13 tok/s ear rate ✓).
  Training OOM chain (3 jobs, root-caused — do NOT repeat): batched_mm
  experts keep ~24 GiB (gate_up) + ~12 GiB (down) per-token expert
  gathers alive per layer → 59 GiB model + 35.7 GiB = the whole 95 GiB
  card; gradient checkpointing does NOT fix it (backward recompute holds
  all gathers). Fix: `experts_implementation="eager"` for TRAINING jobs
  (per-expert loop, ~6 MiB); keep batched_mm for inference jobs.
  Also: thinker `text_config.tie_word_embeddings=False` → embed and head
  are SEPARATE tensors; checkpoint saves both deltas
  (emb_delta_new_rows + head_delta_new_rows). Jobs: 6728087/6728254/6728411
  OOMed, 6728455 (eager) SUCCEEDED (32 steps, loss 10.4→~0.9).
  **Probe v1 (held-out, job 6742556): `<sil>` 0% → 95–98% acc on unseen
  lessons; word pieces improved (not a silence collapse); `<t+Ns>` 0% —
  data gap (those lessons contained almost no timing tokens).**
  **v2 (job 6742755, 2026-09-21): 80 class-balanced shards
  (`prepare.slurm PER_CLASS=10`), 400 steps ≈ 57 min, loss → 0.19.**
  Probe v2 (6745751, timed_task held-out): `<sil>` 99.4–99.7%, word
  pieces 20–48%, **`<t+Ns>` STILL 0%** — root cause: ~30 timing positions
  in ~50k graded (0.06%). NOTE: v2 trained into `runs/p1-v1` (default OUT
  before parametrization) and was renamed; the true v1 32-step checkpoint
  survives in `runs/p1-v1-32step` (verified via the `args` dict saved
  inside each checkpoint).
  **v3 (job 6746779, timing×50 weighted CE): MIXED — `<t+30s>` 0%→50%
  acc (prob ~0.55) on held-out, FIRST timing learning; but cost: `<sil>`
  99.5%→53.8/67.7% on 2 of 4 held-out scenarios, words roughly halved.
  The 50× push drags the shared context; everything else pays.**
  Dose-response COMPLETE (all probes corrected, SHARDS_DIR set,
  classes=timed_task,barge_in, 8 held-out each):
  - v2 (1×/1×): t+30s 0% prob 0 | sil 99.5% | yield 0%
  - v4 (k1×w10): t+30s 0% prob 0.15 | sil 97-100% ✓ | yield 0%
  - v3 (k1×w50): t+30s 50% ✓ | sil 53-98% ❌ | yield untested
  - v5 (k5×w1): t+30s 0% prob 0.000 | sil 97-100% ✓ | yield PARTIAL
    (1/6 scenarios 100%, others prob 0.03-0.76 — first yield learning)
  **Conclusion: only the DATA×WEIGHT product matters (~50 effective =
  timing learns); concentrating it in weight (v3) breaks <sil>; yield
  responds to data-spread. <yield> was a second rarity token (≈0.02% of
  graded) no weighting covered until now.**
  **v6 RESULT (CORRECTED 2026-09-22: the run changed THREE things vs
  v4, not one — shards-v6 visit counts barge 150×/timed 100×/others
  20–30×, yield weighting ON, and unshuffled sorted order changes the
  last class seen; "k5" never meant 5× distinct data): t+30s prob 0.000
  (printed; rounded to 3 dp) — the w10 timing signal (v4: 0.15) was
  gone. Sil perfect (97-100%), words best of any run (0.10-0.39),
  yield weaker than v5 (max 0.35). The original "DATA×WEIGHT law is
  dead" conclusion is CONFOUNDED — no clean single-variable
  attribution. Suspected AdamW scale-invariance weirdness stands as a
  hypothesis only.**
  **THIEF NAMED (6756308, v3 rerun with failure-argmax histogram): the
  sil "damage" was never broken silence — it is <t+30s> OVER-FIRING at
  sil positions (029: 584/588 sil failures = t+30s steals; 014/017/018
  same pattern). Timing recall fine; precision broken. Per-position
  weights boost the up-push but the down-push at sil positions stays
  weight 1.**
  **v7 VERDICT (6756309→6756310, k1 w30): t+5s LEARNED (1/6 scenarios
  100% acc prob 0.772; rest 0.047–0.372) with MILD over-firing (4–5
  sil steals/lesson — ~100× better than v3); t+30s NOT learned (prob
  0.014–0.032 — LOWER than v4's 0.15: at this dose the frequent t+5s
  grabs capacity first); sil 95.0–99.8%; yield 0%. Thresholds: t+5s
  crosses between w10 and w30; t+30s needs ~w50.**
  **v8 VERDICT (6756384→6756385, k1 class-cost 50): FAILED — label-blind
  loss. ∂L/∂z_k = p_k(c_k − E[c]); the −c_y term is a CONSTANT so the
  correct answer never enters the gradient — the only learnable content
  was "suppress rare tokens everywhere". Exam: t+*s prob 0.000 in all 8
  scenarios; sil acc 99.3–100% but prob flat 0.250 everywhere (learned
  constant, zero context); yield prob 0.250 acc 0; words ~0.
  --class-cost kept for the record; superseded by --thief-penalty.**
  **v9 LAUNCHED (2026-09-21): k1 w50 + --thief-penalty 10 — v3's proven
  up-push (label-dependent weighted CE) + a separate down-push taxing
  rare-token mass ONLY where the label is not rare (the thief's turf).
  Single-variable change from v3. FULL ABLATION LEDGER: EXPERIMENTS.md
  (per-run recipes, exam numbers, verdicts — keep current with every
  run).**
  **LIVE GATE #1 (v3, 6762718): FAIL 1/5 — constant tone, zero decodable
  words, never sil/yield/timing. FREE-RUN DIAGNOSTIC (6762979, v3 ckpt,
  training-format replay): free-run is CLEAN (sil 0.92–0.94 self-fed vs
  0.38–0.40 gold-forced; zero chatter; 81–110 recoveries to sil) → the
  live babble was the HARNESS FORMAT (chat envelope + processor
  <|audio|> path), not the model. Also: forced gold t+30s history PRIMES
  the thief (675 sil steals); free-run scores 0/6 at gold t+30s
  positions BUT fires t+30s 159× at wrong sil positions (61+98,
  corrected 2026-09-22 — the "needs the primer" story is not
  supported). FIX: duplex.py v2 + LayerATarget v2 — live now
  mirrors prepare.encode_audio (no envelope, 4 s-window ear features
  into <|audio_pad|> slots, one beat per feature, timestamped output
  tape). Live-gate rerun on the same v3 ckpt = the A/B. Full tables:
  EXPERIMENTS.md.**
  **LIVE GATE #2 (v3, format-fixed + per-test reset, 6763486): PASS 2/5
  — t3 ✓, t5 ✓ (0.72 s vs 22.0 s pre-fix); t1 answers but p50 14.9 s
  late, t2 silent after barge, t4 answers during part 1. Two harness
  bugs fixed en route: cross-test session leak (KV/clock carried
  between tests — T3's "babble" was stale-conversation continuation;
  Target.reset() per test collapsed it 562→3 word beats) and render
  tape overflow. Failures are now MODEL quality (sparse/early
  speech), not plumbing. Timing over-firing: 79/9/8 timing beats per
  test, silent — no metric cost yet.**
  **v9 VERDICT (6758553→6758554, k1 w50 + thief-penalty 10): FAILED —
  over-suppression ("the Monk"): sil 99.9–100% acc / prob 0.958–0.995
  (best ever) but t+30s, t+5s, yield, words ALL ≈0.000. The tax fires at
  every non-rare position (~1500/lesson) vs the up-push's ~1/lesson, so
  pen 10 swamped w50. Dial bracketed: pen 0 → timing learns + ~150
  steals/lesson; pen 10 → nothing. Next probe: pen 2. Ledger +
  post-mortems: EXPERIMENTS.md.**
  **v10 VERDICT (6762715→6762716, k1 w50 + thief-penalty 2): rare-token
  thief FIXED (1–2 steals/lesson, bar ≲5) and sil is the best of any
  w50 run (99.2–100% acc, prob 0.93–0.98) — but the tax's cliff sits
  between pen 0 and pen 2: t+30s died anyway (prob 0.003–0.013, 0/6),
  t+5s weak (prob 0.010–0.352, 0/4), yield 0. THE BOTTLENECK MOVED: the
  dominant exam error is now <sil> eating word positions (21–30/lesson)
  and words are below even unweighted v2 (0–19.4% vs v2's 20–48%) —
  every weighted run degraded speaking. The gate's failing tests (t1/t2/
  t4) are speaking failures; passing (t3/t5) are quiet tests. Live gate
  on v10 launched for the A/B (6763949). DECISION POINT: next GPU-hours
  go to word-beat reliability, not more rare-token/countdown surgery.**
  **LIVE GATE #3 (v10, 6763949): PASS 2/5 but MORE muted than v3 — 0
  word beats in all five tests (v3: ~9/test), 1 timing beat total; t5
  false engagement improved to 0.0 s, but t1 has no answers at all.
  THE EXAM PREDICTS THE GATE: word-slot sil-steals on the 10-min exam
  mapped exactly to live muting — the exam is now a trusted gate proxy.
  v2 (unweighted best-speaker, no timing) live gate launched (6764321)
  — if it beats 2/5 the T-gate needs only word+sil decisions and the
  recipe question becomes "inject timing without damaging speaking".**
  **LIVE GATE #4 (v2 baseline, 6764321): PASS 2/5 — ALL THREE ckpts
  pass 2/5; the gate is bottlenecked on speech production, not style.
  v2 is the only one that ever replied post-barge (1600 ms vs 800 bar;
  barge→sil 1760 ms) but it ALSO mutes on t1 (0 beats) and misplaces
  speech on t5 (10 word beats into background chatter, 0.48 s). So
  under-speaking is STRUCTURAL: sil covers ~95%+ of training ticks and
  every model, even unweighted, learns a silence attractor. Full
  three-way table: EXPERIMENTS.md. Next lever: weighted-CE on WORD
  positions (weight 1 today vs 95% sil background), doses bracketed.
  AWAITING USER DECISION on the next training run (stop-agreement).**
  **USER DECISION (2026-09-21): speak-up dose approved. v11 LAUNCHED
  (6765121 → exam 6765122, chained afterok): w50 + thief-pen 2 +
  --word-weight 3. Bracket plan: words flat on exam → 10; sil breaks →
  1.5. Exam is the decision instrument; live gate rerun only if the
  exam moves. New dial lives in train_p1.py (--word-weight) +
  train-p1.slurm (WORD_WEIGHT env).**
  **EVAL-GATE PATH BUILT (2026-09-21, pre-verdict): `layera/duplex.py`
  (DuplexThinker — M1 append-only tick loop + checkpoint deltas +
  trained-model emission decisions; mask = chat envelope only, the
  duplex ids MUST stay decodable since register_into marks them special)
  + `evals/harness/targets.py` LayerATarget (1 s chunk → 12-13 ticks;
  word beats emit a synthetic 220 Hz tone — voice is M2 — so the energy
  metrics measure speaking DECISIONS; n_timing/n_yield logged per
  chunk) + `evals/layera-live.slurm` (CKPT env, TESTS env). Run:
  `sbatch evals/layera-live.slurm` after a cluster `git pull`. Replay
  metrics score T1–T5 from the wav envelope exactly as for MiniCPM.**
  (v7/v8 done — verdicts above; per-run ledger lives in EXPERIMENTS.md.)
  Probes use
  env inheritance (NOT `--export`) for comma values — `--export` splits
  on commas, which silently truncated CLASSES=timed_task,barge_in to
  timed_task in probe 6746780. **probe-p1.slurm GOTCHA: always pass
  SHARDS_DIR** — without it the probe silently includes TRAINING rows
  (header prints "training set was 0 shards"); first v4/v5 probe attempt
  (6749312/6749314) was discarded for exactly this. Corrected:
  6751164 (v4) / 6751180 (v5). Checkpoints kept:
  runs/p1-v1-32step, runs/p1-v2-400step-unweighted, runs/p1-v3, p1-v4, p1-v5.
  **Hours-scale begun (2026-09-18): `synth-big1` = 256 scenarios, 2.2 h,
  1h24m GPU (0.85 GPU-h/audio-h), 182/256 pass (71%), WER median 0.018 /
  p90 0.111.** **VOLUME TARGET CLOSED (2026-09-19): 50.43 h gate-passed /
  6,388 scenarios across big1–31** (chatterbox 53%, minicpm 47%; all 42
  voices; class coverage 274–942 per class — lighter classes flagged for
  Phase-1 sampler weighting). Factory now runs on-demand for Phase-1;
  quality tail (class rebalance, optional Sesame engine) as needed.
  Failure profile is ONE engine limit: MiniCPM-o TTS babbles
  on 1–2-word prompts (tail_babble ×147; backchannel 32/32, timed_task
  29/32 fails). **Second engine staged and winning: Chatterbox
  (Apache-2.0)** — overlay `pylibs-chatterbox` (torch-stripped, weights in
  hf-cache, native 24 kHz), renderer refactored to pluggable engines
  (`--engine {minicpm,chatterbox}` in render.py, `ENGINE=` in
  synth-render.slurm; same voice pool + all mix-time policies + same
  filter). **`synth-big2` (Chatterbox, 256 scen., seed 303): 210/256
  (82%), WER median 0.000, timed_task 8 vs 29 fails, backchannel 23 vs 32;
  register-distinct voices active. Night 2026-09-18: big3 (chatterbox,
  206/256) + big4 (minicpm, 178/256) — engine rates stable (CB ~81%,
  MCPM ~70.5%). **FACTORY TOTAL: 5.81 h gate-passed / 776 scenarios
  (runs/synth-big{1..4}).** Clip-cache for ≤6-word renders LANDED, active
  from the next batch (`--clip-cache`, engine-scoped keys;
  `purge_cache.py` closes the loop — bad keys re-roll, passing clips
  freeze). Sesame CSM-1B
  fits the same abstraction later (mimi codec stack present in
  pylibs-pplex). **Render jobs self-chain their filter via `afterok`** —
  batches finish server-side through cert lapses; only the rsync-to-gallery
  step needs the agent.
  **Gate set added 2026-09-17/18 (from user auditions):** yield_latency
  >2500ms, yield_overlap <−160ms, insertion gate (tail_babble: ≥2 extra
  words anywhere, ≥1 on ≤6-word refs — caught a hallucinated "Doctor"),
  barge-yield policy pushes assistant speech behind realized barge end
  +240 ms (render-side, step 2.5, AFTER cut), user/assistant voices must
  differ in F0 register (voices_f0.json, pyin-measured).
- **Overlays**: `pylibs` (Qwen, transformers 5.17, +whisper judge),
  `pylibs-minicpm` (transformers 4.51 pinned, +accelerate 1.12),
  `pylibs-pplex` (moshi; sphn source-built — recipe in environment.md),
  `pylibs-mcpmdemo` (live demo).
- **Isambard workspace**: `/projects/u6si/duplex-live/` (`repo/` = git clone,
  read-write via deploy key; `models/hf-cache`, `models/voices/`, `runs/`,
  `logs/`, `tools/`). Login-node background processes are reaped when the
  ssh session exits — run long staging in the ssh FOREGROUND or sbatch it.
  SLURM `--output=logs/...` is relative to the SUBMISSION cwd (repo/ vs $DL/
  both appear in logs — check both).

## The program

Build **Duplex Live**: a full-duplex, time-aware interaction model (GPT-Live /
TML interaction-model class) at mini open-source scale on Isambard-AI.

## Roles of the two references (clarified after live testing)

- **MiniCPM-o 4.5** (Apache-2.0): (a) reference implementation for our
  Layer-A tick loop (notes-minicpm-layera.md), (b) the factory's TTS voice
  engine (proven), (c) possible fast-path fine-tune base if Qwen3-Omni
  Layer-A slips. Its demo latency is a systems artifact, not the ceiling.
- **PersonaPlex 7B** (NVIDIA OML — eval/research only, NEVER training data):
  the interactivity bar + failure taxonomy source (T4 pass / T5 fail etc.).

Path to GPT-Live quality = our own model: Qwen3-Omni 30B + Layer-A spec
(docs/09) + synthetic-factory data + timing-aware evals (Issues #2/#3/#6/#7).
The references define the bar; they are not the product.

## Key decisions (ADRs in decisions/)

- **ADR-0003 (2026-09-20, supersedes ADR-0002's bright line):** provenance
  classes replace the closed-API training ban — open / closed-labeled /
  closed-content (owner sign-off). Jev may label training data; streams
  tagged at creation; purge-by-provenance retained. TypeSafe Jev wired:
  `evals/harness/jev_scorer.py` (use case 1 live — semantic verdicts
  alongside energy metrics), SDK overlay `pylibs-typesafe`, key at
  `$DL/.typesafe_key`. Use cases 2-4 (bank audit, sampler scores, Layer-2
  labeling) queued.
- **One product model:** Qwen3-Omni 30B-A3B (Apache-2.0) — fine-tune duplex in
  via the Layer-A spec (docs/09): 80 ms tick loop, interleaved causal stream,
  new `<sil>`/timing/`<delegate>`/`<yield>` embeddings. Backbone frozen Phase 1.
- **Brain:** Kimi API behind an async delegation boundary. Closed-API outputs
  never enter training data.
- **Duplex = 3 layers:** capability (architectural: tick loop), policy (data:
  synthetic factory), clock (systems: streaming session).

## Hard constraint

Isambard-AI Phase 2 allocation: **4,974 node-hours, terminates 2026-10-21**
(~5 weeks from 2026-09-14). Weeks 1–2: bring-up + bake-off + delegation v0.
Weeks 3–5: Layer-A + Phase-1 fine-tune → model v1.

## Next actions (in order)

1. Issue #3 — keep the factory accumulating (big5/big6 rendering; batches
   self-chain render→filter→purge; rsync + audition via viz.listen). Then
   Issue #5 — delegation boundary v0; ACP is the preferred shape
   (docs/notes-qwen-live.md) over a Kimi-specific adapter. Issue #6 — eval
   harness v0 (steal the frameHash/ACK diagnostics pattern from qwen-live).
2. Issue #7 — Layer-A on Qwen3-Omni + Phase-1 fine-tune (start ~Sep 28;
   `experts_implementation="batched_mm"` mandatory). Training sampler
   consumes tokenizer/ + factory train_manifests. Demo option: serve
   Layer-A behind a DashScope-realtime-compatible endpoint → qwen-live
   Host/daemon as free demo client (docs/notes-qwen-live.md).
3. Layer-2 alignment toolchain (CANDOR/Fisher/PMQs collector output →
   separation → 80 ms grid) once Layer-A training shape is stable.
4. If Qwen3.5+ Omni ships open weights: candidate Layer-A base swap
   (docs/notes-qwen-live.md watch item).

**Issue #5 v0 DONE (2026-09-19):** `serving/delegation/` — async boundary
(fire() 0.1 ms vs a 5 s brain; non-blocking poll(); pre-warmed session;
<3 s p50 first-chunk budget; ADR-0002 by construction). Backends: Mock
validated; Kimi (OpenAI-compatible SSE, needs key, base_url swappable to
self-hosted vLLM); ACP client spec-correct but live handshake pending
(local claude/codex builds predate --acp). Remaining: live Kimi run,
<delegate> interception in the Layer-A tick loop (#7).

**Issue #6 v0 DONE (2026-09-19):** `evals/harness/` — T-protocol replay
harness; reproduces every bake-off finding from the raw recordings
(MiniCPM: T3 pass / T2 5040ms / T5 14.6s engaged; PersonaPlex: T4 pass /
T5 7.5s; both T1 p50 >2s). Metrics run on the 80 ms grid — same cadence
as the factory gates; this is the Layer-A checkpoint gate. LIVE PATH
ADDED same day: MiniCPMDuplexTarget behind the Target seam (frameHash
chunk telemetry), `run.py --live minicpm` drives + scores in one command
(`evals-live.slurm`) — baselines and live runs share one metrics path.

**Issue #2 v0 DONE (2026-09-19):** `tokenizer/` — 80 ms micro-turn grid +
interleaved serializer + reserved vocab + SPEC.md contract; round-trip and
invariants verified over all 8 local batches (860 scenarios). Upgrades
queued: forced word alignment, BPE-level sampler tokens.

## Repo conventions

- **Communication (user rule, 2026-09-20): plain language always.** Explain
  any technical term or jargon the first time it appears in a reply
  (barge-in = user interrupting; WER = word error rate, how different the
  transcription is from the script; overlay = a folder of extra Python
  packages we prepend so cluster jobs find them; KV cache = the model's
  running memory of the conversation). Keep tables/numbers, label them in
  everyday words. Applies to ALL agents working in this repo.
- Task board: GitHub Issues #1–#7.
- **Voice-sample audit loop (user request, 2026-09-17): anything that
  produces audio must be audible to the user via the localhost gallery.**
  After every render/filter/voice step: rsync the batch + any voice assets
  to local `runs/` (recipes in `viz/listen.py` docstring), make sure
  `python -m viz.listen` is running, and tell the user there is new audio
  at **http://localhost:8765** (they refresh; pages read the synced dirs
  live). The server is a detached nohup process — check
  `pgrep -f viz.listen` and restart on port 8765 if the Mac rebooted.
- `viz/index.html` is the living visualization — every decision updates it
  (viz/GOVERNANCE.md). Repo copy is canonical (local editable copy
  `duplex-live-viz/` does not exist yet).
- Local clone (OneDrive-synced): `~/Library/CloudStorage/OneDrive-Personal/Shared GitHub Repos/duplex-live`
- Isambard clone: `/projects/u6si/duplex-live/repo` — keep both pushed in sync.
- **GATE 2 (DAgger) VERDICT (job 6916784): diagnosis confirmed — mute
  fixed, onsets 16/17 exact at thr 0.5 (bar-passing) — but chatter
  regressed to the other wall (31-77% trained, 52-82% held; bar <10%),
  threshold-flat on dialog lessons. Bar still FAILED.** Round 2
  free-ran TRAIN4 with gate1 active, buffered 2400 on-policy proposal
  states with gold labels, retrained; gate P at gold onsets jumped to
  0.66-1.00 (gate1: <0.35). Gate1+stopgap-rules audition (6916786):
  FULL MUTE, 0/21 exact — rules compound gate1's over-blocking;
  gallery batches model-audio-dense-pt3-gate035 / -gate-combo.
  Composite probe 6922786 (gate2@0.5 + min-gap 8/16 + user-gate
  hold 3) measures the last decode-side lever: gap divides chatter
  ~1:gap, gold run-gaps >=93 ticks so responses can't be clipped.
  Base model proposes words on ~91% of quiet ticks — the speech-rate
  prior itself is the residual problem (gate round 3 w/ iterated
  DAgger is the training-side candidate).
- **GATE2 COMPOSITE VERDICT (job 6922786): decode dial EXHAUSTED.**
  gate2@0.5 + min-gap 8/16 + user-gate fixed chatter as projected
  (held lessons all 5.2-9.7%, under bar) but strangled responses:
  min-gap counts from EVERY accepted word and gold words arrive every
  ~4-5 ticks inside a response, so follow-up words are blocked, the
  fed-back sil diverges the run, exact onsets 2/17 trained / 0/15
  held, recall 0.93 -> 0.19. Four configs now mapped: gate1=mute,
  gate2=chatter, gate2+rules=strangled, rules-alone=compromise. You
  cannot rule your way out of a wrong prior. What's left: training
  (gate round 3, iterated DAgger w/ quiet rebalance) or acceptance
  (M2 on gate2@0.5 + user-gate). Decision brought to owner.
- **M2.2a RESULT (job 6923669): parity property HOLDS on the serving
  decode — paced vs dry picks identical, 0 mismatches on all 4
  TRAIN4 lessons with gate2@0.5 + user-gate in the loop; picks-m2-
  serving saved as the live-loop reference. The speed gate FAILED,
  now precisely measured: quiet ticks p50 55.3 ms (fits 80 ms);
  SPEAKING ticks p50 417 ms, p95 ~430 ms, max ~497 ms (piece loop,
  ~3.3 pieces/tick avg) -> adherence 3.8-10%.** Milestone 2b
  (decisions on time, pieces in arrears) is the validated fix;
  eager/compile kernel bench deferred into the 2b job (job exited
  at arm 1 under set -e; the loop shape changes in 2b anyway).
  Gate round 3 first submission timed out at 2h (12-lesson rollout
  ~12 min/lesson; 9/12 collected, no checkpoint) -> walltime 4h,
  resubmitted.
- **GATE 3 VERDICT (12-shard DAgger, job 6952939, 10,119 states):
  FIVE-POINT LAW — three gate rounds are three operating points on
  ONE frontier (gate1=mute pole, gate2=chatter pole w/ perfect
  placement, gate3=middle: held chatter 21-28% but onset P collapsed
  to 0.01-0.35, exact 0-2/15). Buffer acc 0.869 (not memorizing) and
  still fails -> the trunk state alone cannot separate turn-start
  from chatter in this recipe family. Murmur fix = state coverage =
  the scaling run. Demo config: gate2@0.5 + user-gate. Optional
  round-4 hypothesis: gate input += user-audio energy.**

## 2026-09-30 — scaling run prepared (owner-approved "scaling run + M2")

One command launches the whole chain, cert-lapse-proof (each stage
self-chains the next with sbatch --dependency=afterok):
`sbatch evals/scale1-data.slurm`
1. **scale1-data** (12h): banks (Qwen3-Omni text-only, ADR-clean;
   reused if present) -> 240 dense timelines (120/class, seed 23 —
   dense_dialog+dense_barge ONLY, the classes every probe measured)
   -> MiniCPM render, 2 workers. Timeout-safe: clip cache resumes;
   then chains filter -> shard jobs.
2. **scale1-shard** (12h): prepare over filter-passed lessons;
   resume-safe (skips existing shard .pt files — new). Chains train.
3. **scale1-train** (24h): probe-9 recipe UNCHANGED (v2 labels,
   ww3, SH0) on ~224 lessons, 2000 steps (~9 passes/lesson vs
   probe9's ~100 — same recipe, coverage instead of memorization),
   --save-every 200 (new; tmp+rename atomic). Held-out = idx
   113-120/class, closing free-run on 4 of them. Calibrate STEPS
   vs probe9 elapsed (sacct 6905339) before first submit; resume
   = TRAIN_INIT=<ckpt>.
Trainer changes: train_p1 --save-every; prepare skip-existing.
BEFORE SUBMIT: `! clifton auth` (cert lapsed again), git pull on
cluster, md5sum check, then check walltime caps + probe9 elapsed.

### Research passes (2 background agents, 2026-09-30) — what prior work says

**Serving side (M2):** Moshi's Mimi codec = 80ms causal frames,
zero lookahead, 200ms end-to-end — our 80ms tick is a published,
working design point. Rolling-window incremental encode = standard
cached-left-context streaming; the hard 4s drop-oldest truncation
is OUR untested part (M2.3). Piece bank = "limited-domain
concatenative/unit-selection synthesis", a named 30-year-old
technique; known join artifacts (pitch/phase/energy mismatch) with
cheap mitigations: cut units at pitch epochs + short crossfade.
Barge-in in deployed stacks = cut at output-buffer granularity
(~20-100ms), hard cut with a few-ms gain ramp; no one fades.
**M2 actionables: 2c piece bank should cut at pitch epochs and
crossfade joins; barge-in per-piece check matches practice.**

**Training side (scaling run):** field floor for real duplex
behavior is ~2,000 REAL hours (Fisher; Moshi: 7M hours mono did
NOT buy placement — it needed Fisher + RL). Largest synthetic
push: SyncLLM 212k synthetic hours (helped meaningfulness; timing
still collapsed out-of-domain 0.43 -> 0.23). Our 240 lessons ≈
single-digit AUDIO-hours — tiny in hours, but (a) our trunk is
already fully speech+language-capable (we train ONLY 10 timing
tokens), (b) our lessons are saturated with turn-taking events
(Fisher hours are sparse in them), (c) direction is precedented
(synthetic duplex-format data at scale helps), magnitude untested.
Aux turn-taking heads on frozen trunk states ARE published
(Freeze-Omni: linear 3-state head on hidden states — near-identical
to our gate); our exact failure mode (can't split turn-start from
chatter on those states) is NOT published; nearest analog dGSLM
answered nearly every pause (TOR 0.93). Kyutai's own fix for
timing was RL with timing rewards (interrupt latency 1.38 -> 0.46s)
— the known complement if scaling alone lands short. Behavior-
balanced mixtures are the other lever (our scenario classes ARE
that taxonomy; currently only the 2 dense classes scale).

## 2026-09-30 (later) — SCALE1 CHAIN LAUNCHED + RL plan + ADR amended

**Chain is RUNNING.** Data job 6960233; watcher armed (stage
transitions + terminal states; chain is cert-lapse-proof internally).
Calibration measured before launch: probe9 = 400 steps / 1h05m35s
(~10 s/step) -> 2000 steps ≈ 5.5 h (24 h budget fine); shard
encoder = 12 lessons / 67 s (~5 s/lesson -> ~20 min for 240);
prior render 4.5 min warm; clip cache holds 2,753 clips; workq
MaxTime=UNLIMITED. md5 verified all 7 touched files local==cluster.

**ADR-0003 Amendment A (2026-09-30): closed-content cleared
generally** — enterprise agreement licenses closed-API content for
training; provenance tags stay as hygiene, not a gate. Agent rule:
stop citing public provider terms here (owner: they are not the
current authority). Wave-2 option now open: content banks from the
ChatGPT API (same banks.json format, banks-gpt module = small) —
decide at wave 2; the running chain uses Qwen3-Omni banks.

### RL exploration (owner 2026-09-30: "explore the right policy and
### training method while we have compute")

Two-probe ladder, cheapest first, START AFTER scale1 reads out:
- **RL-probe 1 — head-DAgger (~hours, reuses gate-2/3 machinery).**
  train_gate's DAgger rollout already produces (on-policy state,
  label) pairs; relabel them GOLD for the WORD HEAD and train the
  head rows on that buffer (new small trainer, e.g.
  layera/train_head_dagger.py; save rollout states once to
  runs/dagger-states). Rationale: chatter = the ~91% quiet-word
  prior never corrected on visited states; DAgger gives exposure
  WITHOUT poisoned context (self-history probes failed on corrupt
  inputs; this keeps inputs real, labels gold — exactly what fixed
  placement ranking in gate round 2). Pass bar v2 unchanged: held
  chatter <10% AND exact onsets >=90%.
- **RL-probe 2 — segment GRPO with timing rewards (~1-2 GPU-days to
  first signal; Kyutai's published lever).** Mine short behavior-
  dense segments (60-120 ticks around pauses/interrupts) from
  timelines; K=8 group rollouts per segment; reward = bar-v2 metrics
  (exact onset +, chatter −, mute −); group-normalized advantage;
  update ONLY the 10 new-token rows (+gate). Kyutai precedent:
  interrupt latency 1.38 -> 0.46 s. Cost wall = full-lesson rollouts
  (12 min/lesson) — segments are the fix; batch the K samples.
- Decision rule: probe 1 first. Chatter fixed -> stop. Partial ->
  GRPO. Both fit the allocation before it ends 2026-10-21.

## 2026-10-01 — scale1 verdict (FAIL), chain postmortem, gate1-on-scale1 launched

- **Scale1 readout: the pass bar failed both halves** (5/17 exact
  onsets, false speech 41-75%, one fully-mute held lesson). 33x
  coverage on the unchanged recipe did not break the five-point
  frontier — same trade at scale, per-lesson pole mixing. Full table
  in EXPERIMENTS.md "SCALE1 (the scaling run)".
- **Three chain incidents, all fixed + pushed**: shard --limit 8 bug
  (commit 9cc4213, invalid 8-lesson train cancelled ~2.5 GPU-h in);
    rerender manifest-append duplication (scale1-rerender.slurm);
  dialogroll part-file naming (globbed). Dialog filter collapse is
  SYSTEMATIC (base-rate clip babble; re-rolls +5/round, exhausted).
  Render-side per-clip ASR check = the queued fix if dialog coverage
  matters. Gallery: runs/synth-scale1 synced (240 lessons audible).
- **LAUNCHED: evals/scale1-gate1.slurm** — gate round 1 on the scaled
  trunk (runs/scale1-pt1), UNGATED 8-lesson DAgger rollout (4 dialog +
  4 barge), sweep mirrors gate rounds 1-3, held = the same 4 lessons
  for direct comparison with the ungated verdict.
- **Next after gate1-on-scale1 reads out**: RL-probe 1 head-DAgger
  (train_gate rollout states -> gold labels for the WORD head rows;
  needs layera/train_head_dagger.py written), then RL-probe 2 segment
  GRPO per the plan above. M2 2b (piece-budget scheduling) still the
  parallel track toward the Oct-14 audible goal.

## 2026-10-01 (part 2) — gate1-on-scale1 FAIL, RL plan hardened, probe 1 running, mitigants in place

- **Gate1-on-scale1 verdict: FAIL** (commit 47a52c2). 0/17 held exact;
  gate P at gold onsets bimodal PER LESSON (0.98-1.00 vs 0.02-0.09) =
  lesson-level property. The five-point law EXTENDS to the 33x scaled
  states: the trunk state alone cannot separate turn-start from
  chatter, and a side gate cannot fix it. Full table in EXPERIMENTS.md.
- **RL plan reviewed against research BEFORE compute** (f274f9d,
  docs/notes-rl-timing-research.md): Kyutai's full-duplex RL paper
  (arXiv 2606.11167 — G=16 GRPO, reward-decoupled normalization,
  context curriculum, joint-axis batches, interrupt latency 1.38->0.46s)
  + DAPO failure modes (arXiv 2503.14476). 10 adoptions recorded.
  Critical assessment (owner challenge): Kyutai is a prior, not gospel
  — their RL SHARPENED existing behavior, ours must CREATE it (raw
  passes here range from 1,161 word-proposals/lesson to ZERO); GRPO's
  honest role = FINISHER on dialog (base rates non-zero), starved on
  barge until DAgger injects behavior; the judge-free inversion stands
  (scripted gold makes their LLM judge unnecessary).
- **RL-probe 1 (head-DAgger) RUNNING**: job 7001927 on scale1-pt1,
  8-lesson raw-decode rollout -> train ONLY the 11 duplex head rows
  (layera/train_head_dagger.py, commit 35426b3) -> UNGATED freerun on
  TRAIN4 + the same HELD4 as scale1 (direct comparison). Rollout
  polarization CONFIRMS per-lesson pole mixing live: dialog-001/010
  ~1,100 proposal states (chatter pole), barge-010 = 0 and
  dialog-012 = 2 (mute pole — DAgger gets NO states there: it cannot
  correct trajectories the model never leaves; known blindspot).
  Buffer ~3,600 states, chatter-side dominant. Readout ~2h from 13:20
  local.
- **Mitigants IN PLACE (7f45ef0)**, per the owner's risk review:
  - layera/panel.py — the regression GATE. Parses freerun logs
    (retroactive), reports exact/+-1s/chatter3/R3/full-mute per
    lesson, pairs candidate-vs-baseline; FAIL on exact/+-1s drop
    >5pt, chatter +50 ticks, or a NEW full-mute (baseline mutes are
    the bar's problem). Validated: reproduces gate1's 0/17 held
    verdict from the real log; self-gate passes. EVERY RL readout
    runs --gate-vs <scale1 baseline log>.
  - Fresh set RESERVED: barge-115/117/119 (never eval'd). GAP: no
    fresh DIALOG lessons exist (held dialog collapsed too; only
    116/118 sharded, both selection-burned) — fix rides the
    re-render decision below.
  - Reward arithmetic PINNED (notes #3): full +2.7 vs shortened +1.6
    vs mute -0.5 vs chatter <0; train_grpo.py MUST assert the
    dominations at startup or refuse to launch.
  - Leashes: KL logged every step (abort >10x initial), entropy
    floor (abort <20% initial), DAPO dynamic sampling for
    zero-advantage groups. Staged: pilot (hours, panel-gated) ->
    owner go/no-go -> 1-2 GPU-day run. Gallery audition via
    render_model_audio + viz.listen at every readout. Parity rule:
    any RL checkpoint in the demo config -> M2.2a re-run.
- **Pending owner decisions (bundled at probe-1 readout)**:
  (a) GRPO go/no-go on the panel-gated probe-1 numbers — if probe 1
  stalls, the decision is the LoRA escalation (notes #9), NOT more
  11-row spend; (b) dialog re-render with per-clip ASR check + NEW
  fresh dialog held lessons (fixes the 29/120 filter collapse AND the
  freshness gap; GRPO wants the segment pool either way).
- **Parallel always**: M2 2b (piece-budget scheduling) — the live
  loop is on the critical path to Oct-14 regardless of RL.

## 2026-10-02 (evening) — D1 + UGATE: the two decisive negatives; model-side levers CLOSED

**Plain words — the problem, the finding, the fix.**
*Problem:* the model cannot tell when it is its turn to speak — it
interrupts or it goes silent, and it rarely starts at the right
moment. Every fix so far just trades one mistake for the other.
*What today proved:* (1) the sense of timing was never built inside
the model — detectors looking for it in the model's internal numbers
all scored coin-flip, so "train a small reader" fixes (LoRA included)
can never work; (2) a loudness meter that mutes the model while the
user talks halves interruptions but makes turns late/missed, and on
user-heavy conversations it silences the model completely. *Root
cause:* the training answer key ("model speaks at tick t") comes from
the written script and was never checked against the actual audio,
clip by clip. The model learned broken timing because the answer key
is probably broken. *Solution, in order:* (1) audit the data —
speech-to-text every clip, compare with the script, drop/re-render
bad clips, rebuild the practice set (minutes of compute, no
training); (2) build the M2 live loop (mic in, speech out) in
parallel — that is the path to something audible by ~Oct 14 with the
model we have. *Was the old data prepared well?* Voices yes,
conversation-level checks yes, individual clips and the timing
answer key no — and that is the gap to close first.

Jobs 7019983 (D1 diag, 85s) + 7019984 (ugate direct-path, ~2h), both
COMPLETED after the PYTHONPATH=$DL/pylibs fix (7a2fd08; both new
scripts had dropped pylibs — transformers unresolved, died in ~10s).

**D1 (EXPERIMENTS.md "D1: STATE DIAGNOSIS")**: linear+MLP probes on
the 4,374 saved free-run states, ALL cells at chance (per-lesson,
pooled, AND leave-one-lesson-out; MLP=linear). Verdict: no turn-start
signal in the trunk states at all. Readout training dead in its
strongest form; retro-explains gate-MLP 1-3 + linear head-DAgger +
head-DAgger CE with one cause (the free-run speech schedule has
DECOUPLED from the script clock; the state encodes the model's own
pole, not the script's clock). **LoRA probe-1b CONTRA-INDICATED
(there is nothing for it to read out).** GRPO stays retired.

**UGATE (EXPERIMENTS.md "UGATE")**: scale1-pt1 + mic-energy VAD
user-gate on HELD4 — FAIL 3/4 gates at both holds. Chatter ~halves
(the channel is real) but onsets pay (116: 5/5->2/5 +-1s) and
dense-user barge lessons saturate the within-lesson percentile
threshold -> barge-113 NEW full mute (barge-114 was already mute).
TRAIN4 fine (dialog-001 4/5, barge-004 4/4; dialog-010 is its
pre-existing pole). Third confirmation of the same conclusion:
**the mute<->chatter frontier is a property of the TASK (script-clock
gold over never-per-clip-verified audio) + a model that cannot
represent the script clock — no mechanism on top moves it.**

**State of the ladder**: recipes (5-point law at 33x), readouts (D1:
chance), exogenous gating (ugate: same frontier) — ALL CLOSED with
data. Remaining levers are objective-level only:
- A. Label-quality audit (near-free): per-clip ASR vs script on the
  rendered assistant clips (clip granularity version of the filter's
  lesson-WER), then the pending dialog re-render + FRESH held
  dialogs. D1's decoupling is exactly what inconsistent gold predicts.
- B. M2 live loop = critical path to audible-by-Oct-14: causal VAD
  (absolute or running floor, NOT per-lesson percentiles — ugate
  showed why), 2b piece-budget scheduling (the 417ms wall),
  parity-gated promotion. The demo composite already ran end-to-end.
- 4% ear-clock skew (52 ear tokens vs 50 ticks per 4s) is a known
  wart of the right size (~150ms over 30s history) to smear onset
  information — fix belongs in the M2 live loop regardless.
- Audio of both ugate arms: job 7020187 -> runs/model-audio-ugate-
  {held,train}; rsync + viz.listen before believing the FAIL.
- Fresh-set rule unchanged: barge-115/117/119 NEVER eval'd before
  final readouts.

## 2026-10-02 (later) — FAIL-CLOSED DATA RULE implemented in code + the audit numbers

**Plain TL;DR:** You ordered the training data fixed so bad data can
never silently proceed again. It is now code, not promises: every clip
is checked while it is spoken and re-spoken if bad; every conversation
gets a quality record; the word-timing answer key gets audited against
the real audio; and all eight training/eval entry points REFUSE to
start unless every lesson they load has a passing record. On the old
batch the new audit shows: only 127/240 conversations would qualify
(dialogs 19/120 — the TTS "tail babble" failure was the killer), and
39 dropped conversations had exactly one bad clip — exactly what
repair-in-render recovers. A re-render of the same timelines with the
judge on is queued (scale1-quality.slurm -> filter -> timing audit ->
stamp), and it deliberately STOPS before training.

**Jobs:** none running for this yet at entry-writing time (chain about
to be submitted; 2b bench 7026771 still running separately — bench is
engineering-only, unaffected by the gate).

**Where the code lives:**
- `data/synthetic/render.py` — repair-in-render (--repair N, default
  2; REJECTED.json per rejected conversation; cache = passing clips
  only, re-judged on reuse).
- `data/synthetic/timing_audit.py` — whisper word timestamps vs grid
  placement (assistant clips only — user words are input audio, not
  answer-key labels; input timing is correct by construction).
- `data/synthetic/quality.py` — quality.jsonl stamp + assert_verified /
  checked_rows; schema dl-quality-v1; per-clip bar
  filter.PER_CLIP_WER_BAR = 0.15 (single definition in filter.py,
  shared by render-time judge, batch filter, and stamp).
- Consumers wired: layera/{prepare,train_p1,train_gate,
  train_head_dagger,freerun_diag,probe_p1,contract_trace}.py + m2/
  parity.py — checked_rows/abort on unverified lessons; train_p1 and
  train_gate also refuse stale shards.

**Audit numbers (runs/synth-scale1, 240 conversations):**
- Old filter: 141 passed / 99 dropped. dense_barge 117/120,
  dense_dialog 24/120 (20%).
- Drop causes: tail_babble x192 (TTS adds non-script words),
  cut_misplaced x2, babble x2.
- Bad clips per dropped conversation: 1 -> 39 convs, 2 -> 31,
  3 -> 17, 4 -> 8, 5 -> 2.
- Passed lessons under the strict per-clip bar (0.15): 127/240
  (barge 108/120, dialog 19/120); 94 of the 141 "passing" ones carried
  over-bar clips (max clip WER 0.80).
- Over-bar clips are mostly long assistant clips (77 of 105 have 7+
  words) — real garbling, not a short-clip ASR artifact.

**What happens next (in order):**
1. scale1-quality chain: re-render (judge on, cache carries) ->
   filter -> timing audit -> stamp. READ quality.jsonl when it lands.
2. If verified yield is healthy: re-shard the NEW batch (prepare now
   gates on the stamp) and retrain p1 — the first training run ever
   on fully verified data. Old scale1 batch is dead for training.
3. M2 track continues in parallel on its own discipline: 2b bench
   readout (job 7026771), causal VAD, parity re-run — verdicts only
   on verified keys once the new batch exists.
4. Ear-clock fix (52-vs-50) still owed in the live loop.

**Fresh-set rule unchanged:** barge-115/117/119 NEVER eval'd before
final readouts.

## 2026-10-03 — 2b bench verdict + timing RE-LABEL implemented

**Plain TL;DR:** Two results. (1) The 80 ms live loop's DECIDE half
fits real time — 59 ms per decision tick with the batched_mm kernel
(eager kernel: 165 ms, eliminated). The VOICE half is the wall: dense
speech needs ~5x more compute per tick than the budget on this
checkpoint; parity is perfect everywhere (the scheduler never changes
what the model says). (2) The timing answer key is now RE-LABELED from
measured audio: the timing audit writes whisper-measured word onsets
into the timeline (word_onsets_ms), the grid builder uses them instead
of the char-proportional estimate, and the audit re-measures to
confirm. Root causes fixed along the way: TTS clips carried multi-
second silence tails (trimmed at render now — a 5.0 s clip held 2.3 s
of speech), and the audit itself had a 24k-vs-16k sampling-rate bug.

**2b bench numbers (job 7026771, 5 arms, TRAIN4):**
- decision-only ticks p50: batched_mm 59 ms / eager 165 ms — kernel
  choice settled (batched_mm).
- speaking ticks on dense_dialog-001: best 386 ms p50 (arrears ppt4
  batched_mm) vs 80 ms budget; backlog max ~1300 pieces; word density
  ~6.6/tick vs ~80 ms/word compute. Voicing dense turns in real time
  needs spec-decode or sparser turns — decision for the demo plan.
- arrears scheduler parity: 0 mismatches in all 5 arms — promote.

**Fail-closed chain state:** smoke proved render-judge -> filter ->
timing audit -> stamp -> consumer gate end to end (gate correctly
HELD on unverified data). The --relabel smoke came back GREEN (barge-004: all 4 assistant
clips re-labeled, p95 offset 40 ms, stamp 1/1 VERIFIED, consumer gate
OPEN) and the full scale1 re-render chain is LAUNCHED (job 7032050 ->
filter -> timing+relabel+stamp, 240 conversations, --repair 2). It
stops before training: re-shard + retrain wait for the stamp and the
owner's go.

**Known render behaviour (fail-closed working as designed):** some
TTS texts babble persistently across retries (dialog-000 clip
15_user: 8 extra words twice; --repair 2 in the real chain gives one
more dice roll, then the conversation is REJECTED with evidence).

## 2026-10-03 (later) — OWNER APPROVAL: 5x compute for the voice wall, on Isambard

**Plain words:** The 2b bench showed the live loop can DECIDE inside the
80 ms tick but VOICING word-dense turns costs ~5x the compute budget on
one GPU. The owner approved spending that compute ("approved on the 5x
compute needed, use isambard"). Two stacked levers, because the
measured ~80 ms per word token is ~10x worse than this hardware should
serve a 3B-active-param model — the gap is per-call overhead, not maths:
1. torch.compile on the decode path (parity.py --compile existed but
   was never benched);
2. sharding the model across 4 GH200s.
Job 7032343 (evals/m2-voice5x.slurm) measures compile@1gpu, shard4+
compile, shard4-plain against the recorded baseline (386 ms p50
speaking tick, backlog ~1300, dense_dialog-001). Parity must stay 0.

## 2026-10-03 (evening) — FIRST VERIFIED CORPUS LANDED; training started; two bugs the gate caught

**Plain TL;DR:** The re-render chain finished: **153 of 184 rendered
conversations are fully VERIFIED** (barge 115/120, dialog 38/64) —
every clip within the 0.15 word-error bar, timing labels RE-WRITTEN
from measured audio (p95 40 ms after re-label). The old batch is dead:
its 141 "passes" included 94 with garbled clips and zero verified
timing. Sharding the 153 took 3 minutes; training (probe-9 recipe,
unchanged) is running as job 7034470 after two startup failures the
fail-closed design CAUGHT (that is it working):
1. manifest alignment bug — the stamp's verified flag was read before
   the timing fold-in, so 2 timing-FAILED lessons stayed in
   train_manifest; the shard job's startup assert stopped on them.
   Fixed (verified flag now read after the fold); manifest = exactly
   the verified 153, backup in train_manifest.all.jsonl.
2. CheckpointError on training step 1 — gradient checkpointing's
   strict recompute check trips when the MoE router flips a bf16 tie
   between forward and recompute (per-expert counts 485 vs 484).
   Fixed with use_reentrant=True.

**Voice-wall attempt 1 verdict (owner-approved 5x compute):**
- shard4-plain: 489 ms p50 speaking tick; shard4+compile: 785 ms —
  BOTH WORSE than one GPU (386 ms). Pipeline-sharding across GPUs adds
  per-token interconnect hops; wrong tool for latency-bound decode.
- compile@1gpu: OOM'd (compile buffers + 30B model > 95 GB; heavy
  fragmentation). Retry running (job 7034277) with
  expandable_segments allocator.
- Parity 0 everywhere; decision ticks 59 ms (1 GPU) remain the
  proven floor.

**Files:** runs/synth-scale1-q/audio (quality.jsonl, timing_report.jsonl,
train_manifest.jsonl = 153); runs/shards-scale1-q (153 shards);
evals/scale1q-{shard,train}.slurm; evals/m2-compile-retry.slurm.

**Next:** read train job 7034470 (free-run diagnostic on held-out
verified lessons runs at its end); compile-retry verdict; then the
cheap re-pass over the 56 render-rejected conversations (clip cache
makes it cheap) to top up dialog count.

## 2026-10-03 (night) — FINAL corpus: 163 verified; sampled re-speaks; corpus frozen

**Plain TL;DR:** Two recovery passes later the verified corpus is FROZEN
at **163 conversations** (barge 115, dialog 48) — every clip within the
0.15 word-error bar, every timing label measured from audio. Recovery
mechanics learned: greedy TTS is near-deterministic, so identical
retries re-babble the same words (pass 1: 10/56 recovered); sampled
re-speaks with rising temperature (attempt>0) do better (pass 2:
11/45), but ~45 texts babble under any dice — diminishing returns,
stopped there. The batch is self-consistent: train_manifest.jsonl =
exactly the 163 verified (re-stamped after every pass).
NOTE: training job 7035090 trains on the 153 shards built BEFORE the
recoveries — the 10 extra lessons are a top-up for any FUTURE run, not
this one (restarting for +6% data mid-run buys nothing).

Training: step 1608/2000, loss 0.63 — the free-run diagnostic on
held-out verified lessons runs automatically at the end (~1h).

## 2026-10-03 (late) — scale1-q training verdict: data was A reason, not THE reason

**Plain TL;DR:** First training run on fully verified data (153
lessons, recipe unchanged, job 7035090, loss 4.15 -> 0.60). Held-out
free-run: dialogs still near-mute (word recall 0.14-0.19, onsets
0-1/5); barges moderate-to-decent (speech F1 0.53-0.60, word acc
0.36-0.41, onsets 2-4/4 within 1 s). The ear-clock skew (52-vs-50,
printed by the diagnostic) is still unfixed and confounds onsets —
this is a LOWER bound. Data quality is retired as the suspect.

**Next in order (per the standing plan):**
1. EAR-CLOCK FIX (sampler.py assumes 50 ear tokens / 4 s; the tower
   emits 52 — resample or tick-align), then one more training round —
   time for exactly one more full round before the allocation ends.
2. Gate round on the scale1-q trunk (train_gate) in parallel — the
   demonstrated frontier mover; no retrain needed.
3. M2 audible demo with the scale1-q checkpoint (causal VAD + arrears
   + batched_mm; decisions fit 59 ms/tick; voice-dense turns lag).
Checkpoint: $DL/runs/scale1-q-pt1/p1_checkpoint.pt.

## 2026-10-04 — EAR-CLOCK FIX + blind-spot sweep (owner: "leave no blind spots")

**Plain words:** The last known fault in what the model HEARS is being
fixed: its ear produced 52 sound-snapshots per 4 seconds while the
clock ticks 50 times — the extra 2 were silently thrown away each
window, so everything the model heard drifted 4% late through every
conversation. The fix resamples the ear onto the clock grid in the one
place all consumers share. Retrain (one variable changed: the ear
alignment; corpus = all 163 verified) chains after the smoke test.

**Blind-spot sweep — every confound either CLOSED or tracked:**
1. ear-clock 52-vs-50 — CLOSED (encode_audio resamples; smoke job
   7039827 asserts feats == ticks; q2 shards rebuilt from scratch —
   the old shards carry the skew and are dead).
2. clip content — CLOSED (per-clip 0.15 bar, repair-in-render,
   fail-closed asserts in all 8 consumers).
3. timing labels — CLOSED (measured onsets; relabel+re-audit).
4. stale shards — CLOSED (train aborts on unverified; q2 chain
   re-encodes).
5. held-out freshness — intact (113-120 held; 115/117/119 NEVER
   evaluated — final-readout reserve).
6. class imbalance — TRACKED, deliberate this round: 115 barge vs
   48 dialog, and dialogs are the weak class. One variable per round
   (this round: ear alignment). NEXT ROUND's lever: oversample
   dialogs in the training list.
7. <speak> never predicted in free-run — TRACKED, pre-existing across
   every checkpoint; argmax is sil-dominated. The sil_bias sweep in
   freerun_diag finds the operating point — RUN it on the q2
   diagnostic instead of argmax-only.
8. self-history compounding — TRACKED: the free-run diagnostic is the
   hardest setting. Also run contract_trace on the q2 checkpoint
   (training-stream vs live-stream parity — isolates label-following
   from compounding).
9. timing-audit circularity — RECORDED: post-relabel, the audit
   re-measures with the same whisper that wrote the labels, so
   "timing ok" is by construction; the real anchor is whisper's
   timestamp accuracy (~50-100 ms) vs the 80 ms grid. Acceptable,
   known, written down.
10. whisper-as-judge bias — RECORDED: a systematically misheard word
    fails SAFE (rejection), never passes bad clips.

**Running:** ear smoke 7039827; gate round on the verified trunk
(scale1q-gate1, job 7039828 — instrument continuity with every prior
gate round). On smoke green: scale1q2 shard -> train chain (163
lessons, ear aligned, end-of-job held-out diagnostic directly
comparable to last night's).

## 2026-10-04 (later) — EAR-CLOCK VERDICT: hearing was the wall

**Plain TL;DR:** One variable changed (ear resampled 52->50 onto the
tick grid; shards rebuilt; same 159 verified lessons, same recipe).
Held-out free-run, before -> after: dialog-116 recall 0.14 -> 0.93
with 5/5 exact onsets; barge-113 0.48 -> 0.89; barge-114 F1 0.53 ->
0.70. The timing sense was in the model all along — it could not hear
the clock. New state: over-speaking (low precision) — the gate round
(job 7039828, COMPLETED, log unread: Clifton cert lapsed again mid-
read) is the operating-point tool. NEXT: read gate log after `!
clifton auth`, then pick the operating point and build the audible
demo composite (causal VAD + arrears + batched_mm) on the q2
checkpoint ($DL/runs/scale1-q2-pt1/p1_checkpoint.pt).

## 2026-10-04 (night) — operating point picked: gate thr=0.10; demo audio rendering

**Plain TL;DR:** The gate threshold sweep on TRAIN4 (protocol: tune on
burned lessons, ONE held-out shot) says thr=0.10 is the only setting
where EVERY lesson works: v3 F1 0.71-0.80 across all four, dialog
chatter cut 22-47% vs ungated, barges at or near their best. Higher
thresholds trade one lesson for another (thr=0.35: dialog-001 F1 0.79
but dialog-010 COLLAPSES to 0.02 — the gate never fires there). The
gate is a mild prior, not a hard blocker. Held-out confirmation at
thr=0.10 + factory-TTS render of the picks is running (job 7063645) —
audio lands in $DL/runs/demo-q2/audio for rsync + viz.listen.

**Also this session:** q2 live-loop smoke COMPLETED (parity 0 on all
TRAIN4 lessons, decision ticks 57-62 ms); the q-gate round proved the
instrument but was paired with the wrong (skewed-ear) trunk — q2 gate
retrained (job 7060708).

## 2026-10-05 — the listen verdict: timing real, content degenerate; honest demo rebuilt

**Plain TL;DR:** Owner audition caught the groan: free-run picks are
93% digit-soup ("0","9",".") — TTS read it as a drone. Corrected the
record: ALL "word acc" numbers in diagnostics are PLACEMENT (started
a word at the gold tick), not content. Timing holds (ear-clock fix);
content was never trained (teacher-forced phase 1). Gallery now lists
model.wav. The honest demo (model onsets + script words, missed turns
= gaps) is building (job 7113366 -> runs/demo-q2/audio-timing).
Word-content generation remains the Talker's job — out of trunk scope;
for the Oct-14 demo, script words at model onsets is the claim we can
defend.

## 2026-10-06 — SIMULATED-LIVE LOOP RUNS END TO END (job 7132354, 21 min)

**Plain TL;DR:** The dress rehearsal works. One program: user audio
streams in 80 ms ticks -> ear windows on the tick grid -> q2 checkpoint
+ gate decide every tick -> the turns the model starts are voiced with
whole-utterance clips (natural prosody) -> yields and user barge-ins
cut playback with a fade. Results on the held-out four: dialog-116
7/7 turns, dialog-114 7/7 (the weak lesson — the gate-grace window for
quick replies WORKS), barge-113 2/3, barge-114 2/4. Causal floor VAD,
no per-lesson percentiles. Gallery: live-demo batch next to
timing-demo. Remaining to the mic demo: bridge real mic/speaker over
SSH (audio pipes) + realtime pacing (--realtime exists, unpaced for
speed in this run). Code: m2/live.py.

## 2026-10-06 (night) — MIC DEMO READY (pipe test passed, job 7135100)

**Plain TL;DR:** The stdin/stdout streaming legs are proven: a 109.6 s
mic-format stream through `cat raw | m2.live --source stdin --sink
stdout` came back as 109.6 s of clean model audio, 7/7 turns taken IN
REAL TIME, logs confined to stderr. The owner-facing demo is
`demo/mic-demo.sh` (sox mic | ssh srun | sox speaker; needs `brew
install sox`; LESSON=dense_dialog-116 by default; Ctrl-C ends).
Known demo behaviours for the ear: answers are the scripted bank
(words out of scope this phase); every user turn-in cuts the answer
(yield) — in the dense lessons the user does come back quickly, so
cuts are often correct, but the yield trigger may be hair-trigger;
turn-taking is the claim being demonstrated.

## 2026-10-07 — maintenance window + the API volume lever PROVEN

**Plain TL;DR:** A 2-day cluster reservation (Oct 6-8) froze all GPU
jobs (research run + open-engine top-up queued, auto-fire at window
end). The ElevenLabs arm ran on the LOGIN node instead (pure API):
**120/120 dialogs shipped, ZERO render rejections** vs the open TTS's
~60% dialog rejection. Cost ≈ $40 of creator credits. Verification
chain (filter -> timing+relabel -> stamp) pre-queued to fire when GPUs
return. Bugs fixed en route: SIGPIPE+pipefail killed the launcher
script (ls|head closes the pipe -> exit 141 -> silent abort; replaced
with a counting loop), a doubled container path, and a missing
module-level librosa. OpenAI key rejected as invalid (ends V2jC) —
banks provider built and waiting for a working key.

## 2026-10-07 (later) — the outage pipeline: local Mac data factory

**Plain TL;DR:** Isambard went down until 22:00 BST mid-push. The data
factory moved to this Mac: torch made optional in render.py, a local
venv (.venv-api) runs the whole API path. NEW OpenAI key VALIDATED —
GPT-4o wrote frontier banks (banks-openai.json), 130 fresh dialog
timelines from them, and 130 ElevenLabs dialogs rendered locally
(100% ship rate, zero GPU). Total API corpus now 250 dialogs (120 on
cluster + 130 local, ~$80 credits total). Auto-handoff armed: on
cluster return — deploy key (bizhleg0c), push local batch + queue its
filter->timing->stamp chain (bkikgqq6z). Then: shard all verified
corpora, launch the self-history training round.

## 2026-10-09 — outage week: science settled, fleet broken, auto-resume armed

**Plain TL;DR:** All questions that could be answered without GPUs are
answered. DECODE FIXES RULED OUT (hygiene arms all WORSE than argmax:
WER 3.6-6.7 vs 0.9) — the digit soup is exposure bias, the DAgger
round is the one remaining fix, rollouts trimmed (4 lessons, 24h).
Corpus: ~1,090 conversations rendered (730 API incl. GPT-5 banks,
300 open top-up, 159 verified in use) — verification chains written.
THE BLOCKER: Isambard fleet-wide GPU driver failure since Oct 8
(Error 803 on every probe: 0/24, 0/8, 0/2; ~85 nodes drained).
Auto-resume loop re-armed: probes every 20 min, first healthy node
submits research run + all verification chains automatically.
Survey committed at docs/research/2026-10-08-duplex-approach-survey.md
(8 ranked recs; DuplexChat 282kh open corpus = the synthetic-ceiling
breaker for VAP/gate training, next after the DAgger round).
Clock: goal Oct 20, allocation ends Oct 21 — fleet repair is the
entire remaining risk.

## 2026-10-11 (later) — full-burn plan AMENDED (owner): core problems over RL

**Plain TL;DR:** Owner amendment: "do RL only if it makes sense — it
reinforces good outcomes that already exist. If we can use the compute
on jobs that fix the core problems, do that instead." Amended plan:
RL demoted to conditional polish (only if arm B lands partial content
AND integration is secure). Spare compute goes to the three core
fixes: (1) content — arm B verdict imminent; (2) delegation+streaming
corpus — chain running (job 7257640, 100 lessons); (3) THE SERVING
WALL — reactivated: raw-forward micro-bench launched (7260758) to
decide between CUDA-graph fix (overhead-bound) vs speculative decode
(kernel-bound). Delegation-answer recorded: data with the right shape
(delegation/timed_task classes, staged onsets) + mixed round + loop
logic = how streaming-in becomes part of the model.


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
