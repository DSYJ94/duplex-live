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
