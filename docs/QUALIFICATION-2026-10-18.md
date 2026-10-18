# Qualification record — 18 October 2026

Plain words: No native duplex model is qualified. The improved hybrid is a separate
experimental reference, with a completed first fresh conversation evaluation. Neither
training mechanics nor clear speech establish reliable task completion. This record
keeps those claims separate and leaves missing evidence explicit.

## Native model: measured improvements, failed qualification

Plain words: The latest balanced-spoken Context Spanning experiment, checkpoint
7510331, completed 520 updates using verified TRAIN supervision. Frozen-weight
checks, exact fresh-process reload and optimizer/random-state restoration pass.
Rechecking the locally preserved checkpoint, training state, data and voice hashes
confirms those artifacts match their reports. All 16 native development WAVs also
match both the original model reports and independent speech-audit 7510355.
These are reused development scenes, not fresh holdouts.

| Requirement | Evidence and result |
|---|---|
| Ordinary answers | Trained lightning, evaporation and day/night replies improve. However, 7510342 says a triangle has “3, 4, or 5” sides; 7510350 says pencil lead is made of lead. Both mistakes are independently transcribed with zero word error and valid timing. FAIL. |
| Private-result use | Actual read receipts and sampled replies exist, including changed private worlds. Some values are spoken, but omitted/contradictory results, internal metadata narration and repetitive output remain. No whole-task qualification. |
| Waiting and foreground work | Actual pending states, backend arrivals and concurrent speech are recorded. Correct waiting cannot be claimed end to end while ordinary answers and subsequent actions fail. UNQUALIFIED. |
| Clarification | TRAIN case 7510343 repeats the clarification and drops part of the actual value. Validation 7510351 produces no actual private read. FAIL. |
| Cancellation | Neither native cancellation scene, 7510344 or 7510352, executes an actual cancellation receipt. A spoken cancellation phrase would not satisfy this requirement. FAIL. |
| Speech intelligibility | Independent timing bounds valid for 15/16 captures; clear speech can still be factually wrong. Mixed speech/task outcome. Owner approval of this voice backbone does not approve these task results. |
| Latency and continuity | Per-case p95 frame compute ranges 36.6–44.5 ms per 80 ms input frame. Maximum input backlog ranges 2.83–12.12 s; nine cases have causal-playback underrun samples. These are simulated playout measures, not microphone-to-ear or room AEC acceptance. |
| Reproducibility | Adapter, optimizer/random state, data, voice and fresh-reload evidence are preserved and hash-checked locally. PASS for preservation, not model quality. |

Plain words: The matching proof and replayable verifier are
`runs/recovery-continuation-20261016/native-balanced-qualification-integrity.json`
and `verify-native-balanced-qualification.py`. The proof inventories all 16 cases,
receipts, transcripts, timing and source-report/audio hashes. Native results retain
an external tool-choice controller; native retrieval-marker decisions must not be
misrepresented as complete native tool selection.

Plain words: Earlier native attempts are retained in EXPERIMENTS.md and the export.
The current failed checkpoint is not promoted. No unchanged extension of this run
is justified by its lower supervised loss or correct answers on some TRAIN scenes.

## Frozen hybrid: first fresh evaluation complete

Plain words: The reference uses frozen Kyutai speech, frozen local Qwen3.5-9B and
actual GPT-6 Astra reasoning when selected. No learned output-head adapter is loaded.
Its verified continuation and cancellation fixes are runtime changes. It is not
native duplex pretraining or Qwen-Omni fine-tuning.

Plain words: Source f09baf0 records the 11 new conversations and their requirements
before audio generation. Every inference-code hash matches the earlier demo source
7915b1c. Input job 7520271 generates only synthetic user speech; independent quality
7520272 applies the unchanged 5% word-error and timestamp gates. Ten conversations
pass. The conversational-memory case 056 fails the lexical gate because of
favourite/favorite spelling differences; it remains unscored and is not rerolled.
This is a limitation of that gate, not evidence that its audio is broken.

| Case | Required behavior | Actual service / independent speech audit |
|---|---|---|
| 047 | Local 19 × 24 = 456; no external calls | 7520283 / 7520284 |
| 048 | Five bags of six minus seven = 23 | 7520285 / 7520286 |
| 049 | Ask for missing appointment information; no invented time/access | 7520287 / 7520288 |
| 050 | Do not claim an unsent dentist message was sent | 7520289 / 7520290 |
| 051 | Stop after a real audible prefix, without a new reply | 7520291 / 7520292 |
| 052 | Retire the long explanation and give only its main idea | 7520293 / 7520294 |
| 053 | Continue the same paused explanation without repetition | 7520295 / 7520296 |
| 054 | Correct counting result plus ordinary foreground answer | 7520297 / 7520298 |
| 055 | Real reasoning-delivery cancellation, same foreground explanation | 7520299 / 7520300 |
| 056 | Retain corrected favourite fruit across three turns | Input gate rejected; unscored |
| 057 | Admit lack of private records; invent no read or code | 7520301 / 7520302 |

Plain words: Correct local reasoning in 054 would not establish delegation coverage.
If the real reasoning result or foreground explanation finishes before 055's
interruption, that timing-dependent capability is uncovered, not a cancellation
success. Case 057 checks honesty about unsupported access; the hybrid has no
private-record tool, so it cannot pass native private-result qualification.

Plain words: All ten eligible services and ten independent speech audits completed.
The original user PCM and every transmitted model packet match their saved sources.
All ten production-playback JavaScript replays preserve wire order without duplicate
source samples. These are simulated AudioContexts, not physical-browser evidence.
Proof: `runs/recovery-continuation-20261016/frozen-conversation-integrity.json` and
`verify-frozen-conversation-integrity.py`.

| Metric | Observed result |
|---|---|
| Ordinary numerical answers | Spoken456 and23; no external requests in either case. |
| Missing information / false completion | Appointment answer admits missing information and asks for details. Dentist answer correctly says no message was sent. No action receipt is fabricated. The appointment reply also suggests granting calendar access, which this profile does not currently implement. |
| Stop / shorten / continue | Genuine stop is silent after its heard prefix. Only the requested refrigerator summary completes. Root explanation resumes the same generator; source-sample replay shows no duplication. |
| Actual delegation and waiting | Case054 sends the actual recognized question unchanged to Astra, returns the correct CPU-oracle count5 and completes the ordinary sea-salt explanation while that result is owed. |
| Actual cancellation | Case055 obtains the actual oracle count4, then cancels its still-pending delivery. The real cancellation receipt targets only turn1; the same soap generator resumes and no count result is spoken. Remote execution is not aborted. |
| Unsupported private access | No code or private-read receipt is invented. However, the one recorded user request is endpointed into two clauses and the assistant speaks the same refusal twice. FAIL for repeated work; privacy honesty alone is not full conversation success. |
| Speech | Eleven completed output clips: nine transcribe exactly, appointment WER0.0435, soap WER0.1136; all have valid bounds. Soap differences include “micelles” transcribed as “my cells” and plural/verb changes. Preserve this discrepancy; no human speech-quality conclusion follows. Six partial clips have no full-answer WER. |
| Decision latency | Ordinary first-turn controller completion is about1.10–3.55seconds after semantic dispatch. These exclude endpointing and final browser/device playback. Speech still waits for full text generation. |
| Coverage | Ten eligible conversations, one unscored memory case. No general agent actions, private-record execution, physical microphone/room AEC or overall native qualification. |

Plain words: A narrow follow-up diagnoses the duplicate refusal. Before the second
user clause is dispatched, the first validated controller answer is already queued
but has not started speech. The pending-speech state contains its text; chronological
assistant history does not. Source-only probe `qwen35_ready_history_probe.py` compares
the exact captured original native proposal with one additional actual queued reply
in chronological history. It changes no weights, dispatches no actions and makes no
external request. Actual diagnostic7520544 reproduces the baseline exactly; adding
the actual queued answer changes the second reply wording but still consumes no
pending speech ID. Reject this as a duplicate-prevention fix; no runtime adoption. The earlier ten
conversations remain frozen development evidence for any subsequent change.

Plain words: Physical microphone, browser-engine playback, room echo/double-talk
remain unverified. The owner accepts the current recorded voice, noting a slight
initial-response delay. Cloud validation
remains paused. The overall goal remains active.

## Next bounded training ablation

Plain words: The previous adapter was trained on every final-call token, but its
less harmful inference variant corrected only function-name tokens. That is a
measurable objective/deployment mismatch. New opt-in function-name-only training
uses the same216 verified TRAIN rows,72 heldout families,rank8,learning rate and
648 fixed updates. Only causal target positions where the deployed hook actually
applies its correction receive loss; argument/content projection stays frozen.
The explicit plan is `evals/fixtures/function-name-training-plan-20261018.json`.
This tests that mismatch without adding any evaluation recording to TRAIN.

Plain words: This corpus lacks positive hard-delegation and the newer resume actions.
Even better72-case scores cannot authorize deployment. Broader reasoning/delegation
and retained-generator checks must pass first. This is a text-controller experiment,
not native voice-model training. Cloud validation remains paused.

Plain words: Initial launch7520602 stops before model loading or updates because
current controller files differ from the approved data compiler. The guard is
unchanged. An isolated experiment checkout restores exactly the two original
compiler files from producerf2e22fb. All216 teacher prompt/action replays and all72
original frozen-baseline rows pass locally with exact source hashes. Experiment
branch `experiments/function-name-loss-20261018`, commit710b305, retains the new
loss-mask code while preserving the original compiler/baseline. Those historical
controller files must not be merged into the current live implementation. This is
a matched training ablation; integration with newer runtime actions remains separate.

## Owner feedback and live connection

Plain words: The owner accepts the recorded hybrid voice, with a slight delay from
initial prompt to response. That listening approval is distinct from a successful
live microphone/browser trial. The owner had been unable to open the temporary demo:
the voice job7519401 was still running, but its localhost8779 SSH forward had exited
with a server-not-responding timeout. Reconnecting the forward restores HTTP200 for
the page and playback scripts and verifies the private session-access binding without
printing its token. Demo remains http://localhost:8767; refresh before Start conversation.
A local watcher monitors the current transport and retries SSH after it exits; it
starts no microphone/model sessions or new compute jobs. The cluster service remains
temporary, subject to its original idle and Slurm wall limits.

## Latest result: session recovery and matched training verdict

Plain words: Repaired demo7520776/source76174d1 now serves the same public link.
An explicit authenticated Start ends the old connection and waits for its media
actor to stop before claiming ownership. A disconnect aborts the actor rather than
waiting on synthetic tail frames or an idle sender. Invalid credentials cannot
replace an active session. End works during startup; late microphone permission
cannot leave an orphan capture. Completion releases the controls. The page shows
whether microphone frames contain sound or quiet audio, and hides the automatic
access code under Connection access.

Plain words: Three actual localhost WebSocket tests, four frame/media-abort tests
and seven Node playback/client-lifecycle checks pass. An actual GPU-service test
opens two zero-audio synthetic connections: the second replaces the first and
reaches session_ready in0.879seconds. No microphone, owner audio or reasoning API
request is used. Exact served page/script hashes and private access binding pass.
Synthetic speech regression7520777/audit778 retains the original user PCM, native
words/actions and same-generator continuation. Production wire replay has no
repeated/discarded source samples; complete independent WER0.02128 with valid
bounds. This is not physical browser/device acceptance. The owner has been asked
whether the refreshed page shows Listening and sound detected while speaking.

Plain words: Matched function-name-only training7520623 completes648updates and
strict adapter/optimizer/random-state restoration; published parameter hashes remain
unchanged. All60 original baseline passes remain, with64/72 overall. All72 canonical
results and verdicts equal the earlier function-scope inference variant7517798,
including the wrong49→50 replacement answer. The checkpoint and full records are
preserved, but this ablation shows no behavioral gain. Do not promote or extend it
unchanged. Proof: `function-name-training-integrity.json` in the continuation folder.
No learned adapter is loaded in the repaired demo.

## Additional native latency evidence and controlled execution experiment

Plain words: The post-run native timing inventory binds all16 model reports,
user stems, event logs and generated WAV hashes to the original user data. The
original diagnostic user WAVs are PCM16; the verifier preserves their measured
quantization error rather than claiming bit-exact float32 capture. Only original
user boundaries are read after inference. On the six single-question scenes,
first sustained recorded acoustic activity is4.07–4.52seconds after the last
aligned user word. This activity threshold is two adjacent10ms RMS windows above
0.003; it is not proof of intelligible or correct answer onset, physical hearing
or microphone-to-ear latency. Fast per-frame compute does not establish fast
response. Artifact: `native-latency-inventory.json` and `verify-native-latency.py`.

Plain words: A separate process-local controller latency experiment compiles only
stateless decoder RMS normalization for sequence length1; prefill, attention,
recurrent state/cache, weights and reply validation stay unchanged. Whole-model
compilation and CUDA graphs are disabled. The live demo is unchanged. Strict
historical replays7520919/7520964 stop before installing the candidate: the frozen
refrigerator answer changes air/interior wording despite the same925 input and85
output tokens. Historical failures are retained. The second run saves the observed
output and individual comparison fields; no speed or quality gain is inferred.

Plain words: The explicit follow-up plan compares exact same-process controls.
All five fixed saved requests still run twice before any change. Their original
repeats must match each other exactly; input length and every non-content action
identity must match the historical capture. The compiled path must then match
those actual same-process outputs exactly on both repeats. All historical wording
changes remain reported. At least1.2median speed ratio on the five second-repeat
pairs, unchanged parameter hashes/versions and actual graph execution are required
before even a voice regression. No deployment follows from a benchmark alone.
Plan: `evals/fixtures/decode-normalization-control-plan-20261018.json`.

Plain words: Same-process normalization run7521006 completes all20 proposals,
with unchanged parameter hashes/versions and one actual compiled graph. The
refrigerator candidate changes words on both repeats. Four eligible second-repeat
speed ratios have median0.99556; the required fifth exact pair is absent. This is
no speed improvement and fails the predeclared component gate. Reject; no runtime
adoption or unchanged extension. Historical replay failures remain failures.

Plain words: Queued-delivery proposal7521032 reproduces the original repeat-risk
fault, then passes only3/5 predeclared action contracts. Explicit prepared-reply
continuation and withdrawal work; preserving old speech while answering six also
works. The original additional-context case still creates another identical
refusal, and a replacement question gives thirteen without retiring turn-1.
Correct arithmetic does not make that replacement action correct. The all-five
implementation gate fails: do not implement or deploy continue_queued_speech from
this result. There are no dispatched actions, API requests or optimizer steps.

Plain words: A distinct bounded latency diagnostic now compiles complete
single-token decode with fullgraph/dynamic shapes and CUDA graphs disabled.
Prefill remains eager and the existing DynamicCache is retained. It avoids the
previous static-cache path rather than repairing that historical failure in place.
All five exact same-process controls and both repeats remain required; stop at the
first candidate execution error, retain it, and never fall back to eager execution.
At least1.2 median speed ratio and actual compiled graphs without graph breaks are
required before any voice regression. This is neither training nor native-model
qualification. Plan: evals/fixtures/whole-decode-dynamic-cache-plan-20261018.json.

Plain words: Whole-decode job7521255/source19bdc70 terminates after the first
candidate error as planned. All10 original controls pass exact within-process
repeat checks. The candidate cannot create a graph: FLA layer_norm_gated_fwd is
explicitly decorated torch.compiler.disable. There is no eager fallback, no speed
claim and no adoption. All parameter hashes/versions remain unchanged. Slurm
COMPLETED describes the diagnostic process, not a passing component; its report
has component_execution_completed=false and promotion=false.

Plain words: The next speech-control proposal uses only the existing
resolve_pending_speech action. Current tool text says NEW answer and emphasizes
revision; the diagnostic clarifies that fulfilling an already prepared unsaid
answer must also retire its ID. General synthetic examples distinguish same
request, superseded answer and separate question with old answer preserved.
No new action, keyword routing, runtime dispatch or optimizer update is added.
The same five development probes require correct native function, exact old ID,
requested numbers, and the unchanged prepared answer for the two same-request
cases. All five must pass before runtime work; any eventual change still needs
actual voice and broader retention checks. This is an instruction probe, not
proof of training, intelligence improvement or native duplex qualification.

Plain words: Existing-action proposal7521271/source23ba558 completes54seconds and
passes4/5. Original same-request context and explicit prepared delivery resolve
turn-1 with exactly its prepared answer; stop withdraws turn-1; replacement speaks
thirteen and retires turn-1. However, the separate question that explicitly keeps
the old answer also resolves turn-1, losing still-owed speech. Reject the all-five
implementation gate, no live adoption. This is evidence of over-consumption of
speech obligations, not an arithmetic failure. Further prompt variants on these
five reused cases would remain development tuning; fresh paired semantic-state
supervision and separate untouched conversation checks are needed before any claim
of general improvement. A new optimizer run needs an explicit data and retention hypothesis, rather than this score alone.
