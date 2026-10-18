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
in chronological history. It changes no weights, dispatches no actions, makes no
external request and cannot by itself establish a runtime fix. The earlier ten
conversations remain frozen development evidence for any subsequent change.

Plain words: Physical microphone, browser-engine playback, room echo/double-talk
and human approval of the current hybrid voice remain unverified. Cloud validation
remains paused. The overall goal remains active.
