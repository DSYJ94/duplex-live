# Overnight recovery: 18 October

Plain words: Updated 18 October 2026, 10:54 UTC / 12:54 Berlin. There is a better
experimental voice system, but no GPT-Live-equivalent native duplex model is
qualified. The current demo uses frozen Kyutai speech recognition/synthesis,
frozen local Qwen3.5-9B for words and actions, and actual GPT-6 Astra requests
when the local model delegates difficult reasoning. It is a modular hybrid.
No native speech weights or trained controller adapter were adopted overnight.

Plain words: The current demo is [localhost:8767](http://localhost:8767).
Refresh an already-open page before Start conversation. The newest two recordings
are first on the [listening page](http://localhost:8766/audio/voice-diagnostics/overnight-results.html#latest).
The temporary Isambard service is 7519401, source 7915b1c. It has a four-hour idle
limit and a four-and-a-half-hour job limit. No agent-started microphone session or
physical-device acceptance is claimed. Earlier connections retain their old service.

## What improved

| Problem | Verified change | Practical limit |
|---|---|---|
| “Please go on” destroyed paused speech and started a new explanation | Retain the generator through lexical recognition, then let the model select `resume_speech`. Four matched spoken controls improve from 2/4 to 4/4. The butterfly explanation completes from the same generator; 435,840 source samples schedule once. | These are development cases. Two unseen valid continuation cases also retain their generators; two other planned inputs fail audio screening and remain unscored. |
| Cancelling reasoning restarted the ordinary explanation | New `cancel_reasoning_and_resume_speech` executes real selected delivery cancellation and resumes retained audio without an announcement. Both matched trials make actual Astra requests; neither speaks the cancelled result. The improved full water-cycle clip has independent WER0 and no repeated source samples. | Cancellation stops delivery, not an already-running remote computation. It does not add arbitrary agent actions. |
| A long third-turn prompt paused for about14seconds | Exact-prompt profiling identifies cold kernel work. Warm synthetic2048/4096-token inputs before readiness. On the same2462-token request, first token 12.763 → 0.139 s; full controller step 14.189 → 1.459 s, with the same 35-token native action. | Server measurements, not microphone-to-ear latency. Cold isolated startup warmup costs 9.71 s; the new demo’s shared-cache warmup costs 0.39 s. |
| The resume action itself had unnecessary ID generation | Bind the sole retained generator in application state. Native resume 43 → 14 tokens; measured controller 1.866 → 0.644 s before the later warmup check. | The model still decides whether to resume. There is no user-keyword router or repaired model argument. |

Plain words: Long-input warmup changes neither user requests nor model weights.
Four saved native proposals preserve exact action/arguments and input-token counts.
Actual long and short network regressions both complete their original generators
with WER0. Production playback JavaScript replays actual wire order without
repeated source samples. The replay uses a simulated AudioContext, not a browser
engine or physical loudspeaker. The new demo retains the controller warmup proof
alongside strict source/model loading evidence.

## What training accomplished—and what it did not

Plain words: A bounded TEXT-controller training pilot completed 216 verified new
teacher labels, 72 held-out text fixtures and 648 optimizer updates. Its rank 8
output correction has 2,019,328 parameters. Published source parameters remain
unchanged; checkpoint, optimizer and random-state restoration pass strict checks.
This is not Qwen-Omni fine-tuning or native duplex pretraining.

| Variant | Development result | Verdict |
|---|---|---|
| Frozen controller | 60/72 full text contracts. All baseline arithmetic values correct; four exact-format failures. Replacement speech retirement0/8. | Preserved baseline. |
| Global learned output correction | 62/72, but six malformed/repetitive calls and four numerical errors. | Rejected. Lower training loss did not establish better behavior. |
| Same tensors, function names only | 64/72; all 60 original passes retained. All eight replacement cases retire old speech, but some wording failures and a 49 → 50 numerical error remain. | Not qualified. |
| Unrestricted function selector on broader questions |Answered 149 × 267 as 40083 instead of 39783. | Rejected for general routing. |
| Selector only when actual pending-speech actions exist | 22 reused reasoning cases, nine actual Astra requests; all 16 numerical results correct, prohibited-tool requests stay local. | Useful diagnostic, not final model qualification or demo adoption. Unsupported completion wording remains. |

Plain words: The best current demo therefore uses the frozen controller with the
verified runtime changes. The training checkpoint and failures are preserved for
further work; they are not silently deployed or described as a native voice gain.

## Is this behavior or intelligence?

Plain words: Both matter. Some failures were missing application capabilities:
no model action could preserve a paused generator, and the media loop destroyed
it before semantic interpretation. Adding the correct action/state boundary fixes
those matched cases with unchanged weights. Other failures involve reasoning and
calibration: direct hard arithmetic can be wrong, learned routing can suppress
needed delegation, and unsupported “all tasks are complete” statements remain.
The local model must choose when to request stronger reasoning. Narrow numerical
success is not evidence of general intelligence or complete-task reliability.

## Voice and evaluation findings

Plain words: The fresh river/magnet inputs pass source-audio screening. Both
actual conversations resume their retained generator. Primary Whisper gives the
river aggregate a repeated transcript with invalid timestamps, so that score is
unaccepted. A LOCAL second recognizer transcribes it without the repeated passage,
with 3/54 word error; the magnet clip has 1/47 error in both recognizers. All eleven
predeclared clips are checked, including the two invalid user inputs; both
recognizers report those repetitions. Original gates remain unchanged.

Plain words: Automatic approval review rejected exporting those clips for external
transcription. That job was not submitted and no clip was sent. The safer local
Kyutai recognizer completed instead, with zero external requests. It is also the
prototype input listener, so it is a second recognizer relative to Whisper, not
an independent end-to-end acceptance system. Owner voice-quality judgment remains
pending; no approval is inferred from transcription scores.

## What the supplied research changed

Plain words: The [Moshi/BayLing review](research/2026-10-15-moshi-bayling-architecture.md)
reinforced retaining the assistant’s generated history and keeping speaker streams
explicit. Actual history fixes and semantic pause now have measured control
results. BayLing’s token fusion is not a drop-in substitution for the current
codec/frame layout. [Resource decisions](research/2026-10-15-resource-decisions.md)
separate implemented seams from claims supported only by papers.

Plain words: The [AEC review](research/2026-10-15-aec-loss-review.md) distinguishes
speaker-owned loss targets from removing assistant context or cancelling acoustic
echo. Ownership masking is implemented in the native training work. The current
ASR/controller/TTS hybrid has no user-stream next-token loss to mask. Browser
capture requests echo cancellation; room/device echo and double-talk still need
physical testing. Masking training labels alone cannot establish echo suppression.

## Remaining work and limits

Plain words: Freeze this candidate for broader fresh full-conversation checks,
including wrong actions, lost work, premature completion, clarification, no-tools
reasoning, cancellation, speech fidelity and latency. Keep those metrics separate.
Train further only against a demonstrated remaining failure with retention checks;
additional compute alone is not the identified unlock.

Plain words: This prototype still waits for a complete ordinary text action before
starting its speech generator. Its controller input is bounded to 4096 tokens.
Native private-result use remains unqualified, and this reasoning profile does
not expose private-record reads or general agent actions. No physical microphone,
browser-engine playback, room AEC, destination-cloud GPU or overall GPT-Live
parity claim is made. Earlier static-cache experiments failed and are not adopted.

Plain words: Cloud validation remains paused; the $25 allowance is unused.
Full public speech/controller assets, new teacher data, checkpoints, optimizer/
random states, captures and reports are saved locally. The export includes 19.33 GB
of Qwen assets and 6.43 GB of speech assets with size/hash checks. The provider-neutral
Context Spanning recipe has CPU/amd64 checks; the current hybrid has not passed a
cloud GPU resume. Rebuild architecture-specific wheels; do not copy ARM binaries
to x86. Owner recordings and credentials are excluded from exported evidence.

Plain words: Exact jobs, continuation commands and artifact locations are in
[HANDOFF](../HANDOFF.md). Main new proofs live under
`runs/recovery-continuation-20261016`: `long-prefill-warmup-integrity.json`,
`cancel-resume-matched-integrity.json`, `semantic-resume-fresh-integrity.json`,
`local-secondary-asr-integrity.json`, and their replayable verifiers. A copied
proof/script inventory lives in `runs/cloud-export-20261016/evidence/overnight-20261018`.
The source bundle is refreshed after report commits. Goal remains ACTIVE.
