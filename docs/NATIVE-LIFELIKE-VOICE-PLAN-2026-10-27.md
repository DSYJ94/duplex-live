# Native lifelike voice work

Plain words: the product needs clear, natural speech as well as correct silence and delegation. The current adapter was trained for conversation control, with its speech generator frozen. Assistant acoustic examples and listening tests are a separate, necessary part of this work. The current candidate remains rejected; no voice improvement is claimed yet.

## Work already executed

- Completed independent Whisper recognition of all36 recorded native outputs. All36 pass the unchanged peak threshold; only5 match the generated user-facing text after unit-boundary spacing correction (raw original4) at the existing WER threshold. These compare speech against generated text, not expected answers. Source-bound report: `NATIVE-VENUS-SPOKEN36-AUDIT-2026-10-27.json`. Word accuracy is necessary but cannot establish naturalness.
- Added an explicit, hash-bound reference-audio option to the native Venus speech generator. It changes its native voice conditioning; it does not replace duplex inference with an external speech API. The released reference stays the default. Custom references must be finite, unclipped mono16kHz audio,2–30seconds, with an exact supplied SHA256.
- Prepared72 authored voice cases:48TRAIN/12DEV/12HOLDOUT across12 families. There are66 spoken targets and6 exact-silence targets. The latter make no synthesis requests. TRAIN contains44 spoken targets; held-out scripts are excluded from training.
- Prepared67 exact ElevenLabs requests: the66 spoken targets plus one neutral reference passage,4168characters. No user context, scoring information, owner conversations or credentials enter payloads. A conservative0.50USD/1000character bound is2.084USD; the voice sub-cap is2.5USD inside the approved10USD API/40USD total allowance. No new speech request has been made while specific credential-file access awaits approval.
- Confirmed ADR0003 Amendment A permits closed-content training under the owner's separate enterprise agreement. Corrected future factory manifests that incorrectly labeled ElevenLabs output as MiniCPM/open-only. New ElevenLabs cache namespaces include the model identity; old unpinned caches and receipts remain intact.

## Comparison completed; fix the diagnosed pacing first

All24 recordings completed and4962files were independently hash-backed up. The base model uses multiple short speech units; step190 often emits a whole sentence andTURN_EOS together, cutting speech to roughly one second. The reference swap did not fix this. Repair the measured word-to-unit timing before more control training. The raw transcript scorer also merged words across unit boundaries; new derived reports retain boundary spaces, keep raw reports and thresholds unchanged, and do not rerun recognition. No naturalness approval follows from these scores.

The new timing source collection has83exact assistant phrases. These local recordings supply word times, not the final voice identity. A new measured native-unit schedule and interrupted-tail resumption are implemented but await full source and native qualification. The additional85USER recordings and64TRAIN/16DEV/24freshHOLDOUT repair source plan are prepared. All originalTRAIN/DEV examples remain unchanged in the source plan.

## Fixed comparison protocol

Run the complete eight ordinary diagnostic scenes under three fixed conditions: base weights/released voice, step190/released voice, and step190/an existing synthetic ElevenLabs reference. The same original system prompt, input audio, seed and sampling apply. Independently recognize the reference first and all24 outputs afterward. Existing inspected scenes are diagnostic, never fresh held-out evidence.

This records adapter/reference differences. A subsequently diagnosed encoder-state leak between scenes makes this comparison descriptive; it is not an isolated causal estimate. New conversations now explicitly reset the encoder state. Preserve raw chunks, text, timing, hashes and failures. Do not choose a voice from a passing subset or call lower control loss a speech improvement. The first comparison attempt stopped before model launch because local watchdog readiness preceded remote publication; the corrected source acknowledges the actual remote write before starting the guest watchdog. Carry both consumed intervals in the ledger.

## Generate and use assistant speech correctly

Render the frozen source plan with a single consistent stock voice first. Retain provider/model/settings, exact request hashes, raw PCM,24kHz WAVs and source receipts. Do not clone a real person, add background effects, normalize away clipping, or blindly repeat ambiguous paid requests. The renderer checks existing account quota and stops instead of buying credit.

The curriculum covers conversational warmth, phrase breaks, emphasis, questions, honest uncertainty, corrections, permitted acknowledgements, calm pacing, pronunciation, pending work and resumption. It does not train compulsive fillers. A silence instruction requires silence, including no acknowledgement.

Independently check all generated words and peaks, then review delivery by listening. Align accepted assistant audio with exact native text/audio targets before any acoustic optimizer use. Adding ElevenLabs **user** audio alone does not train the assistant's voice. This small collection is an acoustic pilot and regression suite, not enough evidence for broad voice convergence.

First retain the native voice-conditioning option if it measurably helps without speech/control regression. If speech remains poor, use the paired results to pin a small native acoustic-adapter/conditioning repair recipe. Do not jump to full9B fine-tuning or a larger reasoning model to fix timbre. The approved control continuation still resumes actualstep190 Adam/RNG and retains the existing objectives; an acoustic optimizer change must be explicitly specified before execution.

## Evidence required for a usable voice

Compare complete paired recordings for intelligibility, natural phrasing/intonation, stable identity, audio artifacts and conversational timing. Use blinded A/B listening with anchored1–5naturalness ratings; a recognizer or automated quality estimate cannot certify humanlike delivery. Preserve model-emitted actions separately from speech, verify interruption and real pending cancellation, and prohibit acknowledgements during required silence. Final adoption also requires the complete fresh24 native control set and ordinary retention.

Reuse the already acquired DailyTalkContiguous collection for research reference before acquiring more audio. Its recorded CC-BY-SA provenance and release restrictions stay explicit. YouTube material enters only through the existing source/license ledger with suitable permission, speaker separation and alignment; a public URL alone is not an approved acoustic target. No new video download is needed for this pilot.

Sources: [ElevenLabs speech API](https://elevenlabs.io/docs/api-reference/text-to-speech/convert), [expressive speech documentation](https://elevenlabs.io/docs/overview/capabilities/text-to-speech), [official API pricing](https://elevenlabs.io/pricing/api). The exemption is the repository's `decisions/0003-provenance-classes.md`, Amendment A; public provider terms do not override that recorded enterprise agreement.

## Current execution, 27 October evening

All83assistant timing sources and all85newUSER recordings now pass complete independent word/peak checks. All218requiredUSER recordings are merged by source hashes; all140private-world scenes have actual completed/cancelled worker receipts. The full292paced teacher set passes the source/token/timing/fold audit:216TRAIN/52DEV/24freshHOLDOUT,811speech units,42matched private-world groups. Old152TRAIN/36DEV source scenes remain unchanged. A first semantic review omitted the native unit-end contract and rejected147teachers for assuming closure at unit onset; its complete failed report remains immutable. One corrected complete review includes the verified public implementation's generate-audio-before-reset contract and explicit unit intervals. No optimizer work may start without every teacher passing.

The prepared continuation restores actualstep190adapter/Adam/RNG, uses LR5e-5 with16update warmup/cosine decay, rank64/alpha64/final-layerq/vLoRA851968, accumulation4, and all old ordinary KL0.1/hidden0.01 objectives. It admits1–8complete216scene epochs by measured throughput, at most432new updates/622total, below the approved512additional limit. Fresh268native and prefix caches are required after the timing and encoder-isolation repairs. A distinct process must match adapter/Adam/RNG/loss exactly before sustaining. All24fresh cases plus all8oldordinary cases are collected for BOTH parent190 and continuation, with unchanged speech thresholds and independent ASR. No voice or candidate is adopted on loss or these recordings alone.

External API allocation remains10USD total: semantic review reserve7.5USD including failed attempts, prepared ElevenLabs voice reserve2.5USD (exact conservative batch ceiling2.084USD). No purchases. ElevenLabs credential-file access remains pending specific human approval after automatic review rejected extraction from ignoredAPI keys.md; the exemption itself is confirmed. No new speech request has been made. A100 work uses the existing100GBdisk; the longer nativeSTOP envelope replaces only the stopped VM after full verified copyback. GPU intervals and all previous spending remain carried.
