# Echo handling and supervision review — 15 October 2026

Plain words: adopt explicit speaker provenance and echo-resistant microphone input. Do not remove the assistant's own history from attention. The current Lychee training experiment has no user-audio prediction loss to mask, so adding an “AEC loss mask” to that experiment alone would not change its optimization or repair the failed delegation behavior.

This is a source and code review. No model, training objective, microphone pipeline or cluster job was changed for this note. Local file references describe the working tree inspected on 15 October.

## What the supplied PersonaPlex paper establishes

[PersonaPlex, section 3.1](https://arxiv.org/html/2602.06053v1#S3.SS1) receives user audio, agent text and agent audio. Its hybrid prompt places role text on the agent text channel and a voice sample on the agent audio channel. Prompt positions do not contribute training loss; non-semantic audio and padded text losses are down-weighted. This is conditioning and target weighting, rather than a proposed acoustic echo canceller. The released-checkpoint appendix describes a different synthetic-voice recipe and additional Fisher training; results from the experimental model should not be silently treated as results of the released weights. The paper leaves external-tool integration to future work. These claims therefore do not demonstrate successful asynchronous delegation in our repository.

## What Moshi says about streams and echo

[Moshi, sections 3.4.3–3.4.4](https://arxiv.org/html/2410.00037v2#S3.SS4.SSS3) combines separate assistant/user audio streams with assistant text. Temporal inputs sum separate embedding tables; the smaller depth transformer models within-step tokens. Training models the joint sequence, but live inference supplies actual user audio rather than sampling predicted user audio. Keeping prior assistant output in context is part of this design.

[Section 4.4](https://arxiv.org/html/2410.00037v2#S4.SS4) already describes robustness augmentation: a scaled copy of assistant audio is delayed 100–500 ms into the user input; echo and reverb are applied together with 30% probability. Its [loss subsection](https://arxiv.org/html/2410.00037v2#S4.SS4) weights semantic audio more heavily than acoustic codebooks. The described echo augmentation is not token-ID matching or removal of all overlap frames. Separated speaker recordings and simulated separation also form part of the training recipe.

The inference recommendation from this evidence is to preserve separate speakers and realistic overlap, while treating microphone echo as a signal contamination problem. Removing every user frame during assistant speech would remove the interruptions we need the model to hear.

## What our current loss actually predicts

| Path inspected | Inputs retained | Targets receiving loss | Consequence for the proposed mask |
| --- | --- | --- | --- |
| [Lychee native data assembly](../../evals/lychee_training_inputs.py), lines 124–154 | Actual native history plus future text, speech, control and declared quiet input embeddings | Only future assistant semantic text and EOT; other labels initialized to `-100` | There is no user audio next-token objective. A user-target mask is presently a no-op. |
| [Lychee trainer](../../evals/lychee_semantic_train.py), lines 108–126 | Full captured embedding sequence | `lm_head(hidden[:, target_position - 1])` against selected text labels | Loss-position shifts must be audited, but there is no audio head CE here. |
| [Moshi recovery trainer](../../evals/recovery_train.py), lines 90–129 | Native multi-stream codes and causal references | Text plus audio slice `audio_offset:audio_offset + dep_q` | This implementation trains the output/assistant audio codebooks, rather than every input codebook. |
| [Moshi native forward](../../evals/moshi_timed_reference.py), lines 43–98 | All audio embeddings and text, native delays, optional frozen references | Native output audio logits/masks are undelayed against that same output slice | Input presence is different from target supervision. Existing masks cover undefined native positions, not waveform echo. |
| [PersonaPlex trainer](../../evals/personaplex_train.py), lines 179–212 | Native delayed history, reference and optional control feedback | Assistant text, first eight assistant audio targets and foreground policy targets | No separate user-stream CE is calculated here either. |

The stereo ingestion path explicitly loads `stem_assistant.wav` and `stem_user.wav`, hashes them and writes separated channels ([moshi_prepare_data.py](../../evals/moshi_prepare_data.py), lines 57–129). That is useful provenance. It is not evidence of how speaker leakage behaves through real speakers and a real microphone. [Persona delayed targets](../../evals/personaplex_history.py) also require channel-specific delay alignment before any new mask is applied.

## Three distinct changes

1. **Target eligibility mask:** choose which known labels contribute CE. Correct when preventing a render/assistant sample from becoming a user label, or excluding contaminated labels with unknown clean target. It does not remove echo from model inputs.
2. **Attention/context mask:** prevent a prediction from attending to certain prior tokens. A blanket ban on prior assistant audio changes the pretrained model's conditioning, and could damage continuity, speaker identity and interruption handling. No evidence reviewed here supports imposing it on the current checkpoints.
3. **Waveform acoustic echo cancellation:** use the actually rendered far-end signal to estimate and suppress its contribution to microphone capture before the audio encoder. This operates during inference, when there is no next-token training loss.

The [WebRTC AudioProcessing API](https://webrtc.googlesource.com/src/+/refs/heads/main/api/audio/audio_processing.h) explicitly separates capture `ProcessStream()` from render `ProcessReverseStream()`, exposes stream delay and recommends placing processing close to the audio hardware. That is the appropriate implementation boundary for device AEC. It processes approximately 10 ms PCM blocks; bridge their timing explicitly to our model chunks rather than changing the model's native token rate. Browser capture additionally exposes an `echoCancellation` constraint ([W3C Media Capture and Streams](https://www.w3.org/TR/mediacapture-streams/#dom-mediatrackconstraintset-echocancellation)); requesting a constraint must be distinguished from recording the applied track settings and testing actual echo rejection.

## Proposed mask contract

Implement a shared, pure target-eligibility utility only when broadening native training beyond the current semantic experiment. Inputs should be explicit per-stream/per-frame provenance, not token values: speaker role, target validity, source/played time, prompt/history membership, clean-target availability, echo-only versus double-talk annotation, and encoder receptive-field support. Require source hashes and a channel-layout/version identifier. Missing or ambiguous speaker metadata rejects a training example; do not infer it from audio-token equality.

For a future user-prediction auxiliary objective, the eligibility mask is:

`valid native user target AND genuine clean-user target source AND outside prompt/history-only region`.

If only contaminated microphone targets exist, additionally exclude support annotated echo-only or unresolved; normalize by the remaining weighted target count and report the exclusions. If a paired clean user stem exists, supervise its encoded target even when the *input* microphone signal contains echo: echo-only intervals have clean-user silence as their target, and double-talk retains the real user target. An ambiguous mixed codec token cannot be decomposed by subtracting the assistant token ID. Do not use the fact that the assistant is speaking as a user-label exclusion rule.

For assistant output, retain valid assistant text/audio supervision, including during double-talk. Mask system/voice prompts, padding/undefined targets and disallowed history targets independently. Keep normal silence targets where valid; the task must learn when to remain quiet. Match masks to the existing native delay/shift transforms, and account for encoder receptive fields at interval boundaries. Each head needs its own eligible-count accounting; an empty head either contributes a documented zero term or causes a declared required-head rejection, never an unguarded zero denominator.

For the current Lychee experiment, a useful immediate guard is an assertion that every unmasked label has role `assistant`, modality `text`, and target position after actual result availability. This makes its present objective explicit without pretending it added AEC. Actual input echo robustness requires a separate approved augmentation/audit path through the waveform encoder and cannot be measured by modifying already captured quiet embeddings.

## Concrete implementation seams and checks

- **Data:** extend the approved-ingestion metadata before `lychee_training_inputs.run` builds labels, or before Moshi/Persona native delay conversion. Preserve clean user and assistant stems; optionally produce an echo-contaminated *input* alongside the clean target with recorded gain, delay, room filter and source SHA. Freeze development/test partitions before sampling these transformations.
- **Loss:** apply a role/modality eligibility mask where `labels` are selected in `lychee_semantic_train.loss`, or where per-codebook CE masks are built in `recovery_train.loss_and_metrics` and `personaplex_train.loss_metrics`. Keep it separate from model attention masks. Do not silently claim a user-loss feature when no such head is trained.
- **Runtime:** add AEC at the future real capture/render boundary. The current [concurrent vocoder](../../evals/lychee_concurrent_vocoder.py) and [utterance playout](../../evals/utterance_playout.py) produce causal playback simulation, not an audio device. Reuse their cancellation accounting to define the eventual render reference: only PCM actually presented to the device is eligible, including real buffering/rate conversion/gain. Generated but canceled/unplayed speech is not a loudspeaker echo source.
- **Unit checks:** mask role confusion; prompt and undefined positions; exact native delay/one-token shift; valid overlap preserved; echo-only contaminated labels excluded only when no clean target; clean-user silence retained; repeated identical user/assistant codec IDs do not trigger exclusion; canceled render excluded from AEC reference; interval-boundary/receptive-field handling; no-target batches; distributed numerator/denominator consistency; unchanged baseline logits/labels when feature is disabled.
- **Behavior checks:** separate near-end-only speech, far-end-only playback and double-talk; sweep delay/gain/room response, include interruptions during assistant speech and reference discontinuities after cancellation. Measure false self-interruptions, real user interruption recall/latency, user intelligibility or WER, foreground answer/delegation correctness, and streaming budget. Echo suppression alone is not a pass if it suppresses the user. Use actual device playback/capture for the final test; synthetic echo replay is a component test.

## Supplied YouTube resource: verified scope and limitation

The public YouTube oEmbed endpoint and watch-page player metadata identify [the video](https://www.youtube.com/watch?v=shJLycXjhVY) as **“Can a Local LLM Speak and Listen at the Same Time? | Full Duplex Voice AI”**, by **Codacus**, duration **1,110 seconds**. The creator's description attributes its method to DuplexCascade: 0.6-second text micro-turns, a top-four-layer LoRA on Qwen3.6-35B-A3B, frozen-layer caching, Kyutai STT and IndexTTS. It reports approximately three seconds to first speech and incomplete hazard/backchannel success on its own tests. Those are creator-reported measurements, not our reproduced results. This is a separate text/STT/TTS architecture, not an acoustic-token fusion demonstration or an established cure for our native-model failure.

An English auto-generated caption track was advertised in player metadata, but its timed-text endpoint returned HTTP 200 with an empty body. Therefore a full transcript was **not** verified or reviewed; the observations above come from first-party metadata/description, not a claim to have watched the demos. The direct watch/oEmbed requests succeeded although the browser research tool returned errors. No video claims about OpenAI internals were used as technical evidence.

## Decision

Proceed with speaker-provenance guards and an explicit capture/render AEC evaluation plan. Preserve assistant history and real double-talk. Do not spend a larger training run on a nominal user-loss mask in the current Lychee semantic-only trainer: there is no affected loss term. Treat an eventual clean-target/echo-input training ablation as a separate intervention from the ongoing result-response behavior experiment, so its outcome is attributable.
