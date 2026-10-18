# Overnight recovery: 18 October

Plain words: This is an interim evidence report. No complete native duplex model
or modular hybrid is qualified. The current experimental voice demo uses streaming
Kyutai speech recognition and synthesis, frozen local Qwen3.5-9B for words/actions,
and explicit GPT-6 Astra requests for harder reasoning. It is not Qwen-Omni
fine-tuning or pretraining. No optimizer steps or native weight adoption occurred
in this overnight continuation.

Plain words: For the live prototype, open http://localhost:8767 on this computer.
The microphone starts only after Start conversation. The temporary service and
SSH access have limited lifetimes; check HANDOFF before restarting. Recordings for
comparison are at http://localhost:8766/audio/voice-diagnostics/overnight-results.html.
The published live service remains frozen on source49e820b; newer development
changes below have not silently replaced it.

| Finding | Actual evidence | Meaning |
|---|---|---|
| Ordinary speech was held behind background reasoning | Original5423: controller finished23.446s, API finished28.974s, first scheduled speech29.736s. Fixed5717: speech24.330s during actual API16.020–30.120s. | A software scheduling defect was fixed; no new weights were needed. |
| Hard reasoning is real, separate work | Typed4992: three easy local answers correct; four harder direct-local counterfactuals wrong; four model-selected real stronger-model answers correct. | Both base reasoning limits and conversation control matter. This does not establish general routing accuracy. |
| Actual network conversations work in narrow tests | 5878/5879: ice answer and actual subset count19, both independently transcribed WER0. 5880/5881: selected counting-result delivery cancelled, only new shadow answer spoken, WER0. | Real user PCM and output packets, no artificial delivery delay. Still synthetic input, not physical microphone acceptance. |
| Shorter action format reduces controller work | Matched resident6121/6122: both ten contracts pass; median generation2.116→1.648s, generated tokens48→38. | A component latency gain. It changes some wording/routing; no full-system speed claim. |
| False completion is a real model-control failure | New typed6385/6387: both claim cancellation with no earlier task. Original format also attempts forbidden external delegation; guard blocks the API. | Earlier easy-case success was insufficient. Failures remain in the record. |
| Task-state instruction improves this narrow failure | 6456/6457: both complete sixteen questions; ten numerical answers match CPU checks, two logic answers correct, unknown private facts not invented, absent-task cancellation not claimed, both no-tools requests stay local. | Five actual external requests each. Some extra wording remains unsupported; this small typed screen is not broad qualification. |
| Invalid speech could mutate cancellation before rejection | Guard now validates content and owed speech identities before cancellation; fixture proves invalid IDs and blank speech leave active work untouched. | Fourteen relevant transport/session/dispatch/reasoning/format tests pass. |
| Static-cache decode is currently broken | Eager6498 passes10/10, median2.001s. Static-eager6499 and static-compiled6500 fail0/10 on hybrid mask/cache errors. | Rejected; compiled-wrapper creation is not successful inference. Live runtime unchanged. |

Plain words: New task-state network regressions6501/6502 preserve the ordinary
answer and count19 with speech WER0. Native6503/6504 and compact6505/6506 cancel
actual result delivery and preserve the shadow answer; independent ASR adds an
and and changes cancelled spelling, WER0.0714. These mismatches remain visible.
Cancellation suppresses result delivery; it does not abort an already running
remote request. General agent actions and reasoning replacement are absent.

Plain words: Speech fidelity is not the only remaining problem. Acknowledgments
still interrupt and restart some explanations. Some record replies narrate
internal unavailable-value metadata; one named-unavailable-project case asks for
identification again. The current published voice has not received owner approval.
The existing MaAI acoustic-head investigation remains shadow-only because model
rights and publisher feature parity are unresolved.

Plain words: Browser microphone settings request acoustic echo cancellation,
but room/device echo handling is untested. Native token-loss masking and physical
echo cancellation are different mechanisms. The current ASR/controller/TTS hybrid
has no user-stream next-token training loss to mask. No browser-engine or physical
microphone acceptance is claimed from network loopback, unit tests or server WAVs.

Plain words: All source changes are committed and pushed, with immutable cluster
worktrees and matching local/remote file digests before jobs. Qwen loads all775
source tensors exactly, including explicitly retained unused MTP weights; no
missing weights or shared speech-runtime edits. Reused inputs and these diagnostic
questions are development evidence, never approved training data.

Plain words: Seven new real-network captures and ten independent played clips
are locally packet/hash/bounds verified, alongside earlier capture backups. The
listening page has28 synchronized recordings with exact served HTTP/WAV hashes.
Public speech assets are complete; check the separate Qwen export integrity
receipt before declaring its19.33GB backup complete. API keys, session credentials
and actual owner conversations are excluded from these exports.

Plain words: The existing portable/context recipe covers the older Context
Spanning trainer, with CPU restore tests; it does not establish cross-provider GPU
resume or deployment of this newer hybrid. The current controller wheels are
ARM64-specific. Keep the pinned public assets and source bundle, and verify a
fresh destination runtime before claiming x86 GPU portability. Cloud validation
remains paused; the owner asked to prioritize Isambard and actual quality gains.

Plain words: Next gates are the spoken absent-task/no-external-tools regressions,
a real browser/device listening trial, and reliable acknowledgments without
sacrificing genuine interruption response. Do not expand training or declare a
backbone accepted from these small controller tests. See [HANDOFF](../HANDOFF.md)
and [EXPERIMENTS](../EXPERIMENTS.md) for exact sources, jobs and current activity.
