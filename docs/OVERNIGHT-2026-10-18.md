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
The current experimental live service is job 7516574 on source 0d63370, with the
compact reply format, task-state instruction and atomic cancellation guard. Its
predecessor is preserved in the evidence; no accepted native weights were replaced.

| Finding | Actual evidence | Meaning |
|---|---|---|
| Ordinary speech was held behind background reasoning | Job 7515423: controller finished23.446s, API finished28.974s, first scheduled speech29.736s. Job 7515717: speech24.330s during actual API16.020–30.120s. | A software scheduling defect was fixed; no new weights were needed. |
| Hard reasoning is real, separate work | Job 7514992: three easy local answers correct; four harder direct-local counterfactuals wrong; four model-selected real stronger-model answers correct. | Both base reasoning limits and conversation control matter. This does not establish general routing accuracy. |
| Actual network conversations work in narrow tests | 7515878 / 7515879: ice answer and actual subset count 19, both independently transcribed WER0. 7515880 / 7515881: selected counting-result delivery cancelled, only new shadow answer spoken, WER0. | Real user PCM and output packets, no artificial delivery delay. Still synthetic input, not physical microphone acceptance. |
| Shorter action format reduces controller work | Matched resident jobs 7516121 / 7516122: both ten contracts pass; median generation 2.116→1.648s, generated tokens 48→38. | A component latency gain. It changes some wording/routing; no full-system speed claim. |
| False completion is a real model-control failure | New typed jobs 7516385 / 7516387: both claim cancellation with no earlier task. Original format also attempts forbidden external delegation; guard blocks the API. | Earlier easy-case success was insufficient. Failures remain in the record. |
| Task-state instruction improves this narrow failure | 7516456 / 7516457: both complete sixteen questions; ten numerical answers match CPU checks, two logic answers correct, unknown private facts not invented, absent-task cancellation not claimed, both no-tools requests stay local. | Five actual external requests each. Some extra wording remains unsupported; this small typed screen is not broad qualification. |
| Invalid speech could mutate cancellation before rejection | Guard now validates content and owed speech identities before cancellation; fixture proves invalid IDs and blank speech leave active work untouched. | Fourteen relevant transport/session/dispatch/reasoning/format tests pass. |
| Static-cache decode is currently broken | Eager job 7516498 passes 10/10, median 2.001s. Static-eager job 7516499 and static-compiled job 7516500 fail all ten cases on hybrid mask/cache errors. | Rejected; compiled-wrapper creation is not successful inference. Live runtime unchanged. |

Plain words: New task-state network regressions 7516501 / 7516502 preserve the ordinary
answer and count 19 with speech WER 0. Native 7516503 / 7516504 and compact 7516505 / 7516506 cancel
actual result delivery and preserve the shadow answer; independent ASR adds an
and and changes cancelled spelling, WER 0.0714. These mismatches remain visible.
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
worktrees and matching local/remote file digests before jobs. Qwen loads all 775
source tensors exactly, including explicitly retained unused MTP weights; no
missing weights or shared speech-runtime edits. Reused inputs and these diagnostic
questions are development evidence, never approved training data.

Plain words: Seven new real-network captures and ten independent played clips
are locally packet/hash/bounds verified, alongside earlier capture backups. The
listening page has 42 synchronized recordings with exact served HTTP/WAV hashes.
Public speech assets and all fourteen Qwen controller files are complete; the
19.33 GB controller export passed every expected size and SHA check. API keys, session credentials
and actual owner conversations are excluded from these exports.

Plain words: The existing portable/context recipe covers the older Context
Spanning trainer, with CPU restore tests; it does not establish cross-provider GPU
resume or deployment of this newer hybrid. The current controller wheels are
ARM64-specific. Keep the pinned public assets and source bundle, and verify a
fresh destination runtime before claiming x86 GPU portability. Cloud validation
remains paused; the owner asked to prioritize Isambard and actual quality gains.

Plain words: Spoken regressions 7516566 / 7516568 both avoid invented cancellation
and answer 24 times 36 as 864 without an external request. Compact speech audit
7516569 has WER 0 on both replies. Original-format audit 7516567 adds then and
mishears the first factor as 824; the planned answer remains correct.

Plain words: Next gates are a real browser/device listening trial and reliable
acknowledgments without
sacrificing genuine interruption response. Do not expand training or declare a
backbone accepted from these small controller tests. See [HANDOFF](../HANDOFF.md)
and [EXPERIMENTS](../EXPERIMENTS.md) for exact sources, jobs and current activity.


Plain words: Later acknowledgment work reproduced a software restart: one model
answer caused two speech-generator starts after an ASR-omitted Uh-huh. A recorded-
only pause experiment retained the generator and restored exactly5,179 unheard
samples; all403,200 source samples were heard once. Its real pause is1.368seconds,
so this is not natural uninterrupted backchannel behavior. Full speech WER remains
0.06818 in both variants. Recognized acknowledgments and browser resume remain open.

Plain words: The genuine one-word Stop check revealed a more serious control
failure: both variants recognize Stop, then speak the full explanation again.
Completed speech WER0 is not task success. A new model-selected silent-withdrawal
action fixes stop-only typed cases, but its earlier seven-case model gate failed
stop+new-answer and cancellation-with-unrelated-speech. A summary also copies the
old full answer despite the right route. These changes have NOT entered the live
demo. All failures, corrected audits and exact original recordings are preserved;
no native weights were adopted, and no development inputs became training data.


Plain words: The later controlled repair preserves the assistant's actual generated
answer in chronological context before an interruption, with explicit delivery
state. It never treats generated words as proof of full playback or physical
hearing. With the same frozen model, eleven typed/captured control checks now pass;
they remain a small development gate, not whole-system qualification.

Plain words: Real audio comparison7517208/7517210 uses identical user PCM,
recognition and first generated answer. Without generated history the controller
repeats all48 words; with it, the requested main point becomes one29-word sentence.
Both completed recordings have speech WER0, showing why speech transcription alone
misses task failure. Bare Stop7212 now retires the actual speech and produces no
new reply; its old prefix remains partial/unscored. First-step revision7214 gives
only the first step, complete speech WER0. Four captures and seven audit clips are
locally reconstructed/hash/bounds verified. The comparison page now has42 players.

Plain words: These are recorded-audio results. The published owner demo remains
on0d63370 and has not adopted the new pause/history path. Browser-owned pause and
resume buffers still need implementation; server timing does not prove device
playback. Natural recognized acknowledgments, general agent actions, native
private-result qualification and physical microphone/AEC acceptance remain open.
