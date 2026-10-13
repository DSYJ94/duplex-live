# Duplex recovery — active from 13 October 2026

**Plain words:** the requested product is a model that generates useful speech,
keeps hearing while speaking, and delegates work while the conversation continues.
The previous timing prototype does not meet that goal. This continuation is
authorized by the owner's 13 October request to get the best result using the
remaining Isambard allocation. Historical completion statements apply only to
the previous phase.

## What the first recovery checks established

| Check | Evidence | Meaning |
|---|---|---|
| Native Qwen content and voice | Job 7377200: three questions, each in text and audio; all six produce coherent generated text and nonempty speech. Job 7377265: ASR word error 0.0 for all six. | The untouched base can answer and speak. These development questions are a component check, not a generalization benchmark or live acceptance. Human audition remains pending. |
| Native Qwen speed | 12.29–17.94 seconds to finish each 3.98–8.54 second audio answer; Transformers 5.17 / NGC 25.10 / GH200 | This batch generation path is not ready for live use. These are full-generation times, not first-packet latency measurements. |
| Existing microphone clock | `python -m evals.live_contract_probe`: actual `m2.live.main`, fake model/ear, numbered input ticks; first 49 ticks have no audio feature, subsequent features are 3,920 ms old | The 80 ms decision cadence conceals four-second input buffering. This synthetic control-flow result does not measure encoder receptive field or actual microphone hardware. |
| Delegation comparison | Full logs of jobs 7333494 and 7356783, plus executable launcher regression | Both scripts lost `--lessons`, `--init-ckpt`, and `--out`. Both consumed the default merged shard selection and saved to `runs/p1-v1`. The claimed mixture comparison and held-out designation do not establish those hypotheses. |
| Launcher repair | `python -m unittest evals.harness.test_slurm_launch_contract` failed before the repair and passes after | The tracked launcher now passes selection/init/output arguments and finds prefixed held-out IDs. It has not been used to retrain; the old negative results have not been magically corrected. |
| Demo input/playback repair | `python -m unittest evals.harness.test_live_inputs`: red before, green after | Unverified fixtures now stop before model loading; answer clips retain their real length instead of one billion padded samples (about 17 hours at 16 kHz). The clock probe still correctly fails after these repairs. |
| Environment | Job 7377265: clean container CUDA works; xformers absent | Qwen's overlay is working. MoshiRAG needs a separate dependency setup and ARC attention-kernel check. |
| Data availability | Primary-source review in the linked research note | DuplexChat is metadata and a reconstruction pipeline, not a downloadable collection of prepared two-channel audio. |

Local evidence: `runs/recovery-20261013/`, `runs/native-content-7377200/`.
Generated speech is exposed by `python -m viz.listen --runs runs --port 8766` at
`http://localhost:8766/batch/native-content-7377200`.

The three completed initial jobs consumed approximately 0.062 allocated GPU-hours
(51 + 135 + 36 GPU-seconds), including the first probe's sample-rate failure.
This is Slurm allocation time for these jobs, not the project's billing balance.

## Direction and decision gates

**Plain words:** preserve working speech from an existing duplex base, then teach
the missing interaction and delegation behavior. Keep the current Qwen timing
checkpoint as a reference. Do not spend the allocation scaling the old recipe
before showing that the new recipe can produce meaningful speech.

1. **Finish native baselines.** Qwen establishes content and speech separately.
   MiniCPM-o 4.5 gets a paced replay on the same verified user audio, with actual
   input-consumption and output-availability times. MoshiRAG is the first
   delegation candidate because released weights already accept background text.
   Its live path, not its offline retrieval loop, must prove concurrency.
2. **Select by complete behavior.** Run fresh, family-disjoint content,
   interruption, backchannel, thinking-pause and background-speech scenarios.
   Archive input/output PCM, generated text, scheduling trace and task events.
   Preserve model and source revision IDs and include cold and warm runs.
3. **Prove adaptation at small scale.** One optimizer step with finite gradients,
   then fit 16 verified conversations, reload the checkpoint, and evaluate with
   its own generated history. Report exactly which parameters train. Native
   speech, content and interruption behavior must survive before scaling.
4. **Expand the right data.** Add varied conversations and timed background
   results only after the baseline exposes its weaknesses. Split by source
   conversation, script family and speaker before synthesis or augmentation;
   hash duplicates across splits. Timing derived from the encoder's actual
   availability must match training and inference. Existing quality stamps stay
   mandatory; no unverified reconstruction enters training.
5. **Scale supervised adaptation.** Use measured frames/second, memory and
   learning curves to choose the GPU count and data amount. Compare small
   backbone adaptation, staged behavior training and separate control output
   where appropriate. Preserve an unchanged reference in each evaluation.
6. **Integrate real work and evaluate jointly.** Background requests carry a
   conversation snapshot, task ID and revision. Late results from cancelled or
   superseded tasks cannot become spoken facts. Test actual bounded tools,
   missing inputs, failures, corrections and streamed partial results while
   speech continues. Provider integration follows the same interface.

Reinforcement learning is conditional on coherent supervised behavior and a
reward that scores meaning as well as timing. Training from scratch is not the
default response to this prototype's failures.

### Provisional engineering acceptance bars

**Plain words:** these are targets to preregister for the next suite, not scores
already achieved or claims of parity with commercial systems.

- Content: at least 16/20 fresh spoken questions answered substantively and
  audibly without supplied answer text; report semantic errors separately from
  speech intelligibility. Human listening accompanies ASR.
- Duplex: at least 16/20 mid-speech corrections incorporated correctly; initial
  interruption-stop target p95 <= 500 ms. Report tolerated backchannels,
  premature replies and ignored interruptions separately.
- Serving: ten-minute sessions with no increasing input/playback backlog;
  warm response-start target p50 <= 700 ms and p95 <= 1,500 ms. Measure at the
  audio boundary, not only at the model's forward call.
- Delegation: 20 tasks with results delayed 0.5/2/5/15 seconds; at least 18 correct
  incorporations, zero fabricated early completions and zero applications of
  cancelled/superseded results. Measure each delay group and continued speech.
- Release evaluation: expand beyond these small engineering probes; seal a new
  test set before candidate tuning. Previously evaluated and accidentally
  trained-on lessons cannot serve as a fresh reserve.

## Compute and calendar

**Plain words:** the owner reports more than 16,000 GPU-hours available. Treat
16,000 as a working ceiling, not a target to burn. The recorded allocation expiry
is 21 October; expiry and remaining project balance still need current portal
confirmation. SSH, account `brics.u6si`, CUDA, and job submission were verified
live on 13 October. Slurm exposes this user's jobs, not an authoritative remaining
project balance.

| Maximum provisional envelope | Work | Release condition |
|---|---|---|
| 100 GPU-h | Baselines, environment and profiling | Bounded individual jobs, useful content and honest timing evidence |
| 400 GPU-h | Data verification and train/inference alignment | Provenance, timing checks and disjoint split manifest |
| 1,000 GPU-h | Small adaptation pilots and controlled comparisons | Reloaded free-run improvement without loss of baseline abilities |
| 10,000 GPU-h | Selected supervised adaptation | Measured throughput, positive scaling signal, fixed validation protocol |
| 1,500 GPU-h | Robustness, independent evaluation and serving optimization | Successful content, duplex and delegation integration |
| 3,000 GPU-h | Reserve | Specific evidence-backed bottleneck; otherwise leave unused |

These envelopes do not submit jobs. Eight full days at 16,000 GPU-hours would
require about 84 GPUs continuously; queue capacity, distributed training support
and enough diverse data are separate constraints. A deadline does not justify
silently repeating a tiny corpus or launching an unverified distributed run.

## Execution tracker

| Work | State | Next evidence |
|---|---|---|
| Current handoff, source and job-log audit | Done for initial scope | Continue to challenge claims against actual commands/results |
| Native Qwen speech diagnostic | Complete, job 7377200 | Human audition; broader fresh suite only if pursuing this candidate |
| Generated-speech ASR check | Complete, job 7377265 | Six samples at WER 0.0; no claim of human voice-quality review |
| Microphone schedule repro | Complete; known failure retained | Replace the delayed stream contract in the selected native path |
| Delegation launcher repair | Regression passes | Do not reuse the historical ratio verdict as evidence |
| MiniCPM native duplex replay | Jobs 7377413/7377469 exposed missing audio-decoder dependencies; bounded retry prepared | TorchAudio 2.9 requires TorchCodec; install only matching ARM wheels into a job-specific overlay. No measured baseline yet. |
| MoshiRAG live delegation baseline | Source seams checked; environment not ready | Isolated dependencies; real xformers ARC kernel; live delayed-result replay |
| Training dataset/split and pilot | Pending baseline decision | Native format plus provenance and independent held-out manifest |
| Broad training | Not started | Only after small reloaded free-run pilot passes |

Source-backed selection and execution details:
[recovery base assessment](research/2026-10-13-recovery-base-selection.md).
The Thinking Machines reference describes the behavior target; reproducing its
276B-parameter system is not a credible use of this allocation.
[Primary reference](https://thinkingmachines.ai/blog/interaction-models/).
