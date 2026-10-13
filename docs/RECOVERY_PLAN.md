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
| Native MiniCPM duplex | Job 7377673: 60-second paced replay, four generated answers; processing p50 0.477 s and p95 1.768 s per one-second input chunk; max backlog 3.095 s | Native words and speech work on the new container, but this is not real-time acceptance. Bread-staling answer incorrectly attributes it to drying. Human audition remains pending. |
| Controlled delayed context | Job 7378025: 10 s worker wait, seven processed packets (six speaking) during wait; result injected 0.467 s after ready; bread answer uses supplied mechanism | Positive reference-delivery/use check on one example. No learned dispatch, actual research or stale-result rejection tested. Max input backlog 3.913 s remains unacceptable. |
| Native Qwen speed | 12.29–17.94 seconds to finish each 3.98–8.54 second audio answer; Transformers 5.17 / NGC 25.10 / GH200 | This batch generation path is not ready for live use. These are full-generation times, not first-packet latency measurements. |
| Existing microphone clock | `python -m evals.live_contract_probe`: actual `m2.live.main`, fake model/ear, numbered input ticks; first 49 ticks have no audio feature, subsequent features are 3,920 ms old | The 80 ms decision cadence conceals four-second input buffering. This synthetic control-flow result does not measure encoder receptive field or actual microphone hardware. |
| Delegation comparison | Full logs of jobs 7333494 and 7356783, plus executable launcher regression | Both scripts lost `--lessons`, `--init-ckpt`, and `--out`. Both consumed the default merged shard selection and saved to `runs/p1-v1`. The claimed mixture comparison and held-out designation do not establish those hypotheses. |
| Launcher repair | `python -m unittest evals.harness.test_slurm_launch_contract` failed before the repair and passes after | The tracked launcher now passes selection/init/output arguments and finds prefixed held-out IDs. It has not been used to retrain; the old negative results have not been magically corrected. |
| Demo input/playback repair | `python -m unittest evals.harness.test_live_inputs`: red before, green after | Unverified fixtures now stop before model loading; answer clips retain their real length instead of one billion padded samples (about 17 hours at 16 kHz). The clock probe still correctly fails after these repairs. |
| Environment | Job 7377265: clean container CUDA works; xformers absent | Qwen's overlay is working. MoshiRAG needs a separate dependency setup and ARC attention-kernel check. |
| Data availability | Primary-source review in the linked research note | DuplexChat is metadata and a reconstruction pipeline, not a downloadable collection of prepared two-channel audio. |

Local evidence: `runs/recovery-20261013/`, `runs/native-content-7377200/`,
`runs/minicpm-recovery-7377673/`, `runs/minicpm-recovery-7378025/`.
Generated speech is exposed by `python -m viz.listen --runs runs --port 8766` at
`http://localhost:8766/batch/native-content-7377200` and
`http://localhost:8766/batch/minicpm-recovery-7377673`; delayed-context comparison:
`http://localhost:8766/batch/minicpm-recovery-7378025`.

Eight bounded jobs consumed approximately **0.143 allocated GPU-hours**
(514 GPU-seconds), including failed environment attempts. Jobs 7377148, 7377200,
7377265, 7377413, 7377469, 7377588, 7377673 and 7378025 are all terminal.
This is Slurm allocation time for these jobs, not the project's billing balance.
No recovery training or other recovery GPU job is currently running.

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
| MiniCPM native duplex replay | Complete, job 7377673; earlier jobs exposed missing decoder dependencies | Tested process-local PCM WAV reader; TorchAudio features and container Torch unchanged. Improve backlog and assess interruptions before acceptance. |
| MiniCPM delayed context injection | Complete, job 7378025; reference used, processing continues | One fixed result only. Add real dispatch, delayed/cancelled results and fresh tasks; reduce backlog. |
| MoshiRAG live delegation baseline | Pinned weights downloaded (7378305); official Llama tokenizer access denied. Isolated environment build 7379025 running after diagnosed source-build issues | Actual ARC kernel and tokenizer access, then live delayed-result replay |
| Training dataset/split and pilot | Native data conversion 7379026 running; executable guarded pilot committed | 16 train + 4 diagnostic conversations, audited intervals, finite backward step, save/reload and speech retention. This is not a fresh generalization split. |
| Broad training | Not started | Only after small reloaded free-run pilot passes |

Source-backed selection and execution details:
[recovery base assessment](research/2026-10-13-recovery-base-selection.md).
The Thinking Machines reference describes the behavior target; reproducing its
276B-parameter system is not a credible use of this allocation.
[Primary reference](https://thinkingmachines.ai/blog/interaction-models/).

### Exact current continuation commands

**Plain words:** these commands validate local repairs and reproduce bounded
cluster probes. Inspect the latest job and ledger before submitting duplicates.

```sh
.venv-api/bin/python -m unittest evals.harness.test_live_inputs \
  evals.harness.test_slurm_launch_contract evals.harness.test_reference_wav
.venv-api/bin/python -m evals.live_contract_probe
# The clock command intentionally exits 1 until the delayed path is replaced.
python3 -m viz.listen --runs runs --port 8766
```

Deploy only through local commit/push, cluster `git pull --ff-only`, and checksum
comparison. Cluster working directory: `/projects/u6si/duplex-live/repo`.

```sh
sbatch evals/native-content-probe.slurm          # 1 GPU, 45 minute cap
sbatch evals/minicpm-recovery-probe.slurm        # 1 GPU, 30 minute cap
sbatch --export=ALL,RESULT_FIXTURE=/projects/u6si/duplex-live/repo/evals/fixtures/bread_background_result.json \
  evals/minicpm-recovery-probe.slurm            # controlled delayed context
sbatch --export=ALL,PROBE_DIR=/projects/u6si/duplex-live/runs/native-content-JOB \
  evals/recovery-environment.slurm              # 1 GPU, 15 minute cap
```

Fetch complete logs before filtering. Probe directories use job IDs and refuse
overwrite. The MiniCPM launcher creates a job-specific dependency overlay and
retains the base Torch/CUDA packages. Its WAV-reader accommodation is necessary
because [TorchAudio 2.9 switched its load function to TorchCodec](https://docs.pytorch.org/audio/stable/generated/torchaudio.load.html);
the attempted [matching TorchCodec version](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)
had no ARM CPU wheel in the queried index (job 7377588). Package import is
insufficient: actual reference decoding, model generation and output audition
are the checks.

## Immediate continuation

**Plain words:** the next independent work is the MoshiRAG live baseline and a
small native Moshi training smoke. MiniCPM now provides a working comparison,
not a final architecture selection. Its general fine-tuning scripts do not
yet establish timed duplex training support.

1. Build an isolated MoshiRAG environment; verify ARC xformers on the actual
   GH200 before downloading more than the required pinned weights. Run the
   live path with delayed local references; do not treat its blocking offline
   retrieval example as concurrent delegation.
2. Prove the official native Moshi training path on 16 verified conversations:
   validate assistant-left/user-right audio and word alignments, perform a
   backward step, save/reload, then generate with its own history. Report
   missing checkpoint keys rather than accepting a partial load silently.
3. Prepare a fresh split across scenario families and speakers before candidate
   tuning. Audit the existing synthetic material for suitability; the prior
   515-example collection is not an untouched evaluation reserve.
4. Profile MiniCPM speaking versus listening compute and playback buffering.
   One-second input chunks and the measured 3–4 second backlog currently miss
   the target; do not label median compute alone as conversational latency.

See the research note for exact official sources, data schemas and environment
constraints. Human audition of all new audio remains open.

## Native Moshi continuation — in progress

**Plain words:** exact weights are now local. Training remains gated on valid
word intervals; failed conversions have not been admitted to the trainer.

- Model acquisition job 7378305 completed: native Moshi, MoshiRAG, ARC and
  local ASR are pinned and hashed in `runs/moshi-models-7378305/model-manifest.json`.
  The official Llama 3.2 tokenizer returned `GatedRepoError`; owner access was
  requested asynchronously. That blocks reference encoding, not native Moshi.
- Environment jobs 7378179/7378259/7378351/7378463 stopped on the Opus/CMake
  source-build boundary. pip's isolated build imports override a Python CMake
  launcher; the latest job uses the actual pinned ELF binary. Existing Torch
  environments are untouched. Job 7379025 is the current bounded retry.
- Conversion 7378330 stopped before annotation because the assumed <=100 s
  length selected no examples. Actual dialogs are 104.84–116.20 s. Conversion
  7378464 rejected user backchannels under an overly narrow event rule and
  rejected zero-width HF word intervals; only one passed, so training stopped.
  Job 7379026 uses the official trainer's WhisperTimestamped alignment method
  and accepts verified user backchannels while retaining complete assistant
  utterances, interval validity, transcript agreement and source approval.
- The committed pilot rechecks all source approval records and artifact hashes,
  requires disjoint 16/4 manifests, audits strict base loading and adapter
  gradients, hashes every frozen parameter, and compares fresh-process reload
  logits. Native speech replay compares the untouched model with the adapter.
  A successful plumbing test will not be reported as a quality improvement.

Exact submission once the new data job passes:

```sh
sbatch --export=ALL,DATA_DIR=/projects/u6si/duplex-live/runs/moshi-data-JOB,MODEL_MANIFEST=/projects/u6si/duplex-live/runs/moshi-models-7378305/model-manifest.json \
  evals/moshi-training-smoke.slurm
```

Each training pilot is one GPU with a 45-minute cap. The script preserves the
container GPU stack, pins plain Moshi/trainer source separately from MoshiRAG,
and refuses overwrite. Do not submit until the data summary says verified.
