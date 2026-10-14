# Duplex recovery — active from 13 October 2026

**Plain words:** the requested product is a model that generates useful speech,
keeps hearing while speaking, and delegates work while the conversation continues.
The previous timing prototype does not meet that goal. This continuation is
authorized by the owner's 13 October request to get the best result using the
remaining Isambard allocation. Historical completion statements apply only to
the previous phase.

## Architecture and training direction — clarified 14 October 2026

**Decision:** adapt a released, pretrained full-duplex speech model and connect
it to a separate reasoning/tool worker. This is supervised fine-tuning, not
pretraining from scratch. MoshiRAG is the working reference-integration candidate;
the final voice backbone is still conditional on audible quality and interaction
comparisons, including PersonaPlex. Do not assume their conditioning interfaces
or checkpoints are interchangeable.

The old Qwen path trained new control-token embedding/output rows while freezing
the language core. It established timing behavior but not useful self-generated
spoken content. Preserve it as a baseline; it is not the main scale-up recipe.
Qwen remains a candidate backend/teacher and an unchanged speech/content
reference. No new Qwen layer-training campaign has been started in recovery.

The completed 64-step adaptation pilot trained LoRA adapters on plain Moshi,
using 16 reused conversations. It verified gradients, frozen-weight preservation
and serialization, not quality improvement. The corrected reference replay used
released MoshiRAG weights with inference-harness repairs, not newly trained
MoshiRAG weights. Training with causally timed reference results is still to be
implemented and verified.

After backbone selection, start with adapters inside the pretrained transformer.
Expand to selected pretrained blocks or broader fine-tuning only when evaluations
show adapters are insufficient. These remain fine-tuning approaches. The target
is useful speech plus waiting, interruption, cancellation, and returning to late
results without fabricated early answers. Do not use lower training loss as
acceptance or scale the tiny reused pilot corpus.

The owner authorized GPT-Live as the reference on 14 October. Benchmark GPT-Live
itself, separately from Realtime API measurements. Use matching conversational
tasks and, where possible, the same client-delegation worker/facts/delays to
isolate voice and interaction quality. API/account access is not yet verified.
Official architecture and client delegation:
https://developers.openai.com/api/docs/guides/live
https://developers.openai.com/api/docs/guides/live-delegation

Owner listening feedback on the corrected replay: “sounds okay not great”.
Record this as one informal audition, not a completed blind voice comparison.

Immediate sequence: (1) GPT-Live reference plus voice-backbone comparison;
(2) actual context-based backend requests and results with task IDs/revisions;
(3) fresh split conversations with timed result supervision;
(4) verified timed-reference backward/save/reload and a bounded adaptation;
(5) joint content/voice/duplex/delegation evaluation, then larger training.

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
The initial eight jobs are terminal. Current follow-on job states are tracked
below; their time is not included in this initial 0.143 GPU-h figure.

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
| MoshiRAG live delegation baseline | Pinned weights downloaded (7378305); environment 7379661 passed actual ARC CUDA attention (max SDPA difference 0.0). Official Llama tokenizer access remains denied | Authorize account; verify tokenizer, then live delayed-result replay |
| Training dataset/split and pilot | Data 7379026 passed. Pilot 7379340 completed four finite optimizer steps, changed 674 adapter tensors and preserved frozen weights; reload comparison failed its 0.05 tolerance | Exact-weight/mode diagnostic 7379832 and native speech comparison 7379833. These 16/4 folds are diagnostics, not fresh generalization. |
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

## Native Moshi continuation — verified training, quality remains open

**Plain words:** native Moshi now trains, saves and reloads correctly on GH200.
A 64-step pilot reduced diagnostic loss, but generated answers remain weak.
Do not scale this corpus or select a model on this loss result.

- Model acquisition 7378305 pinned and hashed native Moshi, MoshiRAG, ARC and
  local ASR: `runs/moshi-models-7378305/model-manifest.json`. Meta's official
  Llama 3.2 tokenizer returned `GatedRepoError`; reference encoding awaits
  owner-approved access. Native Moshi does not need that tokenizer.
- Conversion 7379026 accepted 20 conversations (16 train / four diagnostic
  evaluation), totaling 2,237.36 seconds. Highest retained utterance WER: 0.05.
  Twenty-two candidates were rejected. Actual durations are 104.84–116.20 s;
  native segments round to 120 s. WhisperTimestamped supplies assistant word
  intervals. Source approval, source/converted hashes and word audits are
  rechecked before training. These reused examples are not a fresh family- or
  voice-disjoint generalization test. Human audition remains pending.
- Native trainer commit `2acc879fe7c48f885a18f6cc9548bccb2674d87b` and Moshi
  `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` run without replacing container
  Torch/CUDA. LoRA rank eight trains 24,242,176 parameters / 674 tensors.
  Frozen base weights remain byte-identical; adapter gradients are finite and
  nonzero. Training peak allocation: 23,544,080,896 bytes.
- Four-step job 7379340 trained and saved, then failed its provisional BF16
  logit comparison (text 0.125 / audio 0.375). Diagnostic 7379832 established
  exact saved/reloaded adapter and base values. The numerical discrepancy's
  cause is not fully established. No passing stamp was assigned to that job.
- Jobs 7380342 (four steps) and 7380505 (64 steps, learning rate 1e-5) passed
  exact saved-adapter hashes and fresh-process canonical logit comparison:
  float32, TF32 disabled, checkpointing disabled, compilation disabled;
  maximum text/audio difference 0.0 against a 0.001 bound. This verifies
  serialization under that execution mode, not BF16 logit equivalence.
  Voice generation still uses BF16. Saved adapters are not optimizer-resumable
  training checkpoints.
- Job 7380505 diagnostic loss was 1.9321053 versus 2.20383549 after four steps.
  First/last epoch mean training losses: 2.082494 / 1.813906. Sixteen passes
  over sixteen examples do not establish generalization. Its spoken answers
  contain clear content errors; no quality improvement is established.
- Its 40-second paced replay processed 80 ms frames at 0.0220 s median /
  0.0236 s p95, with maximum input backlog 0.000196 s. Those are frame compute
  times, not response-start or end-to-end conversational latency. Microphone,
  interruption and network acceptance remain untested. Actual speech audit
  7380455 confirms intelligibility on the earlier four-step comparison (WER
  0.0737 base / 0.0206 adapter), not semantic correctness.
- The latest replay code freezes inference parameters and resets sampling
  seed after warmup. Earlier saved speech comparisons predate that correction;
  they are exploratory rather than strictly paired sampling comparisons.

MoshiRAG environment 7379661 built xformers from pinned source on ARM/GH200
and verified the actual BF16 ARC attention kernel against SDPA (max error 0.0).
Pinned upstream source is imported directly because its Hatch package rejects
`../README.md`. Websockets was subsequently installed in the isolated runtime;
container Torch remains unchanged. Full ARC encoding and reference use have
not run yet.

The actual upstream reference callback reproduced an old encoding overwriting
its replacement. `evals/moshi_rag_runtime.py` now rejects superseded, cancelled
and reused-session results. Four tests against the actual upstream channel and
manager pass on the cluster, including revision propagation into the encoding
child task. These are delivery safeguards, not evidence of learned dispatch
or natural-language cancellation.

The owner completed access and cluster browser authentication on 2026-10-13.
Actual pinned tokenizer files downloaded successfully. Credentials remain in
the cluster model cache; no tokens are recorded in chat or the repository.

Reference probe 7386266 passed complete inference-weight coverage and actual
localhost HTTP encoding. All 239 required keys are present; two released ARC
tensors (`embedder.cont_tok.0.weight`, `embedder.rec_tok.0.weight`, each [1,3072])
are absent from and unused by the pinned inference source. Only those explicit
extras are omitted. Empty references intentionally emit zero vectors; nonempty
references must be finite and nonzero. Seven-, twenty- and 361-token references
returned [1,2,4096], [1,5,4096], [1,91,4096] tensors in 51–64 ms. Repeated text
produced identical values. Peak allocation: 14,932,377,600 bytes. This is an
encoder/transport test, not spoken use or delegation acceptance. Earlier jobs
7385889/7386158 failed overly strict probe assertions and remain recorded.

**Correction to all early reference-use verdicts:** the replay queued embeddings
with `update_streaming_sum_tensors`, then called `step` without the upstream
batch runner's explicit `apply_pending_streaming_sum_condition` operation.
Consequently jobs 7386447, 7387021, 7387196, 7387230 and 7387611 never applied
those reference rows to the language-model input. Their worker timing and
speech artifacts remain real, but their failure to repeat supplied facts is
not evidence against the conditioned model or encoder. Do not carry those
negative reference-use verdicts forward.

`evals/moshi_rag_reference_step.py` now applies and verifies each row before
generation, checks clearing after the reference ends, and the replay requires
queued-row and applied-frame counts to match. The regression uses the real
upstream queue/consume methods on CPU and detects an omitted application.

Separately, seeded BF16 histories varied before worker launch: longer warmup
7387021, fresh processes 7387196, and no CUDA graphs/compilation 7387230 did not
establish token-identical prefixes. In 7387230, the first 100 user-code and
traced CUDA RNG hashes matched, yet generated tokens differed from frame 72.
The cause remains unestablished. Eager processing p50 was 0.132–0.145 seconds
per 80 ms frame and backlog reached 39–49 seconds; that mode fails live use.
The corrected CPU causal audit uses the actual worker-start frame, rather
than confusing wall-clock and consumed-input time, and still rejects the
known fresh-process and eager histories. A seed alone is not a causal control.

Request-aligned 7387611 queued three fixture references after native RAG tokens
at frames 47, 231 and 439; its ten-second bread result was queued at 45.280 s.
Actual audio transcription agrees with text (WER 0.0084), but this run predates
the application fix and cannot establish reference use. The corrected run and its scope are recorded below.
These are reviewed fixed facts selected by known request order, not actual
research, semantic tool routing or generalized learned delegation.

Audio audit 7386830 transcribed the generated WAVs, confirming the core wrong
answers, but the full-clip reference transcription also hallucinated repetitive
words with timestamps beyond the recording length over a near-silent tail.
Its large WER is not proof of actual repeated speech. Original audio/report
remain intact; a separate tail-trimmed ASR diagnostic now records the inspected
input duration and timestamp validity. Human listening remains open. The WAVs
are available in the local gallery, e.g.
`http://localhost:8766/batch/moshi-rag-speech-7386447-delayed-reference`.

### Corrected request-aligned speech — application passes, late answer fails

Job **7387777 completed**. Four native retrieval-request tokens launched workers
for known questions; fixtures supply reviewed facts by request order, not actual
research or semantic tool routing. The replay verifies all **33 queued rows
were applied** to the active conditioning input before generation. Two CPU
regressions against real upstream methods pass (0.041 s), including omitted
application detection and clearing after the last row.

Median/p95 frame processing: **0.0165 / 0.0186 s** for 80 ms input frames.
Maximum input backlog: **0.0781 s**. This is paced frame processing, not
microphone/network/playback acceptance or response-start latency.

The bread worker started at 35.218 s and returned at 45.271 s; its condition
was queued at 45.280 s and applied in frame 565. The model had already given a wrong drying/gas
explanation. After arrival it only appended “I suppose”, with no correction
before the next question. **Long-delay result incorporation still fails this
case.** It must learn to defer unsupported answers and return to completed work.
Earlier negative reference-use claims from the missing-application harness are
invalid; this corrected negative result has actual row-consumption evidence.

Audio audit **7387786 completed**. Core answer words agree with text, including
the wrong bread mechanism, but the last truncated answer provokes ASR
repetitions/timestamps beyond recording length. Its full-clip WER is therefore
not valid speech-quality evidence (`asr_time_bounds_valid=false`). Original
WAVs and reports remain intact. Human listening is pending; corrected audio:
`http://localhost:8766/batch/moshi-rag-speech-7387777-delayed-reference`.
No causal paired-seed improvement or generalized delegation claim is made.

All recovery jobs are terminal as of this update. Initial probes consumed
514 allocated GPU-seconds; native continuation 2,076; reference continuation
1,346. Combined: **3,936 seconds / 1.093 GPU-hours**, across 39 allocated
single-GPU jobs. Failed jobs count; one CPU model download and two unallocated
dependency cancellations do not. This is job allocation time, not a verified
remaining project balance. No large training run has started.

Next: introduce fresh held-out reference facts (including facts unavailable
from pretraining), replace request-order fixtures with an actual background
worker, and implement/verify causal timed-reference training. Prioritize
waiting, interruption/cancellation and post-result correction, then evaluate
multiple samples with reviewed labels. Seed equality alone is not an inference
control in this setup. Do not scale the reused 16-example native corpus or
select a production model on diagnostic loss. Keep the native baselines and
all failed evidence; microphone/playback and human voice acceptance remain open.
