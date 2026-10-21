# What we trained, what is saved, and what to test

There is no adopted final production model yet. The latest research candidate is
**native7590052**, a small adapter on the released **Context Spanning DuetaSpan-7B**
full-duplex speech model, itself based on NVIDIA PersonaPlex/Moshi. The retained
comparison baseline is **native520 / job7510331**. Numerical restoration passed on
Isambard; the latest candidate's complete behavioral evaluation has not yet run.

## The cookbook in plain language

1. Start from the published speech model, its original codec/text tokenizer and
   native 80ms streaming clock. Preserve its ability to listen while speaking and
   its native context-span mechanism for asynchronous backend results.
2. Capture synthetic native histories for ordinary answers, private result use,
   clarification, cancellation, listening, lookup and acknowledgment. Validate
   actual waveform/word/timing ownership before deriving training inputs. Keep
   held-out audio separate. Repair diagnosed failures in new versions and keep
   originals, receipts and approval chains.
3. Train small low-rank adapters, keeping the original model weights frozen.
   The original28 history components and retention losses protect earlier skills.
   Old25 and new24 approved teachers add49 speech/text examples. A strict teacher,
   independent audio-quality, semantic, conversion and numerical approval chain
   controls which examples enter training.
4. Compare short pilots using native, unforced speech and asynchronous task
   behavior. Preserve failures. Earlier pilots regressed on ordinary answers, so
   neither a falling loss nor a successful checkpoint reload qualifies a model.
5. For latest job7590052, start from native520's adapters **and Adam moments/clocks**.
   Run eight small updates with the complete original154 loss components at each
   update. Add balanced teacher-text learning over all49 recorded native streaming
   histories, with the declared truncated autograd boundary at each recorded call.
6. Independently reload in a different process. Verify original28, acoustic49 and
   native-stream49 fingerprints, adapters, Adam moments/clocks, and all saved RNG.
   This passed on Isambard. Its next native10 behavioral job was rejected by quota;
   no behavioral result exists for this candidate and it is not adopted.

## Exact latest-stage settings

The machine-readable export is
[NATIVE-7590052-RECIPE-2026-10-21.json](NATIVE-7590052-RECIPE-2026-10-21.json).
It comes from the actual saved training report, not a reconstructed proposal.

| Setting | Actual value |
| --- | --- |
| Training source | `741872fe2f9e49f35d4deba9aba96caff26c51d4` |
| Published vendor source | `79df302fa2820cae10c61bb2b7f302c52edb6ed2` |
| Warm parent | native520 /7510331, existing adapters and Adam state |
| Trainable parameters | **1,894,400**, in26 FP32 adapter tensors |
| Adapter rank / scaling | rank8 / scaling2.0, equivalent rank-scaled alpha16 |
| Adapter locations | last4 temporal layers28–31: attention output and feed-forward input/output; text output head |
| Optimizer | AdamW; beta1=.9, beta2=.999, epsilon1e-8, weight decay0 |
| Effective learning rate | **2e-7**, tenfold below the retained nominal2e-6 recipe field |
| Gradient clipping | max norm1.0 |
| Latest stage updates | **8**; carried Adam parameter clocks **520→528** |
| Sampling / attention | native BF16, original TF32 setting, MATH attention; no compile/CUDA graphs |
| Original actual histories | all28 at each update, balanced across8 behavior kinds |
| Original objectives | all154 components at every update |
| Teacher histories | all25 old +24 new, balanced across4 kinds |
| Base retention | KL weight1 and hidden-state MSE weight10 |
| Ordinary retention |10 word anchors and14 clock anchors, each with retained64-weight objective |
| First response word | retained weight16; text-padding loss weight.5 |
| Assistant acoustic learning | all49 teachers,8 assistant codebooks, weight1; no user/prefix/span targets |
| Native recorded-call text | all49 once over8 updates, balanced by kind; declared truncated call-boundary autograd |
| CUDA devices for restore | expose exactly one GPU, matching the saved CUDA RNG list |

This stage deliberately began a new recipe/RNG lineage while carrying the parent's
Adam moments and clocks. It was not represented as an exact continuation of the
parent RNG trajectory. The new run saved and independently restored its complete
Python/NumPy/Torch CPU/CUDA RNG state. It did not extend rejected7530408 unchanged.

Measured GPU allocation elapsed01:36:25 on one GH200. The raw report's1871.60-second
timer includes post-fit checks/checkpoint/original28 metrics and excludes the final
49 fingerprints and independent restoration; it is not the whole allocation time.
The measured fit/report-window peak was22,071,380,480 allocated bytes. Neither number
is a Runpod throughput or capacity guarantee.

## What is the model, in terms of size?

The published model is named **7B**. The saved PersonaPlex BF16 weight file contains
**8,371,408,896 stored tensor elements**, including its depth/audio machinery; this
is a checkpoint-storage count, not a separately measured count of unique runtime
parameters. The native Context Spanning checkpoint has a similar16.74GB size.
The latest adapter adds1,894,400 trainable parameters and a7,580,728-byte file.
The original model, codec and tokenizer remain separately pinned; the7.58MB adapter
is not a standalone speech model. There is no quantized or merged release yet.

Ordinary speech comes from the native duplex model. External reasoning/tools, when
requested, remain a separate asynchronous backend; its parameters are not counted
as part of the7B speech model. The latest tested backend implementation used
`gpt-6-luna`; that is separate from the speech-model fine-tuning and does not make
the whole system an independently qualified reasoning product.

## Saved artifacts and reproducibility

Local proof directory:

`runs/cloud-export-20261016/tree/runs/context-training-proof-7590052/`

- `adapters.safetensors`: trained adapter tensors.
- `training-state.pt`: adapter values, named Adam state, complete saved RNG and exact recipe/source metadata.
- `report.json`: full recipe, data identity, loss composition, per-update exposure/metrics and checkpoint hashes.
- `reload-report.json`: independent fresh-process restoration evidence.
- `data.json`, `audio49-data.json`, voice and approved audit/reference files: exact inputs and approval lineage.

The separately pinned16.74GB Context Spanning weights,16.74GB PersonaPlex weights,
384.64MB Mimi codec and text tokenizer are already downloaded on this Mac under
`runs/cloud-export-20261016/tree/models/hf-cache/hub/`. The approved migration copies
these and all supporting files to persistent Runpod storage, then verifies hashes
there. Exact original source is recreated with Git history and vendor bundles.

Candidate adapter SHA256:
`7bbb2e9f1822cb43db50f6ead7c14d510e7c280d520cc7e1436dced6edbcfeb5`.
Training-state SHA256:
`6a9fc6cb0949d1562f654af74cb6e86a827a0214a486f869e7bd92b1ae0b63b4`.

## Testing and acceptance

Run the destination state/runtime checks first, with a new provider execution
receipt and actual Pod/device/process identities. Evaluate all10 native development
cases and independent speech, against the retained baseline. Ordinary4/4 and the
complete review remain required before extending training. The sealed20 fresh
questions require all40 paired baseline/candidate outputs and their full review.

A user audition is an experiment, distinct from those acceptance results. A native
test must actually load this adapter and its saved native layout/voice/sampler.
The old `demo/mic-demo.sh` uses scripted words from a different earlier model and
does not test this candidate. The older portable `train.sh` likewise does not
implement the current stream49/full154 stage unchanged.

Historical first approved Runpod trial (now terminated): Pod`q3wqjwols44aos`,2H100SXM,377GB host RAM,40vCPU,
volume`cyjzip53ap` at`/workspace`. Created21Oct09:15:37.923UTC; local deadline guard
terminates this trial by11:15:37.923UTC, retaining the100GB network volume. It is
$7.98/hour compute while running and$7/month storage while retained. Execution and
validation outcomes are recorded separately in the newest HANDOFF.md.


21Oct12:57UTC destination update: the initial H100 trial is terminated. Its fresh
process restored all26 adapters, Adam528 and all saved RNG exactly, with zero new
optimizer updates. Full native numerical/behavioral qualification and a usable
microphone audition remain incomplete. An approved S3-supported replacement volume
and bounded CPU staging now prepare the full backup/runtime before any GPU rental.
See the latest HANDOFF.md and NATIVE-RUNPOD-EXECUTION-2026-10-21.json for current
resource IDs, transfer ownership, spending guards and the precise next test.


21Oct14:21UTC preservation update: complete native archive independently verified
on Mac and Runpod volume49yytzyoge, all28,470files+8links. Runtime reconstructed
with Git, full pinned native source unchanged. No production model adopted and
no usable live audition yet; numerical/behavior gates remain as documented above.


21Oct14:24UTC: all task compute is stopped. Reproducible x86 package inventory:
[NATIVE-RUNPOD-RUNTIME-PACKAGES-2026-10-21.json](NATIVE-RUNPOD-RUNTIME-PACKAGES-2026-10-21.json).
Original full destination verification:
[NATIVE-RUNPOD-ARCHIVE-VERIFICATION-2026-10-21.json](NATIVE-RUNPOD-ARCHIVE-VERIFICATION-2026-10-21.json).
Both100GB volumes remain; the old one awaits unique-receipt recovery. No GPU/live
microphone path currently runs. A single A100 attempt failed at provider capacity;
no model settings or acceptance gates were changed to obtain a result.
