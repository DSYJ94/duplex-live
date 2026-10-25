# Product direction after the native recovery

25 October 2026. LOCAL is the sole executor. The objective remains a good native
duplex voice assistant: understand ordinary requests, listen while speaking, obey
silence and pause instructions, answer simple questions, and delegate harder work
without blocking the conversation. The existing unfinished goal already covers
this; another goal or another training framework is unnecessary.

## What we have, and what we do not

We have real trained adapters, preserved checkpoints and a reproducible native
training/evaluation path. The latest reasoning experiment performed eight updates
from global Adam step520 to528, exposing each of32 new scenes once. It trained
1,894,400 adapter parameters, not the full7B model. The subsequent GCP work
recovered and evaluated that saved state; it did not add training updates.

The endpoint fits those32 training scenes slightly better: text cross-entropy
fell0.1234%, acoustic cross-entropy0.0415%. The old28 weighted objective worsened
0.8026%. In the complete paired development review, ordinary answers fell2/4 to1/4
and speech-agreement passes8/10 to7/10. Speech agreement is not task correctness.
Silence, pause/resume, faithful result delivery and cancellation remain inadequate.
Saved528 remains rejected. See the [complete32 fitting review](NATIVE-SAVED8-COMPLETE32-FIT-REVIEW-2026-10-25.json)
and [complete paired behavior review](GCP-NATIVE-SAVED8-COMPLETE20-REVIEW-2026-10-25.json).

This establishes an unsuccessful small adaptation, not its unique cause. Narrow
coverage, only one new-scene exposure, conservative updates and competing retention
losses are plausible contributors. Batch size, a particular learning rate or model
size has not been isolated as the cause. Exact optimizer/RNG restoration rules out
some state-loss explanations; it does not establish useful learning.

Nor do four questions establish an intrinsic7B intelligence ceiling. A later
adapters-disabled A100 comparison scored one complete, two partial and one wrong
answer under the [corrected strict rubric](NATIVE-CONTROL40-BASE4-SCORING-CORRECTION-2026-10-24.json).
An [older ancestor comparison](NATIVE-ANCESTOR12-COMPLETE-REVIEW-2026-10-20.json)
reported4/4 for published base and native520 under a different execution setup.
Those are not a controlled hardware/precision/decoding comparison. Preserve this
discrepancy; do not convert it into a claim that a larger model must fix the problem.

Critical native artifacts remain preserved:40,132 files,81,154,452,891 bytes and8
links, plus the separately verified semantic/recovery/behavior evidence. The broad
Mac source backup remains partial. Do not repeat the81GB hash audit. A fresh GCP
provider read in this turn found all four project VMs TERMINATED. No new paid
compute, training or external API payload was started for this sense check.

## Data actually acquired

The [acquisition receipt](PUBLIC-NATIVE-DATA-ACQUISITION-2026-10-25.json) records
5,092 files /14,771,623,383 bytes, all verified against pinned upstream hashes.
The per-file ledger stays in ignored `runs/public-data-20261025/`; large public
files live outside OneDrive at `/Users/daryl/.cache/duplex-live/public-data`.

| Source | Measured acquisition | Purpose and limit |
| --- | --- | --- |
| [DailyTalkContiguous](https://huggingface.co/datasets/kyutai/DailyTalkContiguous) |2,541 dialogues,21.674h;13.77GB|Natural conversation and voice-retention candidates; not a reasoning/delegation corpus. Declared CC-BY-SA-4.0; training/release use not approved.|
| [Glaive function calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) |112,960 text rows;63,218 with a function-call marker;271MB|Tool selection/arguments/result grounding candidates. Declared Apache2. These are synchronous text examples, not trained native async behavior.|
| [UltraChat200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) |Only three `train_sft` shards;732MB|General assistant text candidates, declared MIT. Physical row count not measured. No public test shards fetched.|

These are raw candidates. They have not entered TRAIN, changed any project folds,
replaced original audio, acquired semantic approval or become native speech targets.
Text tool calls still need verified spoken timelines with pending questions,
interruptions, delayed results, cancellation and deliberate silence. Downloading a
large generic corpus does not by itself solve our observed failures.

## Choose the foundation before a long repair

**First comparison: Realtime-Venus-Audio9B.** It combines native duplex speech with
asynchronous delegation and a returned-result protocol. This is closer to the
desired product than teaching the entire mechanism to the current small adapter.
The official demo code at `e53ae8d5bde92b30364b26f2e4c13c9db7a8ffea`
supports **both Audio and Omni** (`demos/model/adapter.py::_load_real_blocking` and
`demos/README.md`), despite Omni naming in parts of the Harness README. Select
Audio for this voice product. Pin weights to
`4681004131ea1f1fb9aca9f0db76be322657cc79` and use an isolated runtime.
[Official code](https://github.com/inclusionAI/Realtime-Venus),
[model card](https://huggingface.co/inclusionAI/Realtime-Venus).

The default demo uses Codex task execution. Do not start that backend: this chat
remains the only executor. The initial comparison must use an isolated synthetic
worker with no shell, accounts, browser actions or external model calls. Such a
worker measures delegation mechanics and result fidelity, not backend intelligence.
Only the frontend's own speech counts as native behavioral evidence; do not add
keyword-triggered muting, canned ordinary answers or forced spoken results.

**Second option, only if needed: NVIDIA VoiceChat11B.** This is native duplex with
a9B language backbone and a separate tool-call channel. The published offline
function-call example includes prepared API responses; our async pending-question
and cancellation behavior still needs live testing. It also needs a different
runtime. Avoid simultaneously staging or renting hardware for every candidate.
[Official model and limitations](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B).

Neither candidate is adopted or locally performance-verified. Their9B/11B sizes
are not the reason to select them; stronger pretrained instruction/tool behavior
may remove work that our current adaptation has been trying to recreate.

## Short route to a usable product

1. Finish public Audio checkpoint staging, retaining the original model unchanged.
2. Run one small, frozen-weight native comparison: ordinary reasoning, Geronimo
   silence including its mention in the initial instruction, pause/resume, simple
   follow-up during pending work, faithful delayed results and cancellation.
   Use reused development audio or newly authored development audio, never sealed
   qualification audio. Record real speech, timing and worker events. Reuse current
   baseline recordings; another full state-recovery run is unnecessary here.
3. If Venus is clearly better, connect the existing reasoning worker through a
   narrow backend adapter and make it testable interactively. If it is not, try
   VoiceChat once before returning to the preserved Moshi path. Integration or
   playback defects must be distinguished from frontend policy failures.
4. Train the selected foundation on diagnosed gaps. Start with a broader curated
   mixture of general speech, reasoning/delegation and explicit interaction
   control. Include cases where basic reasoning stays local and difficult work
   is delegated. Natural acknowledgments are appropriate sometimes; training must
   also distinguish them from an explicit request for total silence.
5. Use a sustained LoRA run with loss curves, several data exposures, periodic
   development checks and an independent final evaluation. LoRA rank/coverage and
   full fine-tuning are experimental choices after that comparison. Do not expand
   to all weights merely because the small adapter failed; full training is more
   expensive and can damage speech/retention. Do not transplant Moshi adapters to
   another architecture.

The draft sustained32 learning-curve recipe is **parked**, not authorized or
executed. Its planning helpers are source-only. It is no longer the next
recommended spend. There will be no additional checkpoint-recovery, giant backup
audit or multi-candidate training loop before the short foundation decision.

The original GCP grant is still100USD/12 cumulativeGPUh for saved8 recovery with
zero new optimizer updates. The prior conservative usage bound is91.9844806USD and
40,351 GPU seconds, not an invoice. A different-model GPU comparison needs one
explicit bounded scope/budget decision; the public local downloads do not change
that grant. No new purchase or repeated credit top-up is proposed.
