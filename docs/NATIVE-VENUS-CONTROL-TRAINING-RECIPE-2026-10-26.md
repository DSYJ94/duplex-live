# Venus native control training

This is a new control adapter on the published native Venus Audio model, using its original `Streaming Omni Conversation.` prefix. It does not resume the rejected Moshi adapter or the regressing long-prompt experiment.

The concrete defects are speaking before a release cue, unreliable pause/resume and background-work control, and unreliable use of delivered results. The published Venus baseline answered all four ordinary development questions in generated text. Independent speech/product qualification remains incomplete.

## Data and learning target

There are 152 TRAIN, 36 DEV and 36 HOLDOUT scenes. The authored USER inputs were rendered locally with macOS speech;159 unique exact recordings cover224 complete scenes. The initial Samantha175 collection had15 diagnosed lexical errors. Those15 clips were repaired in new versions using Daniel Enhanced150, Karen125 and Samantha150 with the exact original words; originals and failed audits remain intact. All159 word/peak checks and all224 independent Astra semantic reviews now pass. No owner conversations enter the corpus. All recordings must pass the independent full word/peak audit, and every complete scene must pass independent semantic/timing review before training.

| Behavior | TRAIN scenes | Concrete target |
|---|---:|---|
| Silence until a later cue |32|No spoken acknowledgment; listen through pauses and a distractor; answer only after the later cue.|
| Pause/resume |24|Stop during a two-step explanation; remain silent; continue the remaining step without restarting.|
| Delegate and use a result |32|Emit a native request; await an actual synthetic lookup receipt; speak its result.|
| Handle a question while pending |16|Answer the basic follow-up while the lookup remains pending; later deliver the result.|
| Cancel pending work |16|Emit cancellation; wait for the actual confirmation; suppress the cancelled fact; answer the new ordinary question.|
| Ordinary answers |32|Answer directly without unnecessary delegation; retain the frozen baseline hidden states and output distribution.|

Four matched private worlds produce different actual results for the identical question and input audio. Their response targets differ only after the real receipt becomes available. Key, cue and task-content groups are separated by fold. The original sealed evaluations remain untouched.

This first corpus explicitly asks for background work. It teaches the dispatch/result/cancellation protocol; it is **not evidence of learned automatic routing of arbitrary difficult questions**. It also uses one synthetic user voice. Natural speaker diversity, implicit difficulty routing and a real reasoning-model backend remain product work.

## Reproducible settings

Published model revision: `4681004131ea1f1fb9aca9f0db76be322657cc79`. The native Audio model is about9B parameters; its36-layer Qwen3 language core has4096 hidden dimensions,32 query heads and8 key/value heads of128 dimensions each.

The **active recipe** trains q/v LoRA in the **final attention layer35 only**, rank64,alpha64,no adapter dropout: **851,968 parameters**, checked against actual module shapes. Audio encoding, TTS, the output head, all base weights and the lower35 language layers remain frozen. This layer choice is a bounded memory/retention experiment, not a claim of optimal model anatomy or sufficient production capacity.

The unchanged lower35-layer native inputs are cached once from the approved188 TRAIN+DEV native histories. Every derived scene must reproduce full-native logits AND hidden states with **maximum error exactly0** before training. Full autograd runs through all trainable final-layer q/v computation and its entire temporal KV history. Frozen lower-layer constants do not truncate a gradient of any trainable parameter. The cached trainer loads only the language core; the original audio/TTS modules are used again for native behavioral evaluation.

One to eight **complete** TRAIN passes give38..304 optimizer updates with microbatch1 and accumulation4. A no-update full-gradient calibration on the four largest TRAIN feed workloads and a full36 DEV baseline determine the largest schedule fitting the existing deadline, with1.35 training/1.25 DEV time margins and600 seconds reserved for restoration and backup. This selection uses throughput only, never loss or held-out scores, and is recorded before the first optimizer step. All152 TRAIN scenes appear exactly once per epoch. The actual selected count is in `candidate-v11/recipe.json`;8/304 is a maximum, not an achieved result.

AdamW uses learning rate1e-4,16 warmup updates then cosine decay over the selected schedule, betas0.9/0.999,epsilon1e-8,weight decay0.01 and gradient clipping1. Decision and text losses are averaged separately within each scene, then equally across its microbatch. Ordinary scenes add output KL0.1 and hidden-state retention0.01 against the same frozen model with adapters disabled. There is no acoustic/codec loss in this phase. Deterministic CUDA/CuBLAS settings support the independent restore check. Eval attention mode preserves the released native path while autograd stays enabled; gradient checkpointing is disabled. Lossless saved-activation CPU offload on the GPU host is available only if the no-update calibration requires it. No Mac model/ASR work or CPU throttling is used.

The source cache records the released streaming API's exact audio embeddings and teacher token history, with next-token labels only at native generated choices. Incoming audio, system text and inserted framing receive no target labels. Forced teacher dispatch is data construction, never a claim that the student successfully delegated.

## Minimal checks, then training

1. All224 source/timing/split/receipt checks; all159 independent word/peak checks; all224 independent semantic judgments. No subset acceptance.
2. Released native segmented replay must reproduce captured logits exactly. Zero LoRA and causal future-input checks must pass before optimizer work.
3. The first candidate update is saved with Adam and all RNG state. A distinct process must restore it and reproduce update2 exactly, including optimizer tensors and random probes, before continuing the selected complete-epoch schedule. The expected update2 branch is proof-only; it is not counted as an extra candidate training exposure.
4. Full36 DEV teacher-forced losses are recorded before training and every76 updates. Checkpoints include optimizer/RNG, source/data/recipe hashes. Measured throughput must fit the remaining provider stop and backup reserve; otherwise stop honestly with a checkpoint.
5. Native free-running behavior, all ordinary retention questions, actual receipt/result use and independent spoken output checks are still required before adoption. Lower loss alone does not qualify a useful voice model.

The active allocation uses the existing35USD/4h additional scope, carrying9500 previous seconds. The single A10040 has a4800-second native STOP:14300 maximum cumulative new seconds<14400. Conservative scope admission is31.666667USD including non-GPU/API reserves. No credit purchase, new disk, API activation or IAM grant. Two independent guards plus provider STOP remain active. Exact code, job identities, measured outcomes and verified backup state belong in HANDOFF.md; artifacts are ignored and hashed separately.

## Diagnosed failures retained

The faster whole-sequence BF16 calculation failed the unchanged native numerical gate: maximum target CE delta0.099097 exceeded0.03. Exact native-segmented input subsequently passed with logit/CE error0. The all36-layer q/v LoRA proposal (rank16,7,667,712 trainable parameters) then exhausted39.49GiB on the no-update full-gradient calibration at microbatch1. **Neither attempt changed Venus weights.** These were numerical-path and activation-memory failures, not measured model regression or evidence that the data cannot be learned. Their raw outputs and original recipes remain preserved.

The active last-layer recipe addresses that measured memory failure. It remains unqualified until actual training, fresh-process restoration and complete native behavioral/speech evaluation supply evidence. The older rejected Moshi training is separate and is never resumed by this recipe.

## Loading a resulting checkpoint for native voice tests

After verifying its inventory/hash receipt, pass the actual checkpoint to `python -m portable.venus.native_app --model-dir /workspace/migration/venus-public-v1/Realtime-Venus-Audio --adapter-checkpoint /absolute/path/to/step-NNNN.pt`. Use the existing live provider/cost guard and loopback SSH tunnel. This loads the adapter before constructing the released duplex decoder, checks exact parameter names/shapes/finite values and model revision, freezes inference weights, and records checkpoint SHA/source/update count in the health/session profile. The proof-only expected-update2 branch is rejected. Omitting the option retains the published baseline. The loader marks every candidate experimental, non-adopted and behaviorally unqualified.

This loading path does not allocate compute or replace the training worker/source. The current microphone prototype has only the bounded arithmetic diagnostic worker. Full lookup/cancellation evaluation must wire the existing LookupWorker with its actual synthetic world and clocks; a real reasoning-model backend and implicit difficulty routing remain separate product work. No serving or behavioral success is inferred from a checkpoint loading correctly.
