# Venus native control training

This is a new control adapter on the published native Venus Audio model, using its original `Streaming Omni Conversation.` prefix. It does not resume the rejected Moshi adapter or the regressing long-prompt experiment.

The concrete defects are speaking before a release cue, unreliable pause/resume and background-work control, and unreliable use of delivered results. The published Venus baseline answered all four ordinary development questions in generated text. Independent speech/product qualification remains incomplete.

## Data and learning target

There are 152 TRAIN, 36 DEV and 36 HOLDOUT scenes. The authored USER inputs were rendered locally with macOS Samantha at175 words/minute;159 unique exact recordings cover224 complete scenes. No owner conversations enter the corpus. All recordings must pass the independent full word/peak audit, and every complete scene must pass independent semantic/timing review before training.

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

Published model revision: `4681004131ea1f1fb9aca9f0db76be322657cc79`. The36-layer Qwen3 language core has4096 hidden dimensions,32 query heads and8 key/value heads of128 dimensions each. LoRA trains only q/v projections: rank16,alpha16,no adapter dropout. Actual module shapes and the expected7,667,712 trainable parameters are checked at runtime. Audio encoding, TTS, the output head and all base weights remain frozen.

Eight complete TRAIN passes give304 optimizer updates with microbatch1 and accumulation4. AdamW uses learning rate1e-4,16 warmup updates then cosine decay, betas0.9/0.999,epsilon1e-8,weight decay0.01 and gradient clipping1. Decision and text losses are averaged separately within each scene, then equally across its microbatch. Ordinary scenes add output KL0.1 and hidden-state retention0.01 against the same frozen model with adapters disabled. There is no acoustic/codec loss in this phase.

The cache records the released streaming API's exact audio embeddings and teacher token history, with next-token labels only at native generated choices. Incoming audio, system text and inserted framing receive no target labels. Forced teacher dispatch is data construction, never a claim that the student successfully delegated.

## Minimal checks, then training

1. All224 source/timing/split/receipt checks; all159 independent word/peak checks; all224 independent semantic judgments. No subset acceptance.
2. Released native segmented replay must reproduce captured logits exactly. Zero LoRA and causal future-input checks must pass before optimizer work.
3. The first candidate update is saved with Adam and all RNG state. A distinct process must restore it and reproduce update2 exactly, including optimizer tensors and random probes, before continuing the remaining302 updates. The expected update2 branch is proof-only; it is not counted as an extra candidate training exposure.
4. Full36 DEV teacher-forced losses are recorded before training and every76 updates. Checkpoints include optimizer/RNG, source/data/recipe hashes. Measured throughput must fit the remaining provider stop and backup reserve; otherwise stop honestly with a checkpoint.
5. Native free-running behavior, all ordinary retention questions, actual receipt/result use and independent spoken output checks are still required before adoption. Lower loss alone does not qualify a useful voice model.

The bounded allocation uses the existing35USD/4h additional scope, carrying4600 previous seconds, with one new7200s maximum A10040 interval. No credit purchase. Two independent guards plus provider STOP remain active. The exact code, job identities, outcomes and backup state belong in HANDOFF.md; run artifacts are ignored and hashed separately.

## Native gradient correction after measured numerical failure

The full159 recording/224 semantic reviews subsequently passed in source-audio-v7/timelines-v7. The first real9B check rejected the faster whole-sequence BF16 calculation: maximum target CE delta0.099097 exceeded0.03, even though all21 top-token decisions agreed and exact segmented replay was0. The original failed outputs and thresholds remain intact.

Training now uses `native-segmented-bptt-v1`: the exact released native feed shapes, with full autograd through its KV history. No prefix/cache detach or truncated temporal credit assignment. The language module stays in eval mode to preserve the native attention path; autograd remains enabled. Gradient checkpointing is disabled because it conflicts with mutable native KV history. The small-model gradient test agrees with full-history gradients. Deterministic CUDA/CuBLAS settings support exact independent Adam/RNG continuation. Real9B proof and measured throughput still control admission. Eight-epoch/304-update maximum and all objectives remain unchanged; checkpoints are saved each38 updates (one epoch).
