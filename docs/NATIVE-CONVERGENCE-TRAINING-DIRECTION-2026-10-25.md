# Proposed training direction after the eight-update diagnostic

Status: PROPOSED DIRECTION, NOT AN EXECUTION AUTHORIZATION. No optimizer update,
budget increase, payload transfer, GPU start or gate change is authorized here.
The prepared eight-update implementation remains hard bounded at eight updates.

The owner asked whether a conventional training run is more appropriate. Yes:
the next experiment should be intended to measure learning across checkpoints,
rather than treating another standalone eight-update job as the intended endpoint.
The previous pilot established only one exposure to each of32 new reasoning scenes.
It improved their mean text loss0.1234%, worsened retained old28 loss0.8026%, and
ordinary development correctness fell2/4to1/4. It did not demonstrate convergence.
Each update already includes the full154 retention objective; batch size is not
simply four total examples. Four is the selected new reasoning-scene count.

A proper next recipe needs these elements together:

1. Keep a fixed, versioned TRAIN/validation split, all source/word/audio/receipt
   identities and all old-history/KL/hidden/native objectives. Preserve sealed
   fresh evaluation. Broader coverage should target ordinary questions, real
   request/result-use, pending-task conversation, cancel/replace and silence/cue/
   pause/resume conditions. Thirty-two new scenes provide a narrow diagnostic
   corpus; repeated epochs alone cannot establish broad product generalization.
2. Define exposure in epochs, supervised tokens and audio frames, with a bounded
   maximum update count. With the current four selected new scenes per update,
   eight updates are one new-corpus pass and32 are four passes. These arithmetic
   examples are NOT a selected training length or an authorization. A thousand
   updates would expose the same32 scenes125times, not create more coverage.
3. Pre-authorize one bounded learning-curve experiment after its exact recipe,
   budget, source and scope are reviewable. Save periodic checkpoints and use
   inexpensive fixed validation/retention measurements during the run. Choose
   early stopping from declared validation behavior, rather than imposing eight
   as the scientific training endpoint or selecting solely by training loss.
4. Record actual per-update and per-phase time, learning rate, token/frame counts,
   full objective components, gradient norms, parameter displacement and native
   request-selection/behavior outcomes. The current single higher-rate continuation
   changes rate and exposure together; it cannot isolate the rate's causal effect.
   Do not present fivefold higher rate as a demonstrated fix.
5. Keep full independent optimizer/RNG/native-logit restoration and all numerical,
   audio, behavioral and fresh gates before adoption or any further extension.
   Avoid repeatedly collecting unchanged baseline/recovery evidence. A bounded
   run longer than8 from rejected528 requires a NEW explicit exploration exception;
   neither the old32-update authorization for another candidate nor the proposed
   eight-update exception covers it. Cheap monitoring never substitutes for the
   full gates or authorizes continuing beyond the approved run.

This can remain a sustained LoRA fine-tune. The evidence does not currently isolate
adapter capacity, batch size, clipping, data quality or learning rate as the main
failure cause, and does not establish a need to train all7B weights. The original
[LoRA paper](https://arxiv.org/abs/2106.09685) supports LoRA as a task-adaptation
method, not a Moshi-specific choice of epochs or learning rate. Standard
[SFT metric documentation](https://huggingface.co/docs/trl/main/sft_trainer)
illustrates loss/LR/gradient/token monitoring; native audio/clock masks must retain
this project's approved semantics rather than importing text-padding defaults.

The separate75USD/nine-hour eight-update proposal is UNAPPROVED. It is not silently
expanded by this direction document. Before requesting a training decision, prepare
the sustained recipe, useful validation scope, exposure schedule and a whole-run
cost/time envelope, then make that ONE bounded run and its early-stop conditions
reviewable. Existing source helpers are reusable infrastructure, not model progress.
