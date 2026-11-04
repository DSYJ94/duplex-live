# A substantial native-capacity experiment

2026-11-04. Research/design only: no provider operation, new API payload, training or adoption. This expands the earlier narrow comparison recommendation in response to the user's request for a proper large-capacity variant. The old four-layer adapter is not a ceiling.

## Decision

Build a credible **all-layer, all-projection adaptation variant**, alongside an explicitly specified **six-new-block structural variant adding over one billion foundation parameters**. A dense six-block unfreeze is a useful third option: it trains a similar number of existing parameters without growing the architecture. Do not run all three as an unstructured sweep. Use a common preserved parent, adequate data, sustained learning curves and a predeclared selection decision.

Extra capacity is a legitimate hypothesis. It is not the established cause of the current failures: the completed native24 test fixed a speech cut-off without changing language/control traces or weights. The remaining frozen speech failures still require separate diagnosis. [Current measured result](../NATIVE-VENUS-TTS-COMPLETION-RUNTIME-COMPAT-RESULT-2026-11-04.json).

## Three substantial variants

The following counts use the root agent's **actual pinned config**: 36 decoder blocks, hidden4096, intermediate12288, 32 query heads, 8 KV heads and head dimension128. They supersede generic Qwen configurations with intermediate14336. Counts below are analytical expectations; the actual module/state audit remains authoritative.

| Variant | Expected trainable size | What changes |
| --- | ---: | --- |
| Current last-block q/v rank64 | 851,968 | Existing reference, not a sufficient exploration of LoRA |
| All36 blocks, q/k/v/o/gate/up/down, rank128 | 349,175,808 | Broad attention **and MLP** adaptation |
| Same coverage, rank256 | 698,351,616 | Substantial recommended adaptation arm |
| Same coverage, rank512 | 1,396,703,232 | Genuine >1B adapter parameters; optional if data/learning curves justify rank growth |
| Dense unfreeze six existing blocks | 1,157,678,592 | Existing weights become trainable; foundation parameter count stays unchanged |
| Add six copied identity-initialized blocks, train new blocks | 1,157,678,592 **new foundation parameters** | 36→42 blocks; inference architecture grows |

LoRA counts are `rank * sum(input_dim + output_dim)` over the seven projections and all36 blocks; that dimension sum is75,776 per block. Original adapter preservation may add its851,968 parameters separately, depending on the explicitly audited branch design. Dense-block expectation is192,946,432, including the applicable RMSNorm parameters. Six new blocks increase the approximate9B total by approximately1.158B; the exact total must come from the real model inventory.

**Recommendation:** prepare rank256/all36/all7 as the main adaptation arm, not another tiny last-layer q/v tweak. Rank512 is a valid substantial option, but >1B trainable adapters does not automatically make it preferable to rank256. The structurally expanded42-layer model is the clear answer if the experiment must add >1B actual foundation weights. Dense six-block training tests a less constrained update without the architectural inference growth. A comparison between all36 LoRA and a six-block dense suffix changes both coverage and update type: it is a product comparison, not a clean rank-only causal experiment.

## What the research supports

1. **Coverage matters as well as rank.** Thinking Machines' *LoRA Without Regret* tests Qwen and Llama models, finding that MLP-inclusive/all-matrix adaptation can outperform attention-only LoRA even when parameter counts match. It also shows low-rank curves separating from high-rank/full-training curves later in optimization; a few updates cannot establish capacity saturation. Its learning-rate and batch results are task-dependent. This supports testing broad adaptation and meaningful horizons, rather than treating our current last-layer q/v result as a verdict on LoRA. [Primary research](https://thinkingmachines.ai/blog/lora/).

2. **Low-rank restrictions can matter, with a retention tradeoff.** *LoRA Learns Less and Forgets Less* compares instruction tuning and continued pretraining, including roughly100K instruction pairs and20B-token regimes. Low-rank configurations can learn less than dense training while retaining more source capability; MLP coverage, rank, scaling and learning rate affect the result. Its empirical hyperparameters differ from Thinking Machines', reinforcing the need for controlled recipe choices rather than a universal rank/alpha/LR formula. [Paper](https://arxiv.org/html/2405.09673v2).

3. **Expressivity is real but not an optimization guarantee.** *The Expressive Power of Low-Rank Adaptation* proves sufficient-rank results for idealized networks, including a transformer construction using embedding-dimension/2 rank. That is an existence result under the paper's assumptions, not a required rank for our task, a training guarantee or a proof for this native multimodal/GQA model. [Paper](https://arxiv.org/html/2310.17513v2).

4. **Function-preserving growth is mathematically possible.** Net2Net describes transformations carrying an old network function into a wider/deeper network, avoiding a complete cold start. Architecture-specific conditions still matter. Preservation at initialization means the added capacity initially contributes no new learned knowledge. [Google Research / Net2Net](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/).

5. **Transformer block expansion has a substantial precedent.** LLaMA Pro copies and identity-initializes added transformer blocks, freezes inherited blocks and trains new ones. Its32→40-block example reports an approximately80B-token code/math corpus,15,900 training steps,16 H800 GPUs and2,830 H800 GPU-hours; the reported recipe includes sequence4096, batch1024, LR2e-4,6% warmup and cosine scheduling. These are the paper's scale reference, not a Venus-ready recipe or a minimum required for interaction tuning. [Paper §3.2/§4.1](https://arxiv.org/html/2401.02415v2).

6. **LoRA versus dense training are different parameterizations.** The original LoRA formulation learns factored updates while freezing the underlying pretrained weights. Enlarging those updates does not insert new transformer layers; a merged update still occupies the original matrix shape. [Original LoRA paper](https://arxiv.org/abs/2106.09685).

## Native-safe construction

For the structural arm, distribute six inserted blocks across the36-layer backbone, for example one after each six original blocks. Copy source blocks, then initialize their residual-output projections to produce identity blocks, following a verified architecture-specific expansion. In a pre-normalized block, zero attention output and MLP down-projection outputs are a candidate mechanism; they require actual-code verification. Do not zero every matrix: that can destroy useful inherited features and gradient paths.

Preserve original block identities in a manifest, including adapter locations, cache indexes, hidden-state consumers and checkpoint keys. The old adapted block35 moves when earlier blocks are inserted. New layer count, streaming cache construction and export metadata must agree. A cached35-layer frozen prefix cannot train an all36-adapter or interleaved-expanded model; use a correct differentiated forward and activation checkpointing/sharding as needed.

Before optimization, require a **single bounded admission**, not a new open-ended diagnostic project: native prefill plus multi-frame streaming equivalence of old logits, hidden conditioning, language/control tokens and audio for the identity-expanded model; nonzero gradients on intended new parameters; actual fresh-process optimizer/RNG/OPAL restore. Preserve old parameter states and histories, initialize new moments/counters explicitly, and restore parent RNG after construction if initialization consumed random draws. Existing per-parameter optimizer steps and new-parameter steps must not be silently conflated.

Increasing LoRA rank while retaining its old tensors needs exact scaling/state handling. A separate zero-output new adapter branch can preserve old tensor identity and Adam history; rank-padding a tensor with one inherited scalar step counter is not automatically equivalent to independent new-parameter initialization. Do not merge/reset the old adapter or replay saved updates to simplify the migration.

## A meaningful optimization horizon

The current240 TRAIN/64 DEV corpus is a regression anchor, not evidence of adequate coverage for an approximately1B-parameter knowledge-acquisition run. Repeating it for thousands of updates is not a larger knowledge dataset. Identity expansion preserves existing knowledge; broad new knowledge needs much more varied evidence. A targeted duplex-control adaptation may need far less data than LLaMA Pro's continued pretraining, but that must be demonstrated by generalization.

A concrete **proposed first sustained interaction-training schedule** is a versioned5,000-episode TRAIN collection, roughly balanced across ordinary conversation, selective delegation, pending-work conversation, cancellation, deliberate silence and pause/resume, with matched simple-versus-hard tasks and varied timing/overlap/speakers/accents. Hold out scenario templates, voices and paraphrase families; retain the existing frozen DEV/held-out collections. Five thousand is an operational starting target, not a proven sufficiency threshold or authorization for new speech calls. Use approved synthetic sources; preserve source words, quality receipts and hashes.

At an effective batch of8 episodes,1–3 full epochs yield approximately625–1,875 updates. Predeclare that sustained horizon and evaluate at initialization and roughly quarter-epoch boundaries, rather than declaring failure after8 updates or running repeated disconnected pilots. Log actual supervised tokens/audio duration, family losses, gradient/update norms, inherited/new-group learning rates and optimizer steps. A larger microbatch is a throughput choice; effective batch and exposure must be explicit. Select two credible LR/scaling settings during a bounded calibration, then carry the selected recipe into the sustained schedule with its history intact. Paper LRs are starting evidence, not permission to copy text-only settings into a speech model.

One arm at a time is sufficient operationally. If all36 rank256 already generalizes well, no obligation remains to spend on rank512 or new blocks. If train loss improves but held-out behavior deteriorates, adjust data/retention/recipe; if an adequate rank256 schedule remains update-constrained, test the next capacity arm. State the decision once at the predeclared checkpoints.

## Losses and product decision

Retain approved native action/temporal objectives, actual-history components, KL/hidden retention, old teacher histories, ordinary rehearsal and complete corpus membership. Score complete family cohorts, native audio words/peak/EOS, timing and actual asynchronous task events; do not adopt from training/dev text loss alone.

The speech decoder remains frozen under these language-only variants. Language hidden states condition speech, so broader language adaptation can affect pronunciation and naturalness without directly training the acoustic head. More LLM parameters therefore do not establish a fix for a frozen speech-head defect. Natural British voice training, if required, needs an explicitly appropriate acoustic/conditioning objective and paired data, not merely a larger language loss.

Select a winner only with whole held-out behavior, speech, ordinary retention and fresh-process restoration plus **unprofiled live latency/interruption/async delegation**. Six extra blocks add16.7% to decoder block count and KV-cache layers; this is not a prediction of end-to-end latency. Unmerged large-rank adapters also have runtime costs. Dense unfreezing adds no inference layers. A new structural variant is worth keeping only if its measured product gain justifies the live latency/memory cost.
