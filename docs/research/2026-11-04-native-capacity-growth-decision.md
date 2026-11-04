# Native duplex capacity: growth is possible; necessity is unproven

2026-11-04. Research and decision note only. No training, architecture change, spending authority or candidate adoption.

The current evidence supports fixing specific execution and adaptation failures before expanding the foundation. The completed 24-case comparison fixed a cut-off spoken answer without updating weights; candidate speech/text consistency changed from 4/6 to 5/6. All paired language/control traces stayed identical. Three repaired-cohort recordings still fail the unchanged speech gate, and no general intelligence score follows from this six-input diagnosis. [Measured local result](../NATIVE-VENUS-TTS-COMPLETION-RUNTIME-COMPAT-RESULT-2026-11-04.json).

## Four different changes

| Change | What actually grows | Meaning for this project |
| --- | --- | --- |
| Train broader/higher-rank LoRA | Adapter parameter count and allowed weight-update subspace | Cheapest capacity question to test first. It does not add transformer blocks or pretrained knowledge. |
| Unfreeze existing weights | Number of trainable parameters and optimizer state; total model size stays the same | Can adapt more of the existing network; requires stronger retention and speech checks. |
| Insert blocks or widen the network | Foundation parameter count and usually inference work | Mathematically feasible, but creates a new architecture needing continued training and native validation. |
| Delegate to an external reasoning model | Capacity of the whole system, without enlarging the live model | Fits the goal of a responsive native voice model with a stronger asynchronous reasoning backend. |

LoRA freezes the original weights and learns low-rank updates. For a matrix, `W' = W + BA`; increasing rank increases possible update rank, rather than increasing every underlying transformer dimension. This is distinct from full fine-tuning. [LoRA paper](https://arxiv.org/abs/2106.09685).

Our existing one-layer rank64 q/v adapter has 851,968 trainable parameters. The prepared four-layer variant has 3,407,872 and the conditional eight-layer option 6,815,744. These apply adapters to **existing** layers; they are not new base layers. Four-layer state mapping has only a CPU preparation proof, not native continuation qualification or training. [Capacity preparation](../NATIVE-VENUS-CAPACITY-COMPARISON-PREPARATION-2026-11-02.json), [state preparation](../NATIVE-VENUS-CAPACITY-STATE-PREPARATION-2026-11-02.json).

## Can we grow a trained foundation?

Yes. Net2Net established function-preserving widening/deepening transformations, carrying an existing network's learned function into a larger one. Such transformations depend on architecture assumptions; they are not a drop-in proof for this transformer's attention, normalization and speech pathway. [Net2Net, Google Research](https://research.google/pubs/net2net-accelerating-learning-via-knowledge-transfer/).

LLaMA Pro demonstrates transformer block expansion: its authors initialize added blocks as identity mappings, freeze inherited blocks, and train added blocks on code/math data, expanding a LLaMA2-7B foundation to 8.3B. It is evidence that growth can work, not evidence that our duplex model will benefit. [LLaMA Pro](https://arxiv.org/html/2401.02415v2).

An identity residual insertion illustrates the mathematics: `h_new = h + alpha*F(h)` starts as `h_new = h` when `alpha=0`. Extra capacity initially preserves the old function; new knowledge still requires useful data, an effective optimization schedule and evaluation. Arbitrary random layers can damage behavior. Repeatedly adding parameters does not automatically approach a much larger model's pretrained abilities.

Venus combines native speech interaction with asynchronous delegation; its public project separates the conversational model and execution harness. Expanding the language backbone may alter the hidden states conditioning speech and increases the work required per interaction. **Project inference:** language accuracy, waveform fidelity, interruption timing, KV/cache compatibility and restore behavior must all be validated after growth. No text-only expansion paper proves duplex preservation. [Official Venus repository](https://github.com/inclusionAI/Realtime-Venus). The current public README is not a substitute for our pinned checkpoint/runtime evidence.

## Recommended decision

Do not grow the foundation to address the demonstrated acoustic cut-off: the runtime repair already addresses that cause. Complete bounded review of the remaining frozen speech failures and evaluate the end-to-end product behaviors.

If persistent learning failures remain after data/recipe checks, the first capacity experiment should be **one-layer versus four-layer LoRA**, using the already prepared mechanism. Compare a preserved common parent, the same complete training corpus/folds/minibatch order and training budget, native initialization equivalence, inherited Adam/RNG/OPAL state, and full held-out behavior/speech/retention and live latency checks. Predeclare the decision and stopping point; loss improvement alone is insufficient. Native memory/throughput and paid execution still need explicit scope.

Only a reproducible advantage from broader adaptation, or evidence that the existing base itself cannot meet the product requirements, justifies escalating to unfreezing, structural growth or a different foundation. Heavy reasoning can use the external backend meanwhile. There is no parameter-growth or full-weight run in progress, and this note authorizes none.
