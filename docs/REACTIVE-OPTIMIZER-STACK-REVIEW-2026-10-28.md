# Reactive optimizer stack review — 28 October 2026

Decision: use the existing coding agent as the training supervisor now. Jev can later serve as an optional narrow decision layer; it cannot replace the agent that reasons about training and writes optimizer code. A separate API agent is not required merely to apply the idea. This is an implementation recommendation, not evidence that our model has improved.

## What the research establishes

OPAL puts an agent inside a run: it writes `analysis.py` to inspect training dynamics and `updater.py` to change parameter updates. Its experiments include learning-rate, optimizer, momentum and precision changes. These are results on other tasks, not validation for our 851,968-parameter Venus adapter. The paper acknowledges smaller benefits on short, well-tuned runs and additional run-to-run uncertainty. [Paper, sections 2–4](https://arxiv.org/html/2610.33826)

The repository currently reports version 0.3.8. Integration wraps an existing optimizer with `opal.wrap`, supplies the loss to `step`, and retains the scheduler; `backend="keep"` is the no-agent equivalence mode. It requires PyTorch ≥2.5 on Python below3.13. Our pinned Venus runtime uses2.4, so installing it into the current environment would introduce an unnecessary runtime change before training. The README also reports short runs finishing before a nonblocking agent's first answer. [README](https://github.com/xingjian-bai/opal), [package requirements](https://github.com/xingjian-bai/opal/blob/main/pyproject.toml), [version](https://github.com/xingjian-bai/opal/blob/main/opal/_version.py)

OPAL supports asynchronous decisions and a Mac-side mailbox broker, so credentials need not move to the GPU. Mailboxes use run/attempt identities and exclusive claims. Agent waiting needs an explicit bound: unavailable-agent waiting is otherwise indefinite. Save program version/state alongside model, optimizer and update position. Failed builds/first updates can be restored from snapshots, but later trial failures are not always undone exactly. Generated optimizer code executes with training-process permissions; its boundary scanner records findings rather than enforcing isolation. [Integration and recovery guide](https://github.com/xingjian-bai/opal/blob/main/docs/GUIDE.md)

## Jev and “System One” are not two equivalent coding agents

TypeSafe calls Jev its first System One model. It returns typed decisions rather than generated prose. The official coding-agent guide expressly says it does not write code, converse, or replace the generative model behind a coding agent. [Introduction](https://docs.typesafe.ai/introduction), [coding-agent distinction](https://docs.typesafe.ai/introduction/coding-agents)

It could judge a small host-prepared state: whether a failure description suggests a label/contract mismatch, whether a DEV example answers its question, or which preapproved diagnostic to request. These are proposed uses requiring calibration. Arithmetic, deadlines, slope calculations and learning-rate values stay in code. TypeSafe documents numeric weakness, indirect-reasoning weakness and choice-order sensitivity. Confidence does not establish optimizer expertise. [Known limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

Our existing Jev integration pins `jev-1.13.0`; this is a source configuration, not a live model-availability or account-access check. No key was read and no paid API request was made for this review. Existing old notes claiming a blanket closed-output training ban are superseded by ADR0003's provenance classes; provenance and current transfer/spend limits still apply. [Jev codec](/Users/daryl/Library/CloudStorage/OneDrive-Personal/Shared%20GitHub%20Repos/SystemOne-Voice-ComputerUse/packages/providers/src/jev-wire.ts), [current provenance decision](../decisions/0003-provenance-classes.md)

## The local SystemOne project's model is another, unqualified option

The separate SystemOne computer-use repository is an application/framework, not itself a model. Its pending owned CLM stack declares `Qwen/Qwen3-8B` encoder revision `b968826d9c46dd6066d109eabc6255188de91218` with `Contrastive-LM/CLM-v0.1-8B` heads revision `e939398d4556fcd9400c76fa8c5a513202f42b0a`. It ranks offered candidates. Its current boundary docs say full encoder/live calibration and owner-app qualification remain incomplete; nothing inspected establishes training-optimization competence. Read-only checkout HEAD: `12b6e8af7c73759eb1148fa1fd2b2b7e1979a254`. [Reference implementation](/Users/daryl/Library/CloudStorage/OneDrive-Personal/Shared%20GitHub%20Repos/SystemOne-Voice-ComputerUse/docs/CLM_REFERENCE_ADAPTER.md), [decision boundary](/Users/daryl/Library/CloudStorage/OneDrive-Personal/Shared%20GitHub%20Repos/SystemOne-Voice-ComputerUse/docs/CLM_DECISION_BOUNDARY.md)

Reuse its design pattern—typed proposals, candidate identities, stale-reply rejection and separate host authority—without loading another8B model or adding desktop automation. Adopting its checkpoint for optimizer decisions would first require task-specific evaluation; that should not block the next Venus training run.

## Apply the idea to the actual bottleneck

The latest run was blocked by time admission after successful cache preparation, before an optimizer update. No reactive learning-rate policy could have repaired that scheduling failure. Reuse the complete saved caches, admit a whole training epoch before allocation, and reserve measured evaluation/copyback time.

For the next bounded continuation, inspect training/DEV cross-entropy by family, ordinary KL/hidden retention, gradient norm, update-to-weight size, clipping frequency, throughput and remaining budget. Let a bounded agent decision adjust only explicitly admitted optimizer controls at update boundaries. It must neither inspect HOLDOUT labels nor remove examples, weaken acceptance, expand trainable weights, or reset deadlines. Preserve the complete216TRAIN/52DEV and all objectives.

Checkpoint the decision journal and effective controls with Adam/RNG, then verify a distinct-process restore. On a short run, consult only at meaningful intervals and continue the last valid policy while a response is pending. Jev is optional; start with the current agent and deterministic numerical calculations. Full OPAL code rewriting can be a later versioned experiment if this simpler loop proves insufficient. Native behaviour, actual reasoning delegation and natural voice remain separate product outcomes: lower training loss alone is not adoption.
