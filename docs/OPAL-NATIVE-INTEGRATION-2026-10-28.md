# Actual OPAL integration, without another paid training experiment

Status: OPAL is now integrated into the native trainer as an explicit opt-in.
Actual train-step keep equivalence and distinct-process full-state restoration
pass on a tiny CPU Qwen fixture. No paid optimizer agent or native9B OPAL run has
started; native GPU qualification is still required before adaptive adoption.

Reviewed unmodified upstream commit
`8b16759a3e9930916ed582e28068267859f935f8` (OPAL0.3.8), checked out at
`/private/tmp/duplex-opal-upstream-20261028`. OPAL asks an agent to rewrite
`updater.py` and `analysis.py`, including learning rates, schedules, momentum,
precision and update algorithms. The model, data, loss, batch size, update count
and final evaluation remain outside that optimizer edit contract.
[Pinned README](https://github.com/xingjian-bai/opal/blob/8b16759a3e9930916ed582e28068267859f935f8/README.md)

## What actually passed

The package declares PyTorch≥2.5, whereas our isolated proof environment is2.4.
That metadata difference is **not a demonstrated blocker to its core wrapper**:
the actual, unmodified upstream implementation passed these CPU tests on2.4:

- 100 updates on a tiny frozen linear layer with LoRA: ordinary AdamW and OPAL
 `backend="keep"` matched every loss, final parameter and Adam tensor exactly.
 The plain loop's clipping moved inside OPAL; its warmup/cosine scheduler remained
 unchanged.
- An actual generated updater edit changed weight decay at update4, creating
 programv001. This was a deterministic offline code-writing fixture, not an LLM
 discovering a better algorithm.
- Both programs were checkpointed halfway and resumed in **distinct Python
 processes**. Final model, Adam, scheduler, program/version/hash and continuation
 losses matched; Python/NumPy/Torch RNG probes matched after restoration.

Receipt: `runs/venus-training-20261028/opal-upstream-offline-v3/proof.json`.
The first sandbox attempt retained a shared-memory child-process failure; it was
rerun with normal shared-memory access, without CPU/thread limits. The final
explicit-RNG version isv3. GPU seconds0, API calls0, native candidate updates0.
These are tiny CPU integration proofs, **not** native9B/GPU qualification or
evidence that OPAL improves voice behavior.
[Declared requirements](https://github.com/xingjian-bai/opal/blob/8b16759a3e9930916ed582e28068267859f935f8/pyproject.toml)

## Reusable implementation

`training/venus/opal_adapter.py` imports only that exact clean checkout and rejects
a different package named `opal`. It defaults to the free keep backend; a remote
mailbox requires an explicit finite positive agent allowance. It uses
`blocking=False`, finite decision/unavailable timeouts, an explicit short decision
schedule, `share_code=False`, and no paid report summarizer. It records source
hashes for the updater and analysis program, requires complete wrapped checkpoint
state, and admits only finite numeric TRAIN/DEV aggregates to agent-visible metrics.

The official mailbox supplies run/attempt identities, exclusive request claims
and atomic replies. Credentials/broker stay on the Mac; code proposals and
aggregate metrics cross the existing verified connection. Its SSH broker needs
an existing supported SSH alias; our long IAP argument list is not itself a valid
`--ssh` hostname. No broker was launched. A USD limit still requires trustworthy
broker cost reporting and the project ledger; Codex token reports alone do not
enforce a USD ceiling.
[Mailbox source](https://github.com/xingjian-bai/opal/blob/8b16759a3e9930916ed582e28068267859f935f8/opal/backends/mailbox.py),
[checkpoint source](https://github.com/xingjian-bai/opal/blob/8b16759a3e9930916ed582e28068267859f935f8/opal/misc/checkpoint.py)

## Native integration completed; remaining GPU qualification

Implemented in `training/venus/train.py` and `training/venus/native_optimizer.py`:

- Restore the exact saved Adam/RNG first; wrap only after complete-epoch throughput
  admission. The next OPAL continuation accepts the exact saved244 checkpoint and
  records total lineage244, so it need not repeat190→244.
- Use the equivalent warmup/cosine scheduler. The driver stops writing LR on every
  step; upstream OPAL can own its schedule. Gradient norm is observed without
  mutation and clipping runs exactly once, inside the updater.
- Pass the actual mean accumulated objective (control/text CE plus the unchanged
  ordinary KL/hidden terms) to `step(loss=...)`. Dataset and loss stay unchanged.
- Save the complete OPAL optimizer/program/observations/memory/scheduler state in
  ordinary atomic native checkpoints. Independent continuation compares these
  fields as well as weights, Adam, RNG and loss; replayed update indices fail.
- Full DEV aggregate losses enter OPAL; no held-out answers or owner content.
  Bounded reactive control and OPAL cannot both own the optimizer.
- `--opal-source /exact/pinned/checkout` opts in; default backend is free `keep`.
  A mailbox backend requires an explicit positive cost allowance. No broker or
  external API agent was launched. Live/evaluation-only leases reject training.

Actual proof: `runs/venus-training-20261028/opal-native-trainstep-v2/proof.json`.
ParentPID97687 and childPID97705 used the real native `train_step`/checkpoint seam
with a tiny CPU Qwen fixture; weights, Adam, scheduler, program, observations,
RNG and continuation objective match. Six accumulated updates also match the
plain trainer exactly, including inherited Adam moments and frozen weights.
The v1 restricted child failed in OpenMP shared memory; v2 used normal shared
memory with no CPU/thread throttling. Latest focused suite:58PASS.

Remaining: repeat keep/restore on the actual native GPU/cache before allowing an
adaptive run; add fixed native DEV behavior measurements because loss alone
missed current speech failures. The frozen GPU runtime stays unchanged. No new
native candidate updates, voice generation, GPU seconds or API calls were used
for the CPU proof. This does not establish an OPAL voice-quality improvement.

OPAL saves programs and pending observations in its optimizer state. Arbitrary
generated optimizer code executes with training-process permissions; it is not
a sandbox. Asynchronous application timing can differ between runs, so retain
the actual requested/applied update and program hashes. No dataset movement is
needed to change optimizer behavior.
[Pinned integration/restore guide](https://github.com/xingjian-bai/opal/blob/8b16759a3e9930916ed582e28068267859f935f8/docs/GUIDE.md)

Run the free proof with the existing proof Python:

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/duplex-venus-proof-20261027/bin/python -m training.venus.opal_smoke --source /private/tmp/duplex-opal-upstream-20261028 --output runs/venus-training-20261028/opal-upstream-offline-NEW
```

Choose a new output directory each time. `evals/harness/test_venus_opal_adapter.py`
has8 fast configuration/metric/checkpoint-boundary checks; they do not replace
the actual upstream parity/code-swap/process-restore proof.


## Continual dataset-versus-recipe reviews, 29 October

The native loop now logs per-family TRAIN decision/text cross-entropy, KL,
hidden retention and gradient norm each accumulated batch. Every eight updates
it writes an immutable diagnostic packet; each complete epoch runs all52 DEV
scenes and records the same decomposed losses. The exact approved source manifest
is required before model loading, so OPAL receives TRAIN/DEV scene and distinct
input counts. Counterfactual input reuse is identified as intentional matching,
not automatically classified as faulty data. No HOLDOUT answers enter the agent.

The supervisor compares four separate causes: data/targets/coverage, optimizer
and schedule dynamics, language-to-speech objective mismatch, and inference/UI
runtime. Lower TRAIN loss with worse DEV loss raises a possible coverage or
regularization issue; it does not establish causality. Native DEV speech and
control tests at phase boundaries remain necessary. Change one justified factor
at a time and retain the requested/applied optimizer program and hashes. A data
or target repair creates a new version between phases; it never silently changes
the currently hashed corpus. This is ongoing supervision, not continuous full
speech evaluation during every update.

`--maximum-epochs 1` can admit exactly one complete216-scene epoch (54 accumulated
candidate updates), retaining the full development/evaluation reserve. This
provides a native speech checkpoint before a further training phase rather than
replaying the earlier190-to244 updates. Recipe admission still refuses a partial
corpus. OPAL may adapt the optimizer within this schedule; it cannot extend the
approved update/time limits.

58 focused optimizer/training/diagnostic/TTS-attention tests passed, plus26
native-app/browser/transport/conditioning tests. Distinct-process proof
`runs/venus-training-20261029/opal-native-trainstep-v4/proof.json` now also proves
exact restoration of diagnostic history (parent48415, child48426). The v3
restricted shared-memory failure remains intact; v4 used normal shared memory,
without CPU or thread limits. These are CPU integration proofs, not native9B
GPU qualification. No paid broker/Jev agent, new API call, native adaptive update
or OPAL performance gain is claimed.


## Current-agent Git supervisor and actual mailbox proof, 29 October

`training/venus/opal_supervisor.py` runs the unmodified upstream mailbox broker
with a local Git supervisor. The current LOCAL agent reads its request and
TRAIN/DEV evidence, then writes a request-bound keep decision or commits an
optimizer/analysis program under `training/venus/opal_programs/`. No secondary
agent, Jev process or provider API is launched by this backend.

An adapt decision names the exact request ID/hash, rationale, Git commit and
program paths. The broker reads committed bytes rather than the working tree.
Only updater.py and analysis.py can change through this interface; data, loss,
folds, update/time limits and adoption remain outside it. For a remote run,
source must first be pushed/fetched through the normal Git deployment. The
broker verifies the same program bytes in the training host's Git object before
returning them through OPAL. A cancelled/expired decision cannot apply after
that verification. Syntax checks do not make arbitrary Python safe: program
review and upstream trials are still required, and this is trusted optimizer
code, not a sandbox.

The process-local SSH adapter reuses the controller's exact strict IAP argv.
It requires strict host keys, identities-only and noninteractive authentication;
it creates no SSH alias, credential or network resource. It restores the
upstream Folder class on exit and leaves the pinned checkout unmodified. The
caller still must verify the actual provider lease, cost guard and source
before starting the remote broker. No live remote SSH/IAP test was performed
while the GPU remained stopped.

Actual local mailbox proof PASS: six tests, persistent receipt
`runs/venus-training-20261029/opal-supervisor-proof-v4/proof.json`. The actual
native train_step on tiny CPU Qwen requested a decision at update16, continued
through18 while the request was unanswered, then applied a Git-committed updater
at19. Upstream events ended at programv001, and the applied code SHA matches the
committed proposal. This deterministic fixture changes weight decay; it is not
an agent discovering a better recipe or evidence of voice improvement. Tests
also reject stale attempts, uncommitted edits, missing remote Git deployment and
cancellation during verification. GPU seconds0, provider calls0, charge0,
real native candidate updates0. Earlier proof folders remain intact; v1 had a
fixture-parent setup error and v3 used the wrong exception-class assertion.

The wrapper's existing positive-budget admission uses a0.01USD metadata ceiling
for this proof; the Git supervisor always reports actual cost0 and makes no
service call. Upstream's generic Claude cost estimate is not an incurred charge
or the backend used. Real deployment must retain a finite decision schedule and
all existing project limits; this is not permission to substitute a paid broker.

Observation cadence remains each batch, diagnostic review each8 updates, full52
DEV each epoch. The current native54-update phase has optimizer decision
boundaries16 and32; logging every8 does not mean code is rewritten every8.
Keep is a valid decision when the evidence does not justify a change. Root
supervision must make the actual data/recipe/objective/runtime judgement; the
transport neither decides causality nor silently changes the corpus.
