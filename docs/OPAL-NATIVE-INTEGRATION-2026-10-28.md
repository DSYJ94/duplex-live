# Actual OPAL integration, without another paid training experiment

Status: optional upstream seam and free CPU proof implemented; the native trainer
has **not** yet been wrapped and no paid optimizer agent was started.

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
state, and admits only finite numeric DEV summaries to agent-visible metrics.

The official mailbox supplies run/attempt identities, exclusive request claims
and atomic replies. Credentials/broker stay on the Mac; code proposals and
aggregate metrics cross the existing verified connection. Its SSH broker needs
an existing supported SSH alias; our long IAP argument list is not itself a valid
`--ssh` hostname. No broker was launched. A USD limit still requires trustworthy
broker cost reporting and the project ledger; Codex token reports alone do not
enforce a USD ceiling.
[Mailbox source](https://github.com/xingjian-bai/opal/blob/8b16759a3e9930916ed582e28068267859f935f8/opal/backends/mailbox.py),
[checkpoint source](https://github.com/xingjian-bai/opal/blob/8b16759a3e9930916ed582e28068267859f935f8/opal/misc/checkpoint.py)

## Small remaining native integration

1. Restore the original Adam state **before** wrapping. Replace the trainer's
 per-step manual LR assignments with an equivalent scheduler so they do not
 overwrite an agent-owned schedule. Move clipping into OPAL once, not twice.
2. Pass the actual accumulated weighted loss to `step(loss=...)`. Preserve the
 full corpus, decision/text balancing and ordinary KL/hidden objectives.
3. Save/load the full wrapped state, including code and analysis memory. The
 trainer's current Adam-only `state['state']` restore comparison must compare
 OPAL's underlying Adam states **and** program/version/observation state.
4. Run native keep equivalence and a distinct-process native restore on the same
 cached inputs, before any paid adaptive extension. The existing pinned GPU
 runtime stays untouched until this path is checked.
5. Expose full DEV losses plus small fixed native **DEV** behavior probes. Lower
 prediction loss alone missed our current speech failure. Final held-out
 evaluation stays separate; the agent never gets its answers.

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
