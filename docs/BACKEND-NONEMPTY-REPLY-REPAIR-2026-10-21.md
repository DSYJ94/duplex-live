# Nonempty backend reply constraint

The preserved `native-v2-development-7582080_7/worker-events.json` contains an
actual gpt-6-luna lookup of the identified Willow North room marker followed by
a completed message `{"route":"delegate","content":""}`. The final response used
16 output tokens and reported completed status. Existing delivery validation
correctly rejected it: no verified result span was injected. The original whole
task remains failed and native counterfactual result-use coverage remains missing.

The outgoing result schema previously admitted that exact empty content. The
repair adds the supported string pattern `\S` to both structured final content
and the `respond_to_user` action. It requires at least one nonwhitespace character.
It changes no tool operation, execution count, receipt validation, cancellation,
revision guard, provider model, retry policy or local fail-closed content guard.

The official [Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas)
lists `pattern` as a supported string constraint for base models; its fine-tuned
model exclusion does not apply to the current base-model provider. No new model,
credential or GPU service is introduced.

Verification: the actual `run_worker` payload seam test failed for all three
protocols before the repair, then passed after it. It checks the final structured
schema and direct reply tool, accepts ordinary nonempty text, and verifies one
actual in-memory lookup receipt and two transport requests. All46 relevant worker,
replacement, dispatcher, backend-reference and backend-failure tests pass locally.
The existing test still rejects a blank custom/provider reply after real execution.
These are offline transport-contract tests, not proof of API acceptance, provider
reliability or native speech behavior. A live API check and complete native review
are still required before using the revised backend in qualification.

The in-flight training7590052 and watcher7589159 stay pinned to741872f. This fix
is not hot-patched into either process or its eventual native10 evaluation.

Actual live validation completed21Oct00:02:43UTC in CPU step7589159.0, immutable
source49b9d013e6532cf62238aeab2aaa503d64b964cd: six synthetic cases (ordinary and
private under each protocol) all pass, ten provider requests, three actual
synthetic read receipts, no retries. The API accepts the constraint. Complete
287-test CPU suite independently passes with CUDA uninitialized in a separate
step of the same existing CPU allocation. No new Slurm or GPU job was submitted.
Both reports and complete CPU log are independently size/SHA verified on the Mac:
3files77664bytes. This is provider and software evidence, not native qualification.
