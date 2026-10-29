# Native-selected asynchronous reasoning bridge

Implemented, not deployed or provider-qualified. The native model still owns
speech, listening and delegation decisions. The default live server remains the
arithmetic diagnostic until explicitly configured otherwise. Candidate244 is
unchanged and rejected for product adoption; this bridge is not evidence that
its routing or voice quality has improved.

`portable/venus/reasoning.py` supplies a bounded asynchronous worker. A parsed
native `<delegate>` request starts work without blocking subsequent audio. The
worker binds each task to its own identity, revision, request hash and provider
response identity. Only one validated completed semantic action can become a
`<backend>` receipt. Responses cannot terminate that context delimiter. Ready
receipts are injected once; this records context queuing, not proof that the
voice model read the answer correctly.

The reasoner receives the native model's selected task description and relevant
pending task descriptions. These are explicitly **not verified USER
transcripts**. It must clarify missing premises rather than invent them. The
next native evaluation must check query faithfulness as well as solved answers;
a stronger backend cannot fix a missing or misheard premise by itself. No ASR
router, canned ordinary answer or keyword mute has been added.

Cancellation and correction select exact still-pending task IDs. Every target
must remain valid before any target changes; unrelated work is retained.
Cancelled, superseded and disconnected completions are suppressed. Disconnect
returns the voice slot immediately while bounded HTTP work finishes. Transport or semantic failures produce an explicit failure receipt for the
native context, containing the error type and no fabricated answer. There is
no automatic retry or claim that already-injected content can be recalled. This foreground implementation cancels delivery, not provider compute.
The official [background-mode guide](https://developers.openai.com/api/docs/guides/background)
describes provider-managed background response cancellation; this bridge does
not enable that different transport.

## Local credential boundary

`portable/context/local_responses_proxy.py` now has a separate
`--profile venus-reasoning`. Its original context-lookup profile remains the
default and rejects the new payload. The new profile accepts only the exact
bounded builder: `gpt-6-astra`, 4096 output-token limit, store=false, one of three
fixed semantic functions and explicit selected-task provenance. Arbitrary model,
tool, storage or metadata changes are rejected. This uses the existing Mac-only
provider key path and fixed official HTTPS endpoint; no key is copied to GCP.

The GPU client accepts only an explicit loopback URL over the existing SSH
connection and a matching execution ID. It ignores environment HTTP proxies,
rejects redirects, and bounds response size and the async wait. A conversation
allows at most six requests. The Mac proxy independently limits request count
and deadline. Its receipts retain hashes, IDs, status and measured token counts;
request/answer text and credentials are not written. Those limits are not a
USD cost estimator: real execution still requires an admitted budget and ledger.

The native server's opt-in arguments are `--reasoning-proxy` and
`--reasoning-run-id`. The UI distinguishes a configured reasoning bridge from
the arithmetic diagnostic. No running service has these arguments yet. Before
launching, verify the exact SSH forwarding and proxy health, approved synthetic
payload scope, source commit, key locality and real request allowance. The
pending four-hour GPU proposal explicitly allows zero new API calls; it does
not silently authorize this bridge's paid provider tests or owner microphone
content. No provider credential was read or real request sent during this work.

## Evidence and remaining acceptance

Ten new worker/proxy cases passed. Together with the affected existing transport,
native-app and browser checks, the final suite passed35 tests plus6 subtests. Tests cover independent pending jobs, selective cancellation,
correction/revision, stale targets, disconnected replies, bounded failures,
request limits, delimiter safety and credential/content boundaries.

One test traverses the real FastAPI WebSocket handler, incremental native delegate
parser, asynchronous worker, actual loopback HTTP server, fixture provider and
native context-injection seam. Audio frames continue while the first HTTP call
is pending; a second semantic call cancels it; the cancellation receipt is
injected once and the deliberately late original result is suppressed. Provider
replies and the native engine in that test are fixtures, explicitly classified
as such. This proves the application wiring, not native inference, external
provider correctness, real-world cancellation timing or owner voice acceptance.

Still required: admitted real synthetic provider calls; real native routing and
query/answer fidelity; ordinary follow-ups while work runs; cancellation and
correction under actual latency; complete paired speech/control evaluation and
owner listening. No GPU time, API spend or native optimizer updates were added.
